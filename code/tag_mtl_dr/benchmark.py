"""Benchmark of TAG-MTL-DR against reproduced baselines under one protocol.

Every method -- baselines, the proposed model and its ablations -- is trained and
evaluated on the same patient-grouped folds, seeds, image pipeline and external
datasets. The script writes per-image predictions; ``analysis.py`` turns them into
the comparison, ablation, external-validation and per-grade tables of the paper.
No number is entered by hand.

Manifest (CSV): ``image,grade,patient,dataset,role[,mask]``
    role = internal (cross-validated) | external (never used for training)

Usage
-----
    python benchmark.py --manifest data/manifest.csv --out runs/main \
        --methods all --seeds 42 43 44 --folds 5 --epochs 30
    python analysis.py --runs runs/main --reference "TAG-MTL-DR (ours)"
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tag_mtl_dr import (NUM_CLASSES, CoralHead, FundusDataset, TAGMTLDR, ce_mse_loss, collate,
                        coral_loss, metrics, patient_folds, seg_loss)

# ----------------------------------------------------------------------------- methods
# kind: "cnn" = timm backbone + head; "tag" = TAG-MTL-DR (with ablation switches)
METHODS = {
    # reproduced baselines (same protocol)
    "ResNet-50":               {"kind": "cnn", "backbone": "resnet50", "head": "ce", "size": 300},
    "DenseNet-121":            {"kind": "cnn", "backbone": "densenet121", "head": "ce", "size": 300},
    "EfficientNet-B3":         {"kind": "cnn", "backbone": "efficientnet_b3", "head": "ce", "size": 300},
    "EfficientNet-B3 + CORAL": {"kind": "cnn", "backbone": "efficientnet_b3", "head": "coral", "size": 300},
    "ConvNeXt-T":              {"kind": "cnn", "backbone": "convnext_tiny", "head": "ce", "size": 288},
    "ViT-B/16":                {"kind": "cnn", "backbone": "vit_base_patch16_224", "head": "ce", "size": 224},
    "Swin-T":                  {"kind": "cnn", "backbone": "swin_tiny_patch4_window7_224", "head": "ce", "size": 224},
    # proposed model
    "TAG-MTL-DR (ours)":       {"kind": "tag", "size": 300},
    # ablations of the proposed model
    "w/o vessel graph":        {"kind": "tag", "size": 300, "use_graph": False},
    "w/o segmentation loss":   {"kind": "tag", "size": 300, "lam_seg": 0.0},
    "concat instead of cross-attention": {"kind": "tag", "size": 300, "fusion": "concat"},
    "CE + MSE instead of CORAL": {"kind": "tag", "size": 300, "head": "ce_mse"},
}
BASELINES = ["ResNet-50", "DenseNet-121", "EfficientNet-B3", "EfficientNet-B3 + CORAL",
             "ConvNeXt-T", "ViT-B/16", "Swin-T"]
ABLATIONS = ["w/o vessel graph", "w/o segmentation loss", "concat instead of cross-attention",
             "CE + MSE instead of CORAL"]


class CNNClassifier(nn.Module):
    def __init__(self, backbone, head, pretrained):
        super().__init__()
        import timm
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = self.encoder.num_features
        self.drop = nn.Dropout(0.3)
        self.head = CoralHead(d) if head == "coral" else nn.Linear(d, NUM_CLASSES)

    def forward(self, x, graphs=None):
        return self.head(self.drop(self.encoder(x))), None, None


def build(spec, pretrained, backbone_override=None):
    if spec["kind"] == "cnn":
        return CNNClassifier(backbone_override or spec["backbone"], spec["head"], pretrained)
    head = spec.get("head", "coral")
    return TAGMTLDR(pretrained=pretrained, head="coral" if head == "coral" else "linear",
                    use_graph=spec.get("use_graph", True), fusion=spec.get("fusion", "cross"),
                    backbone=backbone_override or "efficientnet_b3")


def head_of(spec):
    return spec.get("head", "coral")


def class_probs(logits, head):
    """(B, 5) class probabilities for every head type."""
    if head == "coral":
        s = torch.sigmoid(logits)                                   # P(y > k), k = 0..3
        s = torch.cummin(s, dim=1).values                           # enforce monotonicity
        ones, zeros = torch.ones_like(s[:, :1]), torch.zeros_like(s[:, :1])
        cum = torch.cat([ones, s, zeros], 1)
        return (cum[:, :-1] - cum[:, 1:]).clamp_min(0)
    return logits.softmax(1)


def loss_fn(spec, logits, y, vlog, m):
    head = head_of(spec)
    if head == "coral":
        loss = coral_loss(logits, y)
    elif head == "ce_mse":
        loss = ce_mse_loss(logits, y)
    else:
        loss = F.cross_entropy(logits, y)
    if spec["kind"] == "tag":
        loss = loss + spec.get("lam_seg", 0.5) * seg_loss(vlog, m)
    return loss


# ----------------------------------------------------------------------------- data
def read_manifest(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    need = {"image", "grade", "patient", "dataset", "role"}
    if not rows or not need <= set(rows[0]):
        raise ValueError(f"manifest needs the columns {sorted(need)}")
    for r in rows:
        r["grade"] = int(r["grade"])
        if r["role"] not in ("internal", "external"):
            raise ValueError(f"role must be internal|external: {r}")
    return rows


def loader(rows, idx, spec, train, bs, workers, seed):
    items = [(rows[i]["image"], rows[i]["grade"], rows[i].get("mask") or None) for i in idx]
    ds = FundusDataset(items, size=spec["size"], augment=train, with_mask=spec["kind"] == "tag")
    sampler = None
    if train:                                                       # class-balanced sampling
        g = np.array([rows[i]["grade"] for i in idx])
        w = 1.0 / np.bincount(g, minlength=NUM_CLASSES)[g]
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(idx),
                                                         generator=torch.Generator().manual_seed(seed))
    return torch.utils.data.DataLoader(ds, batch_size=bs, sampler=sampler, shuffle=False,
                                       num_workers=workers, collate_fn=collate, drop_last=train and len(idx) > bs)


@torch.no_grad()
def predict(model, dl, spec, device):
    model.eval()
    P = []
    for x, _, _, _ in dl:
        logits, _, _ = model(x.to(device))
        P.append(class_probs(logits.float(), head_of(spec)).cpu())
    return torch.cat(P).numpy()


# ----------------------------------------------------------------------------- one run
def run(rows, spec, name, train_idx, val_idx, eval_sets, args, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(spec, not args.no_pretrained, args.backbone_override).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    tr = loader(rows, train_idx, spec, True, args.batch, args.workers, seed)
    va = loader(rows, val_idx, spec, False, args.batch, args.workers, seed)
    best, best_state, wait = -np.inf, None, 0
    yv = np.array([rows[i]["grade"] for i in val_idx])
    for epoch in range(args.epochs):
        model.train()
        for x, y, m, g in tr:
            x, y, m = x.to(device), y.to(device), m.to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits, vlog, _ = model(x, g)
                loss = loss_fn(spec, logits.float(), y, None if vlog is None else vlog.float(), m)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        q = metrics(yv, predict(model, va, spec, device).argmax(1))["qwk"]   # model selection on validation QWK
        if q > best + 1e-4:
            best, wait = q, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    out = {}
    for split, idx in eval_sets.items():
        out[split] = (idx, predict(model, loader(rows, idx, spec, False, args.batch, args.workers, seed), spec, device))
    n_params = sum(p.numel() for p in model.parameters())
    return out, {"epochs": epoch + 1, "best_val_qwk": float(best), "params": n_params}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=["all"], help="'all', 'baselines', 'ablations' or names")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--backbone_override", default=None, help="for smoke tests only")
    ap.add_argument("--size_override", type=int, default=None, help="for smoke tests only")
    args = ap.parse_args()

    names = []
    for m in args.methods:
        names += (list(METHODS) if m == "all" else BASELINES if m == "baselines"
                  else ABLATIONS if m == "ablations" else [m])
    unknown = [n for n in names if n not in METHODS]
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}; choose from {list(METHODS)}")
    names = list(dict.fromkeys(names))

    rows = read_manifest(args.manifest)
    internal = np.array([i for i, r in enumerate(rows) if r["role"] == "internal"])
    externals = sorted({r["dataset"] for r in rows if r["role"] == "external"})
    grades = np.array([rows[i]["grade"] for i in internal])
    patients = np.array([rows[i]["patient"] for i in internal])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"args": vars(args), "methods": {n: METHODS[n] for n in names}, "runs": [],
            "n_internal": int(len(internal)), "externals": externals}

    pred_path = out / "predictions.csv"
    fpred = open(pred_path, "w", newline="")
    w = csv.writer(fpred)
    w.writerow(["method", "seed", "fold", "split", "dataset", "image", "patient", "y", "pred",
                "p0", "p1", "p2", "p3", "p4", "n_train", "n_test"])
    from sklearn.model_selection import GroupShuffleSplit
    for seed in args.seeds:
        for k, (tv_rel, te_rel) in enumerate(patient_folds(grades, patients, args.folds, seed), start=1):
            tv, te = internal[tv_rel], internal[te_rel]
            gss = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed + k)
            tr_rel, va_rel = next(gss.split(tv, groups=[rows[i]["patient"] for i in tv]))
            tr, va = tv[tr_rel], tv[va_rel]
            assert not ({rows[i]["patient"] for i in tr} & {rows[i]["patient"] for i in te})
            eval_sets = {"internal": te}
            for ds in externals:
                eval_sets[f"external:{ds}"] = np.array([i for i, r in enumerate(rows)
                                                        if r["role"] == "external" and r["dataset"] == ds])
            for name in names:
                spec = dict(METHODS[name])
                if args.size_override:
                    spec["size"] = args.size_override
                t0 = time.time()
                preds, info = run(rows, spec, name, tr, va, eval_sets, args, seed * 100 + k, device)
                for split, (idx, P) in preds.items():
                    for i, p in zip(idx, P):
                        r = rows[i]
                        w.writerow([name, seed, k, split.split(":")[0], r["dataset"], r["image"], r["patient"],
                                    r["grade"], int(p.argmax()), *[f"{v:.6f}" for v in p], len(tr), len(te)])
                fpred.flush()
                info.update({"method": name, "seed": seed, "fold": k, "seconds": round(time.time() - t0, 1)})
                meta["runs"].append(info)
                print(f"seed {seed} fold {k} {name:<36} val QWK {info['best_val_qwk']:.4f} "
                      f"({info['seconds']} s)", flush=True)
                with open(out / "runs.json", "w") as f:
                    json.dump(meta, f, indent=1)
    fpred.close()
    print(f"wrote {pred_path} and {out / 'runs.json'}")


if __name__ == "__main__":
    main()
