"""Benchmark of TAG-MTL-DR v2 against reproduced baselines under one protocol.

Every method -- baselines, the proposed model and its ablations -- is trained and
evaluated on the same patient-grouped folds, seeds, cache, budget and external
datasets. Per-image predictions (temperature-calibrated on the validation patients)
are written to ``predictions.csv``; ``analysis.py`` turns them into the paper tables.

Speed: one pre-processing cache shared by all methods (``build_cache``), GPU resizing
and normalisation, mixed precision, channels-last memory, persistent workers.

Manifest (CSV): ``image,grade,patient,dataset,role[,mask]`` (role = internal | external)

Usage
-----
    python benchmark.py --manifest manifest.csv --cache cache/ --out runs/main \
        --methods all --seeds 42 43 44 --folds 5 --epochs 30
    python analysis.py --runs runs/main
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tag_mtl_dr import (NUM_CLASSES, CachedFundus, CoralHead, TAGMTLDR, build_cache, ce_mse_loss, class_probs,
                        collate, coral_loss, gpu_preprocess, metrics, patient_folds, seg_loss)

# kind: "cnn" = timm backbone + head; "tag" = TAG-MTL-DR v2 with ablation switches
METHODS = {
    "ResNet-50":               {"kind": "cnn", "backbone": "resnet50", "head": "ce", "size": 300},
    "DenseNet-121":            {"kind": "cnn", "backbone": "densenet121", "head": "ce", "size": 300},
    "EfficientNet-B3":         {"kind": "cnn", "backbone": "efficientnet_b3", "head": "ce", "size": 300},
    "EfficientNet-B3 + CORAL": {"kind": "cnn", "backbone": "efficientnet_b3", "head": "coral", "size": 300},
    "ConvNeXt-T":              {"kind": "cnn", "backbone": "convnext_tiny", "head": "ce", "size": 288},
    "ViT-B/16":                {"kind": "cnn", "backbone": "vit_base_patch16_224", "head": "ce", "size": 224},
    "Swin-T":                  {"kind": "cnn", "backbone": "swin_tiny_patch4_window7_224", "head": "ce", "size": 224},
    "TAG-MTL-DR (ours)":       {"kind": "tag", "size": 300},
    "w/o vessel-path GRU":     {"kind": "tag", "size": 300, "use_paths": False},
    "w/o lesion-aware nodes":  {"kind": "tag", "size": 300, "lesion_aware": False},
    "w/o vessel graph":        {"kind": "tag", "size": 300, "use_graph": False},
    "w/o segmentation task":   {"kind": "tag", "size": 300, "use_seg": False},
    "concat instead of cross-attention": {"kind": "tag", "size": 300, "fusion": "concat"},
    "CE + MSE instead of CORAL": {"kind": "tag", "size": 300, "head": "ce_mse"},
}
BASELINES = ["ResNet-50", "DenseNet-121", "EfficientNet-B3", "EfficientNet-B3 + CORAL",
             "ConvNeXt-T", "ViT-B/16", "Swin-T"]
ABLATIONS = ["w/o vessel-path GRU", "w/o lesion-aware nodes", "w/o vessel graph", "w/o segmentation task",
             "concat instead of cross-attention", "CE + MSE instead of CORAL"]


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


def head_of(spec):
    return spec.get("head", "coral")


def build(spec, pretrained, backbone_override=None):
    if spec["kind"] == "cnn":
        return CNNClassifier(backbone_override or spec["backbone"], spec["head"], pretrained)
    return TAGMTLDR(pretrained=pretrained, head="coral" if head_of(spec) == "coral" else "linear",
                    backbone=backbone_override or "efficientnet_b3",
                    use_graph=spec.get("use_graph", True), use_paths=spec.get("use_paths", True),
                    lesion_aware=spec.get("lesion_aware", True), fusion=spec.get("fusion", "cross"),
                    use_seg=spec.get("use_seg", True))


def loss_fn(spec, logits, y, vlog, m):
    head = head_of(spec)
    loss = (coral_loss(logits, y) if head == "coral" else ce_mse_loss(logits, y) if head == "ce_mse"
            else F.cross_entropy(logits, y))
    if vlog is not None:
        loss = loss + spec.get("lam_seg", 0.5) * seg_loss(vlog, m)
    return loss


def read_manifest(path):
    rows = list(csv.DictReader(open(path, newline="")))
    need = {"image", "grade", "patient", "dataset", "role"}
    if not rows or not need <= set(rows[0]):
        raise ValueError(f"manifest needs the columns {sorted(need)}")
    for r in rows:
        r["grade"] = int(r["grade"])
        if r["role"] not in ("internal", "external"):
            raise ValueError(f"role must be internal|external: {r}")
    return rows


def ensure_cache(rows, cache_dir, size, workers):
    cache = Path(cache_dir)
    meta = cache / "index.json"
    if meta.exists():
        m = json.load(open(meta))
        if m["images"] == [r["image"] for r in rows] and m["size"] == size:
            return cache
        raise SystemExit(f"{cache} was built for another manifest/size; use a new --cache folder")
    print(f"building the pre-processing cache in {cache} (once for all methods) ...", flush=True)
    t0 = time.time()
    build_cache([(r["image"], r.get("mask") or None) for r in rows], cache, size, workers)
    print(f"cache built in {time.time() - t0:.0f} s", flush=True)
    return cache


def loader(cache, rows, idx, spec, train, args, seed):
    tag = spec["kind"] == "tag"
    ds = CachedFundus(cache, idx, [rows[i]["grade"] for i in idx], augment=train,
                      need_mask=tag and spec.get("use_seg", True), need_graph=tag and spec.get("use_graph", True))
    sampler = None
    if train:                                                       # class-balanced sampling
        g = np.array([rows[i]["grade"] for i in idx])
        w = 1.0 / np.maximum(np.bincount(g, minlength=NUM_CLASSES), 1)[g]
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(idx),
                                                         generator=torch.Generator().manual_seed(seed))
    return torch.utils.data.DataLoader(ds, batch_size=args.batch, sampler=sampler, shuffle=False,
                                       num_workers=args.workers, persistent_workers=args.workers > 0,
                                       pin_memory=torch.cuda.is_available(), collate_fn=collate,
                                       drop_last=train and len(idx) > args.batch)


@torch.no_grad()
def predict_logits(model, dl, spec, device, size):
    model.eval()
    L, n, t = [], 0, 0.0
    for x, _, _, g in dl:
        x = gpu_preprocess(x, size, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits, _, _ = model(x, g)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t += time.time() - t0
        n += x.size(0)
        L.append(logits.float().cpu())
    return torch.cat(L), 1000 * t / max(n, 1)


def fit_temperature(logits, y, head):
    """Temperature minimising the validation negative log-likelihood (grid search)."""
    y = torch.as_tensor(y)
    best_T, best = 1.0, float("inf")
    for T in np.linspace(0.5, 3.0, 26):
        P = class_probs(logits, head, float(T)).clamp_min(1e-7)
        nll = -torch.log(P[torch.arange(len(y)), y]).mean().item()
        if nll < best:
            best, best_T = nll, float(T)
    return best_T


def run(cache, rows, spec, train_idx, val_idx, eval_sets, args, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    size = args.size_override or spec["size"]
    model = build(spec, not args.no_pretrained, args.backbone_override).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    tr = loader(cache, rows, train_idx, spec, True, args, seed)
    va = loader(cache, rows, val_idx, spec, False, args, seed)
    yv = np.array([rows[i]["grade"] for i in val_idx])
    best, best_state, wait, t_train = -np.inf, None, 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        for x, y, m, g in tr:
            x, y = gpu_preprocess(x, size, device), y.to(device)
            m = m.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits, vlog, _ = model(x, g)
            loss = loss_fn(spec, logits.float(), y, None if vlog is None else vlog.float(), m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        lv, _ = predict_logits(model, va, spec, device, size)
        q = metrics(yv, class_probs(lv, head_of(spec)).argmax(1).numpy())["qwk"]
        if q > best + 1e-4:
            best, wait = q, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    t_train = time.time() - t_train
    model.load_state_dict(best_state)
    if args.save_models:                                            # for explain.py and later inference
        ck = Path(args.out) / "checkpoints"
        ck.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ck / f"{spec['name'].replace('/', '_').replace(' ', '_')}_s{seed}.pt")
    lv, _ = predict_logits(model, va, spec, device, size)
    T = fit_temperature(lv, yv, head_of(spec))                     # calibration on validation patients only
    out, ms = {}, []
    for split, idx in eval_sets.items():
        L, ms_img = predict_logits(model, loader(cache, rows, idx, spec, False, args, seed), spec, device, size)
        out[split] = (idx, class_probs(L, head_of(spec), T).numpy())
        ms.append(ms_img)
    info = {"epochs": epoch + 1, "best_val_qwk": float(best), "temperature": T,
            "params": int(sum(p.numel() for p in model.parameters())),
            "train_seconds": round(t_train, 1), "infer_ms_per_image": float(np.mean(ms))}
    return out, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=["all"], help="'all', 'baselines', 'ablations' or names")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache_size", type=int, default=300)
    ap.add_argument("--train_fraction", type=float, default=1.0,
                    help="use a patient-level fraction of the training data (data-efficiency curve)")
    ap.add_argument("--save_models", action="store_true", help="save the best checkpoint of every run")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--backbone_override", default=None, help="smoke tests only")
    ap.add_argument("--size_override", type=int, default=None, help="smoke tests only")
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
    cache = ensure_cache(rows, args.cache, args.cache_size, args.workers)
    internal = np.array([i for i, r in enumerate(rows) if r["role"] == "internal"])
    externals = sorted({r["dataset"] for r in rows if r["role"] == "external"})
    grades = np.array([rows[i]["grade"] for i in internal])
    patients = np.array([rows[i]["patient"] for i in internal])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import sklearn, timm, torch_geometric
    meta = {"args": vars(args), "methods": {n: METHODS[n] for n in names}, "runs": [],
            "n_internal": int(len(internal)), "n_internal_patients": int(len(set(patients))),
            "externals": externals,
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "timm": timm.__version__, "torch_geometric": torch_geometric.__version__,
                            "sklearn": sklearn.__version__,
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}}

    fpred = open(out / "predictions.csv", "w", newline="")
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
            if args.train_fraction < 1.0:                           # patient-level subsampling
                tp = np.array(sorted({rows[i]["patient"] for i in tr}))
                keep = set(np.random.RandomState(seed + k).choice(tp, max(1, int(round(args.train_fraction * len(tp)))),
                                                                   replace=False))
                tr = np.array([i for i in tr if rows[i]["patient"] in keep])
            pt, pv, pe = ({rows[i]["patient"] for i in s} for s in (tr, va, te))
            assert not (pt & pe) and not (pv & pe) and not (pt & pv), "patient leakage"
            eval_sets = {"internal": te}
            for ds in externals:
                eval_sets[f"external:{ds}"] = np.array([i for i, r in enumerate(rows)
                                                        if r["role"] == "external" and r["dataset"] == ds])
            for name in names:
                spec = dict(METHODS[name], name=name)
                preds, info = run(cache, rows, spec, tr, va, eval_sets, args, seed * 100 + k, device)
                for split, (idx, P) in preds.items():
                    for i, p in zip(idx, P):
                        r = rows[i]
                        w.writerow([name, seed, k, split.split(":")[0], r["dataset"], r["image"], r["patient"],
                                    r["grade"], int(p.argmax()), *[f"{v:.6f}" for v in p], len(tr), len(te)])
                fpred.flush()
                info.update({"method": name, "seed": seed, "fold": k})
                meta["runs"].append(info)
                print(f"seed {seed} fold {k} {name:<36} val QWK {info['best_val_qwk']:.4f}  T={info['temperature']:.2f}  "
                      f"{info['train_seconds']} s  {info['infer_ms_per_image']:.1f} ms/img", flush=True)
                json.dump(meta, open(out / "runs.json", "w"), indent=1)
    fpred.close()
    print(f"wrote {out / 'predictions.csv'} and {out / 'runs.json'}")


if __name__ == "__main__":
    main()
