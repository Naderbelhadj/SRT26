"""Patient-level, inductive re-evaluation of TOPORET (first perspective of the thesis).

The released pipeline (``pipeline.py``) evaluates TOPORET with image-level
stratified folds and one population graph over all images, so the test images
are unlabelled nodes during training (transductive).  This script keeps the
same features, graph rule (alpha, tau) and models, and changes only the
protocol:

* ``--split patient``    folds are grouped by patient (StratifiedGroupKFold), so
                         the two eyes of a patient never fall in different folds;
* ``--graph inductive``  the graph is built from the training images of each
                         fold; validation and test images only *receive*
                         messages from training images (directed edges
                         train -> new), and BatchNorm statistics are computed
                         on training images only.

``--split image --graph transductive`` reproduces the released protocol, so the
2 x 2 design separates the fellow-eye effect from the transductive effect.

Usage
-----
    # 1) extract and cache features once per dataset
    python src/patient_level.py extract --dataset kaggle_dr --out features/kaggle_dr.npz
    python src/patient_level.py extract --dataset messidor2 --out features/messidor2.npz \
        --patient_map data/messidor2/patient_map.csv
    python src/patient_level.py extract --dataset aptos2019 --out features/aptos2019.npz

    # 2) evaluate baseline and TOPORET, patient-level and inductive, five seeds
    python src/patient_level.py evaluate --features features/kaggle_dr.npz \
        --split patient --graph inductive --seeds 42 43 44 45 46 \
        --out results/patient_level/kaggle_dr

    # 3) H1 statistics (paired t, Nadeau-Bengio, Fisher) over all datasets
    python src/stats_h1.py results/patient_level/*/folds.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("patient_level")

VARIANTS = {
    "CNN_only":      {"use_tda": False, "use_graph": False},
    "CNN_TDA":       {"use_tda": True,  "use_graph": False},
    "CNN_Graph":     {"use_tda": False, "use_graph": True},
    "CNN_TDA_Graph": {"use_tda": True,  "use_graph": True},   # full TOPORET
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ----------------------------------------------------------------------------- patients
def patient_id(dataset: str, path: Path, mapping: dict | None) -> str:
    """Patient identifier of an image.

    kaggle_dr : ``10_left.jpeg`` -> ``10`` (both eyes share the id)
    aptos2019 : one image per patient, the file stem is the id
    other     : read from ``--patient_map`` (columns ``image,patient``)
    """
    stem = path.stem
    if mapping is not None:
        key = stem if stem in mapping else path.name
        if key not in mapping:
            raise KeyError(f"{path.name} missing from the patient map")
        return mapping[key]
    if dataset == "kaggle_dr":
        m = re.match(r"^(\d+)_(left|right)$", stem, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"unexpected Kaggle DR file name: {path.name}")
        return m.group(1)
    if dataset == "aptos2019":
        return stem
    raise ValueError(f"dataset '{dataset}' needs --patient_map (image,patient)")


def read_patient_map(path: str | None) -> dict | None:
    if not path:
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"image", "patient"} <= set(rows[0]):
        raise ValueError("patient map must have the columns 'image' and 'patient'")
    out = {}
    for r in rows:
        out[Path(r["image"]).stem] = r["patient"]
        out[Path(r["image"]).name] = r["patient"]
    return out


# ----------------------------------------------------------------------------- extraction
def extract(args: argparse.Namespace) -> None:
    """Compute CNN and TDA features once and cache them with labels and patient ids."""
    import cv2
    import yaml
    from cnn_features import extract_cnn_features
    from preprocessing import preprocess_fundus
    from skeletonisation import extract_skeleton, get_branch_points
    from tda_features import batch_tda_features

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mapping = read_patient_map(args.patient_map)
    root = Path(cfg["data"]["root"]) / args.dataset / "train"
    if not root.exists():
        raise FileNotFoundError(f"{root} not found; see data/README_data.md")

    paths, labels = [], []
    for grade_dir in sorted(root.iterdir()):
        if grade_dir.is_dir() and grade_dir.name.isdigit():
            for fp in sorted(grade_dir.iterdir()):
                if fp.suffix.lower() in IMG_EXTS:
                    paths.append(fp)
                    labels.append(int(grade_dir.name))
    labels = np.asarray(labels, dtype=np.int64)
    patients = np.asarray([patient_id(args.dataset, p, mapping) for p in paths])
    logger.info(f"{len(paths)} images, {len(set(patients))} patients, grades {np.bincount(labels).tolist()}")

    pre = cfg["preprocessing"]
    preprocessed, branch_points = [], []
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"cannot read {p}")
        pp = preprocess_fundus(img, clahe_clip=pre["clahe_clip_limit"],
                               clahe_tile=tuple(pre["clahe_tile_size"]),
                               tophat_radius=pre["tophat_radius"],
                               closing_radius=pre["closing_radius"],
                               image_size=cfg["data"]["image_size"])
        skel = extract_skeleton(pp, min_component_px=pre["min_component_px"])
        preprocessed.append(pp)
        branch_points.append(get_branch_points(skel))
        if (i + 1) % 1000 == 0:
            logger.info(f"  preprocessed {i + 1}/{len(paths)}")

    tda = batch_tda_features(branch_points,
                             persistence_threshold=cfg["tda"]["persistence_threshold"])
    cnn = extract_cnn_features(preprocessed, labels.tolist(),
                               image_size=cfg["data"]["image_size"],
                               batch_size=cfg["training"]["batch_size"],
                               num_workers=cfg["data"]["num_workers"],
                               pretrained=cfg["backbone"]["pretrained"],
                               device=cfg["hardware"]["device"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, cnn=np.asarray(cnn, np.float32), tda=np.asarray(tda, np.float32),
                        labels=labels, patients=patients,
                        files=np.asarray([p.name for p in paths]), dataset=args.dataset)
    logger.info(f"saved {args.out}")


# ----------------------------------------------------------------------------- folds
def make_folds(labels: np.ndarray, patients: np.ndarray, split: str, n_splits: int,
               seed: int, val_frac: float = 0.125) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(train, val, test) index triples, 70/10/20 as in the released code.

    With ``split='patient'`` every patient falls entirely in one of the three sets.
    """
    from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                         StratifiedKFold)

    idx = np.arange(len(labels))
    if split == "patient":
        outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = outer.split(idx, labels, groups=patients)
    else:
        outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = outer.split(idx, labels)

    folds = []
    for k, (tv, test) in enumerate(splits, start=1):
        if split == "patient":
            inner = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed + k)
            tr_rel, va_rel = next(inner.split(tv, labels[tv], groups=patients[tv]))
            train, val = tv[tr_rel], tv[va_rel]
        else:  # identical to train.run_cross_validation
            rng = np.random.RandomState(seed + k)
            tv = tv.copy()
            rng.shuffle(tv)
            n_val = max(1, int(val_frac * len(tv)))
            val, train = tv[:n_val], tv[n_val:]
        folds.append((np.sort(train), np.sort(val), np.sort(test)))
    return folds


def check_disjoint_patients(patients: np.ndarray, train, val, test) -> int:
    """Number of patients shared between the three sets (0 for a patient-level split)."""
    a, b, c = set(patients[train]), set(patients[val]), set(patients[test])
    return len(a & b) + len(a & c) + len(b & c)


# ----------------------------------------------------------------------------- graph
def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def graph_edges(cnn: np.ndarray, tda: np.ndarray, sources: np.ndarray, targets: np.ndarray,
                alpha: float, tau: float, d_max: float, block: int = 1024) -> np.ndarray:
    """Directed edges source -> target with S_ij > tau (no self-loops).

    S_ij = alpha * clip(cos(cnn_i, cnn_j), 0, 1) + (1 - alpha) * exp(-d_ij / d_max),
    d_ij = Euclidean distance between l2-normalised TDA vectors, as in
    ``similarity_graph.compound_similarity_matrix``.  Computed block-wise, so the
    N x N matrix is never stored.
    """
    v, t = _l2(cnn), _l2(tda)
    vs, ts = v[sources], t[sources]
    rows, cols = [], []
    for s in range(0, len(targets), block):
        tg = targets[s:s + block]
        cos_v = np.clip(v[tg] @ vs.T, 0.0, 1.0)
        cos_t = np.clip(t[tg] @ ts.T, -1.0, 1.0)
        d = np.sqrt(np.maximum(2.0 - 2.0 * cos_t, 0.0))
        S = alpha * cos_v + (1.0 - alpha) * np.exp(-d / d_max)
        r, c = np.nonzero(S > tau)
        keep = tg[r] != sources[c]
        rows.append(tg[r][keep])
        cols.append(sources[c][keep])
    tgt = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    src = np.concatenate(cols) if cols else np.zeros(0, np.int64)
    return np.stack([src, tgt]).astype(np.int64)          # PyG: row 0 = source


def max_tda_distance(tda: np.ndarray, idx: np.ndarray, block: int = 2048) -> float:
    """d_max over the given images (training images for the inductive graph)."""
    t = _l2(tda[idx])
    m = 0.0
    for s in range(0, len(t), block):
        cos_t = np.clip(t[s:s + block] @ t.T, -1.0, 1.0)
        m = max(m, float(np.sqrt(np.maximum(2.0 - 2.0 * cos_t, 0.0)).max()))
    return m if m > 1e-8 else 1.0


# ----------------------------------------------------------------------------- training
def fold_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    def nr(g):
        m = y == g
        return float((pred[m] <= 1).mean()) if m.any() else float("nan")
    return {"accuracy": float(accuracy_score(y, pred)),
            "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "sev_nr": nr(3), "pdr_nr": nr(4),
            "n_severe": int((y == 3).sum()), "n_pdr": int((y == 4).sum())}


def run_fold(X, labels, cnn, tda, train, val, test, variant, graph_mode, cfg, device, seed):
    import torch
    import torch.nn as nn
    from graphsage_model import build_model
    from train import compute_class_weights

    torch.manual_seed(seed)
    np.random.seed(seed)
    use_graph = VARIANTS[variant]["use_graph"]
    alpha, tau = cfg["graph"]["alpha"], cfg["graph"]["tau"]
    n = len(labels)

    if graph_mode == "transductive":   # released protocol: every node in the training forward pass
        allidx = np.arange(n)
        E_full = (graph_edges(cnn, tda, allidx, allidx, alpha, tau, max_tda_distance(tda, allidx))
                  if use_graph else np.zeros((2, 0), np.int64))
        E_train_local = None
    elif use_graph:
        d_max = max_tda_distance(tda, train)
        E_tt = graph_edges(cnn, tda, train, train, alpha, tau, d_max)
        new = np.concatenate([val, test])
        E_tn = graph_edges(cnn, tda, train, new, alpha, tau, d_max)   # train -> new only
        E_full = np.concatenate([E_tt, E_tn], axis=1)
        remap = -np.ones(n, np.int64)
        remap[train] = np.arange(len(train))
        E_train_local = remap[E_tt]
    else:
        E_full = np.zeros((2, 0), np.int64)
        E_train_local = np.zeros((2, 0), np.int64)

    Xt = torch.from_numpy(X).float().to(device)
    yt = torch.from_numpy(labels).long().to(device)
    Ef = torch.from_numpy(E_full).long().to(device)
    tr = torch.from_numpy(train).long().to(device)
    va = torch.from_numpy(val).long().to(device)

    model = build_model(use_tda=VARIANTS[variant]["use_tda"], use_graph=use_graph,
                        hidden_dim=cfg["model"]["hidden_dim"],
                        dropout=cfg["model"]["dropout"]).to(device)
    tc = cfg["training"]
    crit = nn.CrossEntropyLoss(weight=compute_class_weights(labels[train], 5, device))
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["max_epochs"], eta_min=tc["lr_min"])

    if E_train_local is None:                       # transductive: all nodes in the forward pass
        def train_logits():
            return model(Xt, Ef)[tr]
    else:                                           # inductive: training subgraph only
        Xtr = Xt[tr]
        Etr = torch.from_numpy(E_train_local).long().to(device)
        def train_logits():
            return model(Xtr, Etr)

    best, best_state, wait = float("inf"), None, 0
    for epoch in range(tc["max_epochs"]):
        model.train()
        opt.zero_grad()
        loss = crit(train_logits(), yt[tr])
        loss.backward()
        opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xt, Ef)[va], yt[va]).item()
        if vloss < best - 1e-5:
            best, wait = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= tc["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xt, Ef)[torch.from_numpy(test).long().to(device)].argmax(1).cpu().numpy()
    out = fold_metrics(labels[test], pred)
    out.update({"n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
                "n_edges": int(E_full.shape[1]), "epochs": epoch + 1})
    return out


def evaluate(args: argparse.Namespace) -> None:
    import torch
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data = np.load(args.features, allow_pickle=False)
    cnn, tda, labels, patients = data["cnn"], data["tda"], data["labels"], data["patients"]
    dataset = str(data["dataset"])
    dev = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    device = torch.device(dev)
    logger.info(f"{dataset}: {len(labels)} images, {len(set(patients))} patients, "
                f"split={args.split}, graph={args.graph}, seeds={args.seeds}, device={dev}")

    records = []
    for seed in args.seeds:
        random.seed(seed)
        folds = make_folds(labels, patients, args.split, cfg["data"]["n_splits"], seed)
        for k, (train, val, test) in enumerate(folds, start=1):
            shared = check_disjoint_patients(patients, train, val, test)
            if args.split == "patient" and shared:
                raise AssertionError(f"seed {seed} fold {k}: {shared} patients shared across sets")
            for variant in args.variants:
                X = np.concatenate([cnn, tda], 1) if VARIANTS[variant]["use_tda"] else cnn
                t0 = time.time()
                m = run_fold(X, labels, cnn, tda, train, val, test, variant, args.graph, cfg,
                             device, seed * 100 + k)
                m.update({"dataset": dataset, "seed": seed, "fold": k, "variant": variant,
                          "split": args.split, "graph": args.graph, "shared_patients": shared,
                          "seconds": round(time.time() - t0, 1)})
                records.append(m)
                logger.info(f"seed {seed} fold {k} {variant:<14} QWK={m['qwk']:.4f} "
                            f"acc={m['accuracy']:.4f} Sev-NR={m['sev_nr']:.4f} edges={m['n_edges']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "folds.json", "w") as f:
        json.dump({"dataset": dataset, "split": args.split, "graph": args.graph,
                   "config": {"alpha": cfg["graph"]["alpha"], "tau": cfg["graph"]["tau"]},
                   "records": records}, f, indent=1)
    logger.info(f"wrote {out / 'folds.json'}")


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="compute and cache CNN/TDA features")
    e.add_argument("--config", default="configs/default_config.yaml")
    e.add_argument("--dataset", required=True, choices=["kaggle_dr", "messidor2", "aptos2019"])
    e.add_argument("--patient_map", default=None, help="CSV with columns image,patient")
    e.add_argument("--out", required=True)
    v = sub.add_parser("evaluate", help="cross-validate the variants on cached features")
    v.add_argument("--config", default="configs/default_config.yaml")
    v.add_argument("--features", required=True)
    v.add_argument("--split", choices=["patient", "image"], default="patient")
    v.add_argument("--graph", choices=["inductive", "transductive"], default="inductive")
    v.add_argument("--variants", nargs="+", default=["CNN_only", "CNN_TDA_Graph"], choices=list(VARIANTS))
    v.add_argument("--seeds", nargs="+", type=int, default=[42])
    v.add_argument("--device", default="auto")
    v.add_argument("--out", required=True)
    args = p.parse_args()
    extract(args) if args.cmd == "extract" else evaluate(args)


if __name__ == "__main__":
    main()
