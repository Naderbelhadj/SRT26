"""Builds TAG_MTL_DR_Colab.ipynb from the tested source files (one notebook to upload in Colab)."""
import json
from pathlib import Path

HERE = Path(__file__).parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def writefile(name):
    return code(f"%%writefile {name}\n" + (HERE / name).read_text())


cells = [
    md("""
# TAG-MTL-DR — complete pipeline for Google Colab

Topology-aware graph + multi-task learning for diabetic retinopathy grading, with a
single-protocol benchmark (reproduced baselines + ablations), external validation,
statistics, LaTeX tables and figures for a journal article.

**How to use**
1. `Runtime > Change runtime type > GPU` (T4 or better).
2. Run the cells **in order**.
3. Set `DEMO = True` first: the notebook builds a tiny synthetic dataset and checks the
   whole chain in a few minutes. Its numbers are meaningless.
4. Then set `DEMO = False`, point `DATA_DIR` to your images on Google Drive, build the
   manifest (cell 5) and run the benchmark.

No result is included: every number comes from your own runs.
"""),
    md("## 1. Installation"),
    code("""
!pip -q install timm torch-geometric scikit-image scikit-learn scipy matplotlib pillow
import torch
print("PyTorch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (enable a GPU runtime)")
"""),
    md("## 2. Settings"),
    code("""
DEMO = True                     # True: synthetic check; False: your data
USE_DRIVE = not DEMO            # mount Google Drive to read images and save results

# --- paths (used when DEMO = False) ---
WORK_DIR = "/content/drive/MyDrive/TAG_MTL_DR"        # results are saved here
DATA_DIR = "/content/drive/MyDrive/datasets"          # where your images are

# --- experiment size ---
QUICK = True                    # QUICK: fewer methods/seeds for a free Colab GPU; False: full paper protocol
if QUICK:
    METHODS = ["EfficientNet-B3", "EfficientNet-B3 + CORAL", "ConvNeXt-T", "TAG-MTL-DR (ours)", "w/o vessel graph"]
    SEEDS, FOLDS, EPOCHS = [42], 5, 15
else:
    METHODS = ["all"]           # 7 baselines + ours + 4 ablations
    SEEDS, FOLDS, EPOCHS = [42, 43, 44], 5, 30
BATCH, WORKERS = 16, 2
"""),
    code("""
import os
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
else:
    WORK_DIR = "/content/tag_mtl_dr_demo"
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs("/content/code", exist_ok=True)
%cd /content/code
print("results will be written to", WORK_DIR)
"""),
    md("## 3. Source code (model, benchmark, analysis)\nThese cells write the three modules to `/content/code`."),
    writefile("tag_mtl_dr.py"),
    writefile("benchmark.py"),
    writefile("analysis.py"),
    code("""
!python tag_mtl_dr.py --smoke
"""),
    md("""
## 4. Data manifest

The benchmark reads a CSV with the columns `image,grade,patient,dataset,role[,mask]`:
* `role = internal` → cross-validated (patient-grouped folds);
* `role = external` → never used for training, only for evaluation;
* `patient` groups the two eyes of the same person (very important);
* `mask` (optional) → path of an annotated vessel mask; otherwise pseudo-labels are used.

The helpers below build it from the usual label files. Adapt the paths.
"""),
    code(r'''
import csv, glob, re
from pathlib import Path

def rows_from_label_csv(label_csv, image_dir, image_col, grade_col, dataset, role,
                        patient_fn=lambda stem: stem, ext=None):
    """One row per labelled image found on disk."""
    out, missing = [], 0
    with open(label_csv, newline="") as f:
        for r in csv.DictReader(f):
            stem = Path(r[image_col]).stem
            cands = [Path(image_dir) / r[image_col]] if ext is None else [Path(image_dir) / f"{stem}{ext}"]
            if ext is None and not cands[0].suffix:
                cands = [Path(image_dir) / f"{stem}{e}" for e in (".png", ".jpg", ".jpeg", ".tif")]
            path = next((c for c in cands if c.exists()), None)
            if path is None:
                missing += 1
                continue
            out.append({"image": str(path), "grade": int(float(r[grade_col])),
                        "patient": f"{dataset}:{patient_fn(stem)}", "dataset": dataset, "role": role})
    print(f"{dataset}: {len(out)} images, {len({o['patient'] for o in out})} patients, {missing} missing")
    return out

def write_manifest(rows, path):
    keys = ["image", "grade", "patient", "dataset", "role"] + (["mask"] if any("mask" in r for r in rows) else [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("manifest:", path, len(rows), "rows")

# patient identifiers
eyepacs_patient = lambda stem: re.match(r"^(\d+)_(left|right)$", stem).group(1)   # 10_left -> 10
one_image_per_patient = lambda stem: stem                                         # APTOS 2019
'''),
    code(r'''
MANIFEST = f"{WORK_DIR}/manifest.csv"

if DEMO:
    # tiny synthetic dataset (64 x 64 images) to check the pipeline -- numbers are meaningless
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    os.makedirs(f"{WORK_DIR}/img", exist_ok=True)
    def fake(g):
        a = rng.integers(0, 60, (64, 64, 3)).astype(np.uint8); a[..., 0] += np.uint8(30 * g)
        yy, xx = np.ogrid[:64, :64]; a[(abs(yy - 32) < 2) | (abs(xx - 20) < 2)] = 10
        return a
    rows = []
    for p in range(40):
        g = int(rng.integers(0, 5))
        for eye in "LR":
            f = f"{WORK_DIR}/img/A{p}{eye}.png"; Image.fromarray(fake(g)).save(f)
            rows.append({"image": f, "grade": g, "patient": f"A{p}", "dataset": "synthA", "role": "internal"})
    for p in range(20):
        g = int(rng.integers(0, 5)); f = f"{WORK_DIR}/img/B{p}.png"; Image.fromarray(fake(g)).save(f)
        rows.append({"image": f, "grade": g, "patient": f"B{p}", "dataset": "synthB", "role": "external"})
    write_manifest(rows, MANIFEST)
else:
    rows = []
    # --- EXAMPLES: uncomment and adapt to your folders ---
    # APTOS 2019 (one image per patient), internal:
    # rows += rows_from_label_csv(f"{DATA_DIR}/aptos2019/train.csv", f"{DATA_DIR}/aptos2019/train_images",
    #                             "id_code", "diagnosis", "APTOS2019", "internal", one_image_per_patient, ".png")
    # Kaggle EyePACS (two eyes per patient), external:
    # rows += rows_from_label_csv(f"{DATA_DIR}/eyepacs/trainLabels.csv", f"{DATA_DIR}/eyepacs/train",
    #                             "image", "level", "EyePACS", "external", eyepacs_patient, ".jpeg")
    # Messidor-2: provide a CSV image,grade,patient (patient = exam / person id), external:
    # rows += rows_from_label_csv(f"{DATA_DIR}/messidor2/labels.csv", f"{DATA_DIR}/messidor2/images",
    #                             "image", "grade", "Messidor2", "external", lambda s: s)
    assert rows, "add your datasets above"
    write_manifest(rows, MANIFEST)
'''),
    md("## 5. Benchmark\nEvery method uses the same patient-grouped folds, seeds, budget and pre-processing. "
       "Predictions are appended to `predictions.csv` after each run (safe if Colab disconnects: re-run with "
       "the remaining methods into another folder)."),
    code(r'''
RUNS = f"{WORK_DIR}/runs/main"
methods = " ".join(f'"{m}"' for m in METHODS)
seeds = " ".join(map(str, SEEDS))
extra = "--no_pretrained --backbone_override resnet18 --size_override 64 --epochs 2 --patience 2" if DEMO else f"--epochs {EPOCHS}"
folds = 2 if DEMO else FOLDS
!python benchmark.py --manifest "{MANIFEST}" --out "{RUNS}" --methods {methods} --seeds {seeds} --folds {folds} --batch {BATCH} --workers {WORKERS} {extra}
'''),
    md("## 6. Statistics, tables and figures\n"
       "Optional: upload `literature_reported.csv` (columns `method,year,citation_key,dataset,protocol,metric,value`) "
       "with values copied from papers **with their source**; it becomes a separate table."),
    code(r'''
LIT = f"{WORK_DIR}/literature_reported.csv"
lit_arg = f'--literature "{LIT}"' if os.path.exists(LIT) else ""
!python analysis.py --runs "{RUNS}" --reference "TAG-MTL-DR (ours)" {lit_arg}
'''),
    code(r'''
import json, pandas as pd
from IPython.display import display, IFrame
res = json.load(open(f"{RUNS}/paper/results.json"))
tab = pd.DataFrame({m: {"QWK mean": s["qwk"][0], "QWK sd": s["qwk"][1], "pooled QWK": s["qwk_pooled"],
                        "CI low": s["qwk_ci"][0], "CI high": s["qwk_ci"][1], "Sev-NR %": 100 * s["sev_nr"],
                        "ECE": s["ece"][0], "p Holm": s.get("p_holm")} for m, s in res["internal"].items()}).T
display(tab.round(3))
for ds, d in res["external"].items():
    print("External:", ds)
    display(pd.DataFrame({m: {"QWK": r["qwk"], "Sev-NR %": 100 * r["sev_nr"], "McNemar p": r.get("mcnemar_p")}
                          for m, r in d.items()}).T.round(3))
'''),
    code(r'''
from IPython.display import Image as Img
!pip -q install pdf2image > /dev/null; apt-get -qq install -y poppler-utils > /dev/null
from pdf2image import convert_from_path
for f in ["fig_qwk.pdf", "fig_confusion.pdf", "fig_reliability.pdf"]:
    convert_from_path(f"{RUNS}/paper/{f}", dpi=110)[0].save(f"/content/{f}.png"); display(Img(f"/content/{f}.png"))
print(open(f"{RUNS}/paper/table_sota.tex").read())
'''),
    md("## 7. Download everything\nThe ZIP contains the predictions, `results.json`, the LaTeX tables, the macros and the "
       "figures, ready for the paper skeleton (`paper/main.tex` of the repository)."),
    code(r'''
import shutil
zip_path = shutil.make_archive(f"/content/tag_mtl_dr_results", "zip", RUNS)
from google.colab import files
files.download(zip_path)
'''),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = HERE / "TAG_MTL_DR_Colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, len(cells), "cells")
