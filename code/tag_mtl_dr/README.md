# TAG-MTL-DR: model, single-protocol benchmark and journal paper

A new research pipeline, independent of the thesis. It contains everything needed to
produce the evidence of a journal article: the model, the reproduced baselines, the
ablations, external validation, statistics, LaTeX tables, figures and a paper skeleton.
**No result is included: every number is produced by your own runs.**

| File | Role |
|---|---|
| `tag_mtl_dr.py` | Model (EfficientNet-B3 lesion tokens, supervised U-Net, topological vessel graph + GAT, cross-attention, CORAL), losses, dataset, metrics |
| `benchmark.py` | Trains every method with the same patient-grouped folds, seeds, budget and input pipeline; predicts internal test folds and external datasets |
| `analysis.py` | Metrics, patient-bootstrap CIs, corrected repeated k-fold t-test + Holm, McNemar, Clopper-Pearson for Sev-NR/PDR-NR, ECE; writes the tables, figures and `results_macros.tex` |
| `literature_reported.csv` | Results *reported* by other papers, filled by you with a citation key and the protocol of each value; printed in a separate table |
| `paper/` | Article skeleton (`main.tex`) that inputs the generated tables and macros; reporting checklist |
| `tests/` | Unit tests of the statistics |

## Methods compared (one protocol)

* Reproduced baselines: ResNet-50, DenseNet-121, EfficientNet-B3, EfficientNet-B3 + CORAL,
  ConvNeXt-T, ViT-B/16, Swin-T (ImageNet weights via `timm`).
* Proposed: TAG-MTL-DR.
* Ablations: without vessel graph, without segmentation loss, concatenation instead of
  cross-attention, CE + MSE instead of CORAL.

To add a method, add an entry to `METHODS` in `benchmark.py` (any `timm` backbone).
Retinal foundation models (e.g. RETFound) can be added the same way once their weights are
obtained from the authors.

## Data manifest

`manifest.csv` with columns `image,grade,patient,dataset,role[,mask]`:
`role=internal` images are cross-validated; `role=external` images are never used for
training. `patient` groups the two eyes of a person. `mask` (optional) is an annotated
vessel mask; without it, classical pseudo-labels supervise the U-Net.

## Run

```bash
pip install timm torch-geometric scikit-image scikit-learn scipy matplotlib pillow
python tag_mtl_dr.py --smoke                       # model sanity check
python benchmark.py --manifest manifest.csv --out runs/main --methods all \
       --seeds 42 43 44 --folds 5 --epochs 30
python analysis.py --runs runs/main --literature literature_reported.csv
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The pipeline was checked end to end on a small synthetic dataset (tiny backbone, two
epochs): predictions, tables, macros, figures and the paper compile without errors.

## Rules that keep the paper defensible

1. The comparison table contains only methods **reproduced under this protocol**.
2. Values from other papers go to `literature_reported.csv` with their source and
   protocol, and appear in a separate table; they are never mixed with reproduced results.
3. Fix the primary endpoint (e.g. external QWK and Sev-NR) and the hypotheses before the
   runs, and report the result whatever it is.
4. Fill the red `[...]` placeholders of `paper/main.tex` only with statements the tables
   support.
