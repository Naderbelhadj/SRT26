# Q1 readiness audit — TAG-MTL-DR v2

What strong medical-imaging journals check (CLAIM, TRIPOD+AI, reviewers' usual
requests), and where each item stands. **Code** = implemented and tested here;
**You** = only you can do it (data, runs, writing); nothing below is a result.

## A. Novelty and method

| Requirement | Status | Where |
|---|---|---|
| A clear, testable methodological contribution | Code | Lesion-aware vessel graph (CNN features sampled at GAT nodes) + optic-disc-rooted vessel-path BiGRU + cross-attention + CORAL + auxiliary segmentation (`tag_mtl_dr.py`) |
| Every claimed component isolated by an ablation | Code | 6 ablations in `benchmark.py` (paths, lesion-aware nodes, graph, segmentation, fusion, ordinal head) |
| Novelty verified against the literature | **You** | Search PubMed / Scopus / IEEE / arXiv for "vessel graph" + "recurrent" / "GRU" + "diabetic retinopathy" and for CNN features sampled on vessel graphs before claiming "first" |
| Architecture figure | **You** | To draw from the description in `paper/main.tex` |

## B. Data and protocol

| Requirement | Status | Where |
|---|---|---|
| Patient-level splits, leakage assertion | Code | `patient_folds`, `GroupShuffleSplit`, `assert ... "patient leakage"` |
| External datasets never used for training, model selection or calibration | Code | `role=external`; temperature fitted on validation patients only |
| Several seeds, mean ± SD | Code | `--seeds` |
| Same budget and pre-processing for every method | Code | one cache, one loop, same epochs/patience/optimiser |
| Data description (images *and* patients, grades, devices, countries) | **You** | `runs.json` stores image and patient counts |
| Annotated vessel masks (DRIVE/FIVES) rather than pseudo-labels | **You** (recommended) | `mask` column of the manifest |
| Licences / ethics statement | **You** | public datasets: cite their licences |

## C. Evaluation and statistics

| Requirement | Status | Where |
|---|---|---|
| QWK, accuracy, macro-F1, AUC (referable) | Code | `analysis.py` |
| Safety: Sev-NR, PDR-NR with exact CIs, 0↔4 confusions | Code | `analysis.py` |
| Calibration: temperature scaling, ECE, Brier, reliability diagram | Code | `benchmark.py`, `analysis.py` |
| Patient-level bootstrap CIs | Code | `patient_bootstrap` |
| CV-appropriate test + multiplicity correction | Code | corrected repeated k-fold t-test, Holm |
| External: paired test and ΔQWK CI | Code | McNemar exact, paired bootstrap |
| Per-grade results, confusion matrix | Code | `table_pergrade.tex`, `fig_confusion.pdf` |
| Computational cost (params, training time, ms/image) | Code | `table_cost.tex` |
| Data-efficiency curve | Code (runs: **You**) | `--train_fraction 0.1 0.25 0.5 1.0`, one output folder each |
| Primary endpoint fixed in advance | **You** | write it down (e.g. external QWK and Sev-NR) before running |

## D. Comparison with the state of the art

| Requirement | Status | Where |
|---|---|---|
| Reproduced baselines, same protocol (CNN and transformer) | Code | ResNet-50, DenseNet-121, EfficientNet-B3 (+CORAL), ConvNeXt-T, ViT-B/16, Swin-T |
| Retinal foundation model (RETFound) | **You** | add its weights as a `timm`-style backbone once obtained from the authors; reviewers often ask for it |
| Literature values shown separately with sources | Code (values: **You**) | `literature_reported.csv` → `table_literature.tex` |

## E. Interpretability and failure analysis

| Requirement | Status | Where |
|---|---|---|
| Explanations linked to the model's mechanism | Code | `explain.py`: node attention on the vessel graph, lesion-token attention map, probabilities |
| Failure cases shown, not only successes | **You** | run `explain.py` on misclassified severe/PDR images |

## F. Reproducibility

| Requirement | Status | Where |
|---|---|---|
| Code, configuration, seeds, environment versions | Code | `runs.json` (args, environment, per-run info), `--save_models` |
| Unit tests | Code | `tests/`, `python tag_mtl_dr.py --smoke` |
| Public repository and data-availability statement | **You** | |

## G. Speed (what was changed)

* One pre-processing cache for all methods: resize, vessel mask, optic disc, graph and
  paths computed once in parallel (about 0.2 s of CPU per 300-px image removed from every
  epoch and every method).
* uint8 memory-mapped images; GPU resizing and normalisation; mixed precision;
  channels-last; persistent workers; cuDNN autotuning.
* `QUICK` mode in the Colab notebook for the free GPU.

## H. What would still weaken a Q1 submission

1. Pseudo-label vessel masks only (reviewers will question the graph quality): prefer
   annotated masks for the segmentation task, or report a mask-quality check on DRIVE/FIVES.
2. A single external dataset: two or more (e.g. Messidor-2 and IDRiD or EyePACS) are much stronger.
3. No RETFound comparison.
4. Claims of "first" without a documented literature search.
5. Reporting only the best seed or only successes.
