# TOPORET: patient-level, inductive re-evaluation

Add-on to the released TOPORET code (https://github.com/Nader-BelHadj/plosene).
It implements the first perspective of the thesis: test H1 with **patient-level
folds** and a **graph built from the training images of each fold**, over
**several seeds**, with the same features, graph rule (alpha = 0.6, tau = 0.65)
and models as the released pipeline.

## What changes, and what does not

| | Released `pipeline.py` | `patient_level.py` (default) |
|---|---|---|
| Folds | `StratifiedKFold`, image level | `StratifiedGroupKFold`, grouped by patient |
| Validation split | random 12.5% of train+val images | 12.5% of train+val **patients** |
| Graph | one graph over all images | per fold, from training images only |
| Test images during training | unlabelled nodes (transductive) | absent: they only receive messages from training images (train -> new) |
| BatchNorm statistics | all images | training images only |
| Seeds | one (42) | any list, e.g. 42-46 |
| Features, alpha, tau, models, optimiser, early stopping | unchanged | unchanged |
| Extra outputs | | Sev-NR, PDR-NR per fold; H1 statistics |

`--split image --graph transductive` reproduces the released protocol, so the
2 x 2 design (split x graph) separates the fellow-eye effect from the
transductive effect.

## Installation

Copy `src/patient_level.py`, `src/stats_h1.py` and `tests/test_patient_level.py`
into the `src/` and `tests/` folders of the plosene repository, then install
its requirements (`pip install -r requirements.txt`).

## Patient identifiers

* **Kaggle DR**: taken from the file name (`10_left.jpeg`, `10_right.jpeg` -> patient `10`).
* **APTOS 2019**: one image per patient (the file stem).
* **Messidor-2**: provide `--patient_map` with a CSV `image,patient`
  (the exam/patient pairing distributed with Messidor-2). The script refuses to
  guess.

## Usage (from the plosene root, with `src/` on `PYTHONPATH`)

```bash
export PYTHONPATH=src

# 1. extract and cache features once per dataset (slow step)
python src/patient_level.py extract --dataset kaggle_dr --out features/kaggle_dr.npz
python src/patient_level.py extract --dataset aptos2019 --out features/aptos2019.npz
python src/patient_level.py extract --dataset messidor2 --out features/messidor2.npz \
       --patient_map data/messidor2/patient_map.csv

# 2. patient-level, inductive, five seeds: baseline vs full TOPORET
for d in kaggle_dr messidor2 aptos2019; do
  python src/patient_level.py evaluate --features features/$d.npz \
         --split patient --graph inductive --seeds 42 43 44 45 46 \
         --out results/patient_inductive/$d
done

# 3. H1 statistics: paired t per seed, Nadeau-Bengio corrected t, Fisher, bootstrap, sign-flip
python src/stats_h1.py results/patient_inductive/*/folds.json --json results/h1_patient_inductive.json

# optional: the 2 x 2 design on one dataset
for s in image patient; do for g in transductive inductive; do
  python src/patient_level.py evaluate --features features/kaggle_dr.npz \
         --split $s --graph $g --seeds 42 --out results/${s}_${g}/kaggle_dr
done; done

# optional: full ablation (A-D) under the new protocol
python src/patient_level.py evaluate --features features/kaggle_dr.npz \
       --variants CNN_only CNN_TDA CNN_Graph CNN_TDA_Graph --seeds 42 43 44 45 46 \
       --out results/ablation_patient/kaggle_dr
```

## Checks

`pytest tests/test_patient_level.py` verifies, without image data, that
patient-level folds share no patient, that every image is tested once, that the
inductive edges only leave training images, that the graph rule is identical
to `similarity_graph.compound_similarity_matrix`, and that the statistics
reproduce the values of Chapter 4 (t_4 = 3.82 -> corrected t = 2.45; Fisher
p = 0.015; one-sided sign-flip p = 1/32).

## Memory and time

The graph is computed block-wise, so the 35,126 x 35,126 similarity matrix of
Kaggle DR is never stored. Each fold trains one MLP and one GraphSAGE model;
five seeds x five folds x two variants = 50 trainings per dataset.

## Reporting

Report the result whatever it is. If H1 holds at the patient level, the main
limitation of Chapter 4 is removed; if the gain shrinks, the size of the
fellow-eye / transductive effect is itself a finding (use the 2 x 2 design).
