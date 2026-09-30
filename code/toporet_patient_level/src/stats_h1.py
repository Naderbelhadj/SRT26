"""H1 statistics on the fold-level outputs of ``patient_level.py``.

For each dataset, the paired difference TOPORET - baseline is taken fold by fold
(and seed by seed).  Reported, as in Chapter 4 of the thesis:

* paired two-tailed t-test over the folds (pre-specified test; |t_4| > 2.78 for
  five folds), per seed;
* Nadeau-Bengio corrected resampled t-test, which accounts for the overlap of
  the training sets; over r seeds x k folds it becomes the corrected repeated
  k-fold test (variance factor 1/(r k) + n_test/n_train, df = r k - 1);
* Fisher's combination of the corrected p-values over independent datasets;
* bootstrap 95% interval of the mean difference and an exact sign-flip test.

Usage
-----
    python src/stats_h1.py results/patient_level/*/folds.json \
        --baseline CNN_only --model CNN_TDA_Graph --metric qwk
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict

import numpy as np
from scipy import stats


def paired_t(d: np.ndarray) -> tuple[float, float]:
    k = len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return float("inf") * np.sign(d.mean()), 0.0
    t = d.mean() / (sd / np.sqrt(k))
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


def corrected_t(d: np.ndarray, ratio: float) -> tuple[float, float]:
    """Nadeau-Bengio (2003); with r x k differences, Bouckaert-Frank (2004)."""
    n = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return float("inf") * np.sign(d.mean()), 0.0
    t = d.mean() / np.sqrt((1.0 / n + ratio) * var)
    return float(t), float(2 * stats.t.sf(abs(t), n - 1))


def sign_flip_p(d: np.ndarray, alternative: str = "greater", max_exact: int = 16,
                n_mc: int = 100_000, seed: int = 0) -> float:
    """Sign-flip permutation test of mean(d) = 0.

    ``alternative='greater'`` is the one-sided test used in the thesis (with five
    positive folds its minimum is 1/32); ``'two-sided'`` doubles it.
    """
    if len(d) <= max_exact:
        signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
    else:
        signs = np.random.default_rng(seed).choice([1, -1], size=(n_mc, len(d)))
    means = (signs * d).mean(1)
    if alternative == "two-sided":
        return float(np.mean(np.abs(means) >= abs(d.mean()) - 1e-12))
    return float(np.mean(means >= d.mean() - 1e-12))


def bootstrap_ci(d: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fisher(pvals: list[float]) -> float:
    x = -2.0 * np.sum(np.log(np.maximum(pvals, 1e-300)))
    return float(stats.chi2.sf(x, 2 * len(pvals)))


def analyse(files: list[str], baseline: str, model: str, metric: str) -> dict:
    report = {"baseline": baseline, "model": model, "metric": metric, "datasets": {}, "fisher": {}}
    corrected_p = defaultdict(dict)          # protocol -> dataset -> corrected p
    for fn in files:
        with open(fn) as f:
            res = json.load(f)
        by = defaultdict(dict)
        sizes = []
        for r in res["records"]:
            by[(r["seed"], r["fold"])][r["variant"]] = r[metric]
            sizes.append((r["n_train"], r["n_test"]))
        keys = sorted(k for k in by if baseline in by[k] and model in by[k])
        if not keys:
            raise ValueError(f"{fn}: no fold has both {baseline} and {model}")
        d = np.array([by[k][model] - by[k][baseline] for k in keys])
        n_train = np.mean([s[0] for s in sizes])
        n_test = np.mean([s[1] for s in sizes])
        ratio = n_test / n_train
        per_seed = {}
        for seed in sorted({k[0] for k in keys}):
            ds = np.array([by[k][model] - by[k][baseline] for k in keys if k[0] == seed])
            t, p = paired_t(ds)
            tc, pc = corrected_t(ds, ratio)
            per_seed[seed] = {"n_folds": len(ds), "mean_diff": float(ds.mean()), "t": t, "p": p,
                              "t_corrected": tc, "p_corrected": pc,
                              "positive_folds": int((ds > 0).sum())}
        tc_all, pc_all = corrected_t(d, ratio)
        lo, hi = bootstrap_ci(d)
        protocol = f"{res['split']}/{res['graph']}"
        key = f"{res['dataset']} [{protocol}]"
        if key in report["datasets"]:
            raise ValueError(f"{fn}: {key} given twice")
        corrected_p[protocol][res["dataset"]] = pc_all
        report["datasets"][key] = {"dataset": res["dataset"],
            "split": res["split"], "graph": res["graph"], "n_differences": len(d),
            "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "bootstrap_95ci": [lo, hi], "sign_flip_p_one_sided": sign_flip_p(d),
            "sign_flip_p_two_sided": sign_flip_p(d, "two-sided"),
            "t_corrected_all": tc_all, "p_corrected_all": pc_all,
            "test_train_ratio": ratio, "per_seed": per_seed,
            "baseline_mean": float(np.mean([by[k][baseline] for k in keys])),
            "model_mean": float(np.mean([by[k][model] for k in keys])),
        }
    for protocol, ps in corrected_p.items():      # independent datasets, same protocol
        if len(ps) > 1:
            report["fisher"][protocol] = {"datasets": sorted(ps), "p": fisher(list(ps.values()))}
    return report


def print_report(rep: dict) -> None:
    m = rep["metric"]
    print(f"\nH1: {rep['model']} vs {rep['baseline']}  (metric: {m})")
    print("-" * 96)
    print(f"{'dataset':<14}{'protocol':<22}{'baseline':>9}{'model':>9}{'diff':>9}"
          f"{'95% CI':>20}{'t_corr':>8}{'p_corr':>8}")
    for name, r in rep["datasets"].items():
        proto = f"{r['split']}/{r['graph']}"
        ci = f"[{r['bootstrap_95ci'][0]:+.3f},{r['bootstrap_95ci'][1]:+.3f}]"
        print(f"{r['dataset']:<14}{proto:<22}{r['baseline_mean']:>9.3f}{r['model_mean']:>9.3f}"
              f"{r['mean_diff']:>+9.3f}{ci:>20}{r['t_corrected_all']:>8.2f}{r['p_corrected_all']:>8.3f}")
        for seed, s in r["per_seed"].items():
            print(f"    seed {seed}: t{s['n_folds'] - 1}={s['t']:.2f} p={s['p']:.3f}  "
                  f"corrected t={s['t_corrected']:.2f} p={s['p_corrected']:.3f}  "
                  f"gain>0 in {s['positive_folds']}/{s['n_folds']} folds")
    for protocol, f in rep["fisher"].items():
        print(f"Fisher combination of the corrected p-values ({protocol}; "
              f"{', '.join(f['datasets'])}): p = {f['p']:.4f}")
    print("-" * 96)
    print("Pre-specified decision (thesis): |t_4| > 2.78 on the primary benchmark, two-tailed.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="folds.json files, one per dataset")
    p.add_argument("--baseline", default="CNN_only")
    p.add_argument("--model", default="CNN_TDA_Graph")
    p.add_argument("--metric", default="qwk", choices=["qwk", "accuracy", "macro_f1"])
    p.add_argument("--json", default=None, help="write the report to this file")
    a = p.parse_args()
    rep = analyse(a.files, a.baseline, a.model, a.metric)
    print_report(rep)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
