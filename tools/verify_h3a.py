"""Recompute the H3a quantities that the manuscript cannot document without the raw data.

Input: one CSV of DOTS predictions (validation AND test rows), with columns
    image_id, dataset, split, true_grade, p0, p1, p2, p3, p4
where split is "val" or "test" and p0..p4 are the five grade probabilities.

Outputs, for the thesis:
  1. the referral threshold t (read from your config, or chosen here on validation),
  2. coverage and Sev-NR on VALIDATION at tau_U = 0.73 (to report next to the test values),
  3. the dataset x grade composition of the 10,000-image test set,
  4. Sev-NR and coverage 95% CIs by grade-stratified bootstrap (B = 10,000).

Usage: python verify_h3a.py predictions.csv [--t 0.5]
"""
import argparse

import numpy as np
import pandas as pd

TAU_U = 0.73  # nats, abstention threshold used in the thesis


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(axis=1)


def metrics(df, t):
    p = df[["p0", "p1", "p2", "p3", "p4"]].to_numpy()
    deferred = entropy(p) > TAU_U
    p_ge3 = p[:, 3] + p[:, 4]
    referred = deferred | (p_ge3 > t)
    severe = df["true_grade"].to_numpy() >= 3
    pdr = df["true_grade"].to_numpy() == 4
    return {
        "coverage": 1 - deferred.mean(),
        "sev_nr": (~referred[severe]).mean(),
        "pdr_nr": (~referred[pdr]).mean(),
        "n_severe": int(severe.sum()),
    }


def choose_t_on_validation(val):
    # Largest t keeping validation Sev-NR < 2% (fewest referrals); adapt if your code differs.
    best = 0.0
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        if metrics(val, t)["sev_nr"] < 0.02:
            best = t
    return best


def stratified_bootstrap(test, t, b=10_000, seed=42):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in test.groupby("true_grade")]
    sev, cov = [], []
    for _ in range(b):
        sample = pd.concat([g.sample(len(g), replace=True, random_state=rng.integers(1 << 31))
                            for g in groups])
        m = metrics(sample, t)
        sev.append(m["sev_nr"])
        cov.append(m["coverage"])
    q = lambda x: np.percentile(x, [2.5, 97.5])
    return q(sev), q(cov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--t", type=float, default=None, help="referral threshold from your config")
    ap.add_argument("--b", type=int, default=10_000)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    val, test = df[df.split == "val"], df[df.split == "test"]

    t = args.t if args.t is not None else choose_t_on_validation(val)
    print(f"1. Referral threshold t = {t}  ({'from config' if args.t is not None else 'chosen on validation'})")

    mv, mt = metrics(val, t), metrics(test, t)
    print(f"2. tau_U=0.73  VALIDATION: coverage={mv['coverage']:.1%}  Sev-NR={mv['sev_nr']:.1%}")
    print(f"               TEST      : coverage={mt['coverage']:.1%}  Sev-NR={mt['sev_nr']:.1%}  "
          f"PDR-NR={mt['pdr_nr']:.1%}  (n_severe={mt['n_severe']})")

    print("3. Test composition (dataset x grade):")
    print(pd.crosstab(test["dataset"], test["true_grade"], margins=True))

    (s_lo, s_hi), (c_lo, c_hi) = stratified_bootstrap(test, t, b=args.b)
    print(f"4. Bootstrap 95% CI: Sev-NR [{s_lo:.1%}, {s_hi:.1%}]  coverage [{c_lo:.1%}, {c_hi:.1%}]")


if __name__ == "__main__":
    main()
