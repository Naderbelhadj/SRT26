"""Paper tables, figures and statistics from ``benchmark.py`` predictions.

Outputs (in ``<runs>/paper``):
  table_sota.tex        reproduced baselines vs the proposed model, same protocol
  table_ablation.tex    ablations of the proposed model
  table_external.tex    external validation (models trained on internal data only)
  table_pergrade.tex    per-grade precision / recall / F1 of the proposed model
  table_literature.tex  results *reported* in the literature (from a sourced CSV),
                        printed separately because the protocols differ
  results_macros.tex    \\newcommand macros with every number quoted in the text
  fig_*.pdf             QWK with CIs, confusion matrix, reliability diagram
  results.json          everything above in machine-readable form

Statistics: mean +/- SD over seed x fold runs; 95% CIs by patient-level bootstrap;
comparison with the reference model by the corrected repeated k-fold t-test
(Bouckaert & Frank 2004, Nadeau & Bengio 2003) with Holm correction; external sets:
exact McNemar test and paired patient bootstrap of the QWK difference, using the
ensemble of the cross-validated models; Sev-NR / PDR-NR with exact Clopper-Pearson
intervals; ECE (15 bins) and Brier score.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

GRADES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]


# ----------------------------------------------------------------------------- metrics
def ece(y, P, bins=15):
    conf, pred = P.max(1), P.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def clopper_pearson(k, n, a=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def all_metrics(y, P):
    pred = P.argmax(1)
    out = {"accuracy": accuracy_score(y, pred),
           "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
           "macro_f1": f1_score(y, pred, average="macro", labels=range(5), zero_division=0),
           "ece": ece(y, P), "brier": float(((P - np.eye(5)[y]) ** 2).sum(1).mean())}
    ref = (y >= 2).astype(int)
    out["auc_referable"] = roc_auc_score(ref, P[:, 2:].sum(1)) if 0 < ref.sum() < len(ref) else float("nan")
    for g, key in [(3, "sev"), (4, "pdr")]:
        m = y == g
        k, n = int((pred[m] <= 1).sum()), int(m.sum())
        out[f"{key}_nr"] = k / n if n else float("nan")
        out[f"{key}_nr_ci"] = clopper_pearson(k, n)
        out[f"{key}_n"] = n
    out["conf_0_4"] = int(((y == 0) & (pred == 4)).sum() + ((y == 4) & (pred == 0)).sum())
    return {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) and k not in ("sev_n", "pdr_n", "conf_0_4") else v)
            for k, v in out.items()}


def patient_bootstrap(y, P, patients, fn, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(patients, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        try:
            vals.append(fn(y[idx], P[idx]))
        except ValueError:
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def qwk_of(y, P):
    return cohen_kappa_score(y, P.argmax(1), weights="quadratic")


def corrected_repeated_t(d, ratio):
    n = len(d)
    var = np.var(d, ddof=1)
    if n < 2 or var == 0:
        return float("nan"), float("nan")
    t = d.mean() / np.sqrt((1 / n + ratio) * var)
    return float(t), float(2 * stats.t.sf(abs(t), n - 1))


def holm(pvals: dict) -> dict:
    items = sorted((p, k) for k, p in pvals.items() if not np.isnan(p))
    m, adj, run = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    for k, p in pvals.items():
        adj.setdefault(k, float("nan"))
    return adj


def mcnemar_exact(correct_a, correct_b):
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    return float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0


# ----------------------------------------------------------------------------- loading
def load(runs_dir):
    rows = list(csv.DictReader(open(Path(runs_dir) / "predictions.csv")))
    meta = json.load(open(Path(runs_dir) / "runs.json"))
    data = defaultdict(list)          # (method, split, dataset) -> rows
    for r in rows:
        data[(r["method"], r["split"], r["dataset"])].append(r)
    return data, meta


def arrays(rs):
    y = np.array([int(r["y"]) for r in rs])
    P = np.array([[float(r[f"p{k}"]) for k in range(5)] for r in rs])
    pat = np.array([r["patient"] for r in rs])
    img = np.array([r["image"] for r in rs])
    return y, P, pat, img


def fmt(v, d=3):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def fmt_p(p):
    return "--" if np.isnan(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def tex_escape(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


# ----------------------------------------------------------------------------- analysis
def internal_summary(data, method, ratio):
    rs = [r for (m, s, _), v in data.items() if m == method and s == "internal" for r in v]
    by_run = defaultdict(list)
    for r in rs:
        by_run[(r["seed"], r["fold"])].append(r)
    per_run = {k: all_metrics(*arrays(v)[:2]) for k, v in by_run.items()}
    keys = ["accuracy", "qwk", "macro_f1", "auc_referable", "ece", "brier"]
    summ = {k: (float(np.nanmean([m[k] for m in per_run.values()])),
                float(np.nanstd([m[k] for m in per_run.values()], ddof=1)) if len(per_run) > 1 else 0.0)
            for k in keys}
    # pooled out-of-fold predictions of the first seed: one prediction per internal image
    seeds = sorted({k[0] for k in by_run})
    pooled = [r for k, v in by_run.items() if k[0] == seeds[0] for r in v]
    y, P, pat, _ = arrays(pooled)
    pm = all_metrics(y, P)
    summ["qwk_pooled"] = pm["qwk"]                                   # point estimate matching the CI
    summ["qwk_ci"] = patient_bootstrap(y, P, pat, qwk_of)
    for key in ("sev_nr", "sev_nr_ci", "sev_n", "pdr_nr", "pdr_nr_ci", "pdr_n", "conf_0_4"):
        summ[key] = pm[key]
    summ["per_run_qwk"] = {f"{k[0]}-{k[1]}": m["qwk"] for k, m in per_run.items()}
    summ["pooled"] = (y, P)
    return summ


def external_summary(data, method, dataset):
    rs = [r for r in data.get((method, "external", dataset), [])]
    if not rs:
        return None
    by_img = defaultdict(list)
    for r in rs:
        by_img[r["image"]].append(r)
    imgs = sorted(by_img)
    y = np.array([int(by_img[i][0]["y"]) for i in imgs])
    pat = np.array([by_img[i][0]["patient"] for i in imgs])
    P = np.array([np.mean([[float(r[f"p{k}"]) for k in range(5)] for r in by_img[i]], 0) for i in imgs])
    runs = defaultdict(list)
    for r in rs:
        runs[(r["seed"], r["fold"])].append(r)
    q_runs = [qwk_of(*arrays(v)[:2]) for v in runs.values()]
    m = all_metrics(y, P)
    m.update({"qwk_ci": patient_bootstrap(y, P, pat, qwk_of), "qwk_runs_mean": float(np.mean(q_runs)),
              "qwk_runs_sd": float(np.std(q_runs, ddof=1)) if len(q_runs) > 1 else 0.0,
              "images": imgs, "y": y, "P": P, "patients": pat})
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--reference", default="TAG-MTL-DR (ours)")
    ap.add_argument("--literature", default=None, help="sourced CSV of results reported in the literature")
    ap.add_argument("--no_figures", action="store_true")
    a = ap.parse_args()

    from benchmark import ABLATIONS, BASELINES
    data, meta = load(a.runs)
    methods = list(meta["methods"])
    if a.reference not in methods:
        raise SystemExit(f"reference '{a.reference}' not in the runs: {methods}")
    out = Path(a.runs) / "paper"
    out.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(Path(a.runs) / "predictions.csv")))
    n_train = np.mean([float(r["n_train"]) for r in rows])
    n_test = np.mean([float(r["n_test"]) for r in rows])
    ratio = n_test / n_train
    params = defaultdict(list)
    for r in meta["runs"]:
        params[r["method"]].append(r["params"])

    S = {m: internal_summary(data, m, ratio) for m in methods}
    ref = S[a.reference]
    pvals = {}
    for m in methods:
        if m == a.reference:
            continue
        common = sorted(set(ref["per_run_qwk"]) & set(S[m]["per_run_qwk"]))
        d = np.array([ref["per_run_qwk"][k] - S[m]["per_run_qwk"][k] for k in common])
        S[m]["delta_qwk"] = float(d.mean()) if len(d) else float("nan")
        S[m]["t"], pvals[m] = corrected_repeated_t(d, ratio)
    padj = holm(pvals)

    def row(m, ours=False):
        s = S[m]
        name = tex_escape(m)
        name = r"\textbf{" + name + "}" if ours else name
        sev = f"{100 * s['sev_nr']:.1f} [{100 * s['sev_nr_ci'][0]:.1f}, {100 * s['sev_nr_ci'][1]:.1f}]"
        return (f"{name} & {np.mean(params[m]) / 1e6:.1f} & {fmt(s['accuracy'][0])}$\\pm${fmt(s['accuracy'][1])} & "
                f"{fmt(s['qwk'][0])}$\\pm${fmt(s['qwk'][1])} & {fmt(s['qwk_pooled'])} [{fmt(s['qwk_ci'][0])}, {fmt(s['qwk_ci'][1])}] & "
                f"{fmt(s['macro_f1'][0])} & {fmt(s['auc_referable'][0])} & {sev} & {fmt(s['ece'][0])} & "
                f"{'--' if ours else fmt_p(padj.get(m, float('nan')))} \\\\")

    head = (r"\begin{table*}[t]\centering\small" "\n"
            r"\caption{CAPTION}\label{LABEL}" "\n"
            r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}lrcccccccc@{}}\toprule" "\n"
            r"Method & Params (M) & Accuracy & QWK & Pooled QWK [95\% CI] & Macro-F1 & AUC (ref.) & Sev-NR \% [95\% CI] & ECE & $p_{\mathrm{Holm}}$ \\ \midrule")
    foot = r"\bottomrule\end{tabular}}" "\n" r"\par\smallskip\footnotesize NOTE" "\n" r"\end{table*}"
    proto = (f"Patient-grouped {meta['args']['folds']}-fold cross-validation, seeds "
             f"{', '.join(map(str, meta['args']['seeds']))}; mean $\\pm$ SD over runs; pooled QWK: out-of-fold predictions of "
             f"the first seed, 95\\% CI by patient-level bootstrap; $p$: corrected repeated $k$-fold $t$-test "
             f"against {tex_escape(a.reference)}, Holm-adjusted. All methods trained with the same protocol.")
    sota = [m for m in BASELINES if m in S]
    with open(out / "table_sota.tex", "w") as f:
        f.write(head.replace("CAPTION", "Comparison with reproduced state-of-the-art baselines (internal data, same protocol).").replace("LABEL", "tab:sota") + "\n")
        for m in sota:
            f.write(row(m) + "\n")
        f.write(r"\midrule" "\n" + row(a.reference, ours=True) + "\n")
        f.write(foot.replace("NOTE", proto) + "\n")
    abl = [m for m in ABLATIONS if m in S]
    if abl:
        with open(out / "table_ablation.tex", "w") as f:
            f.write(head.replace("CAPTION", "Ablation of the proposed model (internal data).").replace("LABEL", "tab:ablation") + "\n")
            f.write(row(a.reference, ours=True) + "\n" + r"\midrule" "\n")
            for m in abl:
                f.write(row(m) + "\n")
            f.write(foot.replace("NOTE", proto) + "\n")

    # per-grade table of the reference model
    y, P = ref["pooled"]
    pr, rc, f1, sup = precision_recall_fscore_support(y, P.argmax(1), labels=range(5), zero_division=0)
    with open(out / "table_pergrade.tex", "w") as f:
        f.write(r"\begin{table}[t]\centering\small\caption{Per-grade results of " + tex_escape(a.reference)
                + r" (pooled out-of-fold predictions).}\label{tab:pergrade}" "\n"
                r"\begin{tabular}{@{}lcccr@{}}\toprule Grade & Precision & Recall & F1 & $N$ \\ \midrule" "\n")
        for g in range(5):
            f.write(f"{g} -- {GRADES[g]} & {pr[g]:.3f} & {rc[g]:.3f} & {f1[g]:.3f} & {int(sup[g])} \\\\\n")
        f.write(r"\bottomrule\end{tabular}\end{table}" "\n")

    # computational cost (all methods, same hardware)
    cost = defaultdict(lambda: defaultdict(list))
    for r in meta["runs"]:
        for k in ("params", "train_seconds", "infer_ms_per_image", "temperature"):
            if k in r:
                cost[r["method"]][k].append(r[k])
    with open(out / "table_cost.tex", "w") as f:
        f.write(r"\begin{table}[t]\centering\small\caption{Computational cost on "
                + tex_escape(meta.get("environment", {}).get("gpu", "the same hardware"))
                + r" (mean over runs).}\label{tab:cost}" "\n"
                r"\begin{tabular}{@{}lcccc@{}}\toprule Method & Params (M) & Training (min/run) & Inference (ms/image) & Temperature \\ \midrule" "\n")
        for m in sota + [a.reference] + abl:
            c = cost[m]
            f.write(f"{tex_escape(m)} & {np.mean(c['params']) / 1e6:.1f} & "
                    f"{(np.mean(c['train_seconds']) / 60 if c['train_seconds'] else float('nan')):.1f} & "
                    f"{(np.mean(c['infer_ms_per_image']) if c['infer_ms_per_image'] else float('nan')):.1f} & "
                    f"{(np.mean(c['temperature']) if c['temperature'] else float('nan')):.2f} \\\\\n")
        f.write(r"\bottomrule\end{tabular}\end{table}" "\n")

    # external validation
    ext = {}
    lines = []
    for ds in meta["externals"]:
        R = external_summary(data, a.reference, ds)
        if R is None:
            continue
        ext[ds] = {}
        for m in methods:
            E = external_summary(data, m, ds)
            if E is None:
                continue
            res = {k: E[k] for k in ("qwk", "qwk_ci", "qwk_runs_mean", "qwk_runs_sd", "accuracy",
                                     "auc_referable", "sev_nr", "sev_nr_ci", "sev_n", "pdr_nr", "pdr_nr_ci",
                                     "ece", "conf_0_4")}
            if m != a.reference and np.array_equal(E["images"], R["images"]):
                ca, cb = R["P"].argmax(1) == R["y"], E["P"].argmax(1) == E["y"]
                res["mcnemar_p"] = mcnemar_exact(ca, cb)
                Pj = np.concatenate([R["P"], E["P"]], 1)
                res["delta_qwk_ci"] = patient_bootstrap(R["y"], Pj, R["patients"],
                                                        lambda yy, PP: qwk_of(yy, PP[:, :5]) - qwk_of(yy, PP[:, 5:]))
            ext[ds][m] = res
            sev = f"{100 * res['sev_nr']:.1f} [{100 * res['sev_nr_ci'][0]:.1f}, {100 * res['sev_nr_ci'][1]:.1f}]"
            dci = res.get("delta_qwk_ci")
            name = tex_escape(m) if m != a.reference else r"\textbf{" + tex_escape(m) + "}"
            lines.append(f"{tex_escape(ds)} & {name} & {fmt(res['qwk'])} [{fmt(res['qwk_ci'][0])}, {fmt(res['qwk_ci'][1])}] & "
                         f"{fmt(res['auc_referable'])} & {sev} & {res['conf_0_4']} & "
                         f"{'--' if dci is None else f'[{dci[0]:+.3f}, {dci[1]:+.3f}]'} & "
                         f"{fmt_p(res.get('mcnemar_p', float('nan')))} \\\\")
    if lines:
        with open(out / "table_external.tex", "w") as f:
            f.write(r"\begin{table*}[t]\centering\small\caption{External validation: models trained on the internal "
                    r"data only, ensemble of the cross-validated models.}\label{tab:external}" "\n"
                    r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}llcccccc@{}}\toprule Dataset & Method & QWK [95\% CI] & AUC (ref.) & "
                    r"Sev-NR \% [95\% CI] & 0$\leftrightarrow$4 & $\Delta$QWK (ours $-$ method) 95\% CI & McNemar $p$ \\ \midrule" "\n")
            f.write("\n".join(lines) + "\n" + r"\bottomrule\end{tabular}}\end{table*}" "\n")

    # literature (reported, different protocols)
    if a.literature:
        lit = list(csv.DictReader(open(a.literature)))
        need = ["method", "year", "citation_key", "dataset", "protocol", "metric", "value"]
        bad = [r for r in lit if any(not (r.get(k) or "").strip() for k in need)]
        if bad:
            raise SystemExit(f"literature rows with missing fields (every value needs a source and a protocol): {bad[:3]}")
        with open(out / "table_literature.tex", "w") as f:
            f.write(r"\begin{table*}[t]\centering\small\caption{Results reported in the literature. Values are copied "
                    r"from the cited papers; datasets, splits and metrics differ, so they are not directly comparable "
                    r"with Table~\ref{tab:sota}.}\label{tab:literature}" "\n"
                    r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}llllll@{}}\toprule Method & Year & Dataset & Protocol & Metric & Value \\ \midrule" "\n")
            for r in lit:
                f.write(f"{tex_escape(r['method'])}~\\cite{{{r['citation_key']}}} & {r['year']} & {tex_escape(r['dataset'])} & "
                        f"{tex_escape(r['protocol'])} & {tex_escape(r['metric'])} & {tex_escape(r['value'])} \\\\\n")
            f.write(r"\bottomrule\end{tabular}}\end{table*}" "\n")

    # macros quoted in the text
    best_base = max(sota, key=lambda m: S[m]["qwk"][0]) if sota else None
    mac = {"OursQWK": fmt(ref["qwk"][0]), "OursQWKsd": fmt(ref["qwk"][1]), "OursQWKpooled": fmt(ref["qwk_pooled"]),
           "OursQWKlo": fmt(ref["qwk_ci"][0]), "OursQWKhi": fmt(ref["qwk_ci"][1]),
           "OursSevNR": f"{100 * ref['sev_nr']:.1f}", "OursSevNRhi": f"{100 * ref['sev_nr_ci'][1]:.1f}",
           "OursECE": fmt(ref["ece"][0]), "NSeeds": str(len(meta["args"]["seeds"])), "NFolds": str(meta["args"]["folds"])}
    if best_base:
        mac.update({"BestBaseline": tex_escape(best_base), "BestBaselineQWK": fmt(S[best_base]["qwk"][0]),
                    "DeltaBest": f"{S[best_base]['delta_qwk']:+.3f}", "PBest": fmt_p(padj[best_base])})
    for i, ds in enumerate(ext):
        if a.reference in ext[ds]:
            tag = "Ext" + "ABCDEFGH"[i]
            mac[tag + "Name"] = tex_escape(ds)
            mac[tag + "QWK"] = fmt(ext[ds][a.reference]["qwk"])
            mac[tag + "SevNR"] = f"{100 * ext[ds][a.reference]['sev_nr']:.1f}"
    with open(out / "results_macros.tex", "w") as f:
        f.write("% generated by analysis.py -- do not edit by hand\n")
        for k, v in mac.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # figures
    if not a.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = sota + [a.reference] + abl
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(order) + 1))
        for i, m in enumerate(order):
            lo, hi = S[m]["qwk_ci"]
            c = "#1F4E79" if m == a.reference else ("#888888" if m in abl else "#A7B4C2")
            q = S[m]["qwk_pooled"]
        ax.errorbar(q, i, xerr=[[max(q - lo, 0)], [max(hi - q, 0)]], fmt="o", color=c, capsize=3)
        ax.set_yticks(range(len(order)), order)
        ax.invert_yaxis()
        ax.set_xlabel("Pooled out-of-fold QWK (95% patient-bootstrap CI)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "fig_qwk.pdf")
        plt.close(fig)
        cm = confusion_matrix(y, P.argmax(1), labels=range(5), normalize="true")
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(5), GRADES, rotation=30)
        ax.set_yticks(range(5), GRADES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Reference")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(out / "fig_confusion.pdf")
        plt.close(fig)
        conf, corr = P.max(1), P.argmax(1) == y
        edges = np.linspace(0, 1, 11)
        xs, ys_ = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mk = (conf > lo) & (conf <= hi)
            if mk.any():
                xs.append(conf[mk].mean())
                ys_.append(corr[mk].mean())
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot(xs, ys_, "o-", color="#1F4E79")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"ECE = {ref['ece'][0]:.3f}")
        fig.tight_layout()
        fig.savefig(out / "fig_reliability.pdf")
        plt.close(fig)

    clean = {m: {k: v for k, v in s.items() if k != "pooled"} for m, s in S.items()}
    for m in clean:
        clean[m]["p_holm"] = padj.get(m)
    ext_clean = {ds: {m: {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for m, r in d.items()}
                 for ds, d in ext.items()}
    json.dump({"internal": clean, "external": ext_clean, "reference": a.reference, "test_train_ratio": ratio},
              open(out / "results.json", "w"), indent=1, default=float)
    print(f"tables, macros and figures written to {out}")
    for m in sota + [a.reference] + abl:
        s = S[m]
        print(f"  {m:<36} QWK {s['qwk'][0]:.3f}±{s['qwk'][1]:.3f}  Sev-NR {100 * s['sev_nr']:.1f}%  "
              f"p_Holm={'ref' if m == a.reference else fmt_p(padj.get(m, float('nan')))}")


if __name__ == "__main__":
    main()
