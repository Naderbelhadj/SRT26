"""Checks of the statistics used in the paper tables (no image data needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import clopper_pearson, corrected_repeated_t, holm, mcnemar_exact  # noqa: E402
from benchmark import class_probs  # noqa: E402
import torch  # noqa: E402


def test_clopper_pearson_matches_thesis_interval():
    lo, hi = clopper_pearson(3, 309)               # DOTS internal Sev-NR in the thesis: [0.2%, 2.8%]
    assert abs(lo - 0.002) < 0.001 and abs(hi - 0.028) < 0.001


def test_corrected_t_is_smaller_than_naive():
    d = np.array([0.03, 0.05, 0.04, 0.035, 0.045])
    t_naive = d.mean() / (d.std(ddof=1) / np.sqrt(5))
    t, p = corrected_repeated_t(d, 0.2 / 0.7)
    assert 0 < t < t_naive and 0 < p < 1


def test_holm_is_monotone_and_bounded():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03 and adj["c"] == 0.06 and adj["b"] == 0.06


def test_mcnemar_symmetry():
    a = np.array([1, 1, 0, 0, 1], bool)
    b = np.array([1, 0, 1, 0, 1], bool)
    assert mcnemar_exact(a, b) == 1.0


def test_coral_class_probabilities_sum_to_one():
    P = class_probs(torch.randn(8, 4), "coral")
    assert torch.allclose(P.sum(1), torch.ones(8), atol=1e-6) and (P >= 0).all()
