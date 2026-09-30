"""Tests of the patient-level protocol and of the H1 statistics (no image data needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from patient_level import (check_disjoint_patients, graph_edges, make_folds,  # noqa: E402
                           max_tda_distance, patient_id)
from stats_h1 import corrected_t, fisher, paired_t, sign_flip_p  # noqa: E402


def synthetic(n_patients=120, seed=0):
    rng = np.random.default_rng(seed)
    patients = np.repeat([f"p{i}" for i in range(n_patients)], 2)          # two eyes each
    grade = rng.integers(0, 5, n_patients)
    labels = np.repeat(grade, 2)
    cnn = rng.normal(size=(len(labels), 32)).astype(np.float32) + labels[:, None]
    tda = rng.random((len(labels), 6)).astype(np.float32)
    return cnn, tda, labels, patients


def test_patient_ids():
    assert patient_id("kaggle_dr", Path("10_left.jpeg"), None) == "10"
    assert patient_id("kaggle_dr", Path("10_right.jpeg"), None) == "10"
    assert patient_id("aptos2019", Path("000c1434d8d7.png"), None) == "000c1434d8d7"
    with pytest.raises(ValueError):
        patient_id("messidor2", Path("x.png"), None)
    assert patient_id("messidor2", Path("x.png"), {"x": "P7"}) == "P7"


def test_patient_split_is_disjoint_and_complete():
    _, _, labels, patients = synthetic()
    folds = make_folds(labels, patients, "patient", 5, 42)
    tests = np.concatenate([f[2] for f in folds])
    assert sorted(tests.tolist()) == list(range(len(labels)))      # every image tested once
    for train, val, test in folds:
        assert check_disjoint_patients(patients, train, val, test) == 0
        assert len(set(train) | set(val) | set(test)) == len(labels)


def test_image_split_can_share_patients():
    _, _, labels, patients = synthetic()
    folds = make_folds(labels, patients, "image", 5, 42)
    assert sum(check_disjoint_patients(patients, *f) for f in folds) > 0


def test_inductive_edges_only_leave_training_nodes():
    cnn, tda, labels, patients = synthetic()
    train, val, test = make_folds(labels, patients, "patient", 5, 42)[0]
    d_max = max_tda_distance(tda, train)
    new = np.concatenate([val, test])
    E = graph_edges(cnn, tda, train, new, 0.6, 0.5, d_max)
    assert E.shape[0] == 2 and E.shape[1] > 0
    assert np.isin(E[0], train).all()        # sources are training images
    assert np.isin(E[1], new).all()          # targets are new images
    E_tt = graph_edges(cnn, tda, train, train, 0.6, 0.5, d_max)
    assert np.isin(E_tt, train).all()
    assert (E_tt[0] != E_tt[1]).all()        # no self-loops


def test_graph_matches_released_similarity():
    """Same rule as similarity_graph.compound_similarity_matrix on a full graph."""
    cnn, tda, *_ = synthetic(20)
    idx = np.arange(len(cnn))
    E = graph_edges(cnn, tda, idx, idx, 0.6, 0.65, max_tda_distance(tda, idx))
    v = cnn / np.linalg.norm(cnn, axis=1, keepdims=True)
    t = tda / np.linalg.norm(tda, axis=1, keepdims=True)
    S_vis = np.clip(v @ v.T, 0, 1)
    W2 = np.sqrt(np.maximum(2 - 2 * np.clip(t @ t.T, -1, 1), 0))
    S = 0.6 * S_vis + 0.4 * np.exp(-W2 / W2.max())
    np.fill_diagonal(S, 0)
    ref = set(zip(*np.nonzero(S > 0.65)))
    got = set(zip(E[1].tolist(), E[0].tolist()))
    assert got == ref


def test_statistics_reproduce_thesis_values():
    """Chapter 4: t_4 = 3.82 -> corrected t = 2.45 with a 70/10/20 split; Fisher p = 0.015."""
    ratio = 0.20 / 0.70
    factor = np.sqrt((1 / 5) / (1 / 5 + ratio))
    for t4, tc in [(3.82, 2.45), (4.1, 2.63), (3.5, 2.25)]:
        assert abs(t4 * factor - tc) < 0.01
    assert abs(fisher([0.070, 0.058, 0.088]) - 0.015) < 0.002
    d = np.array([0.03, 0.05, 0.04, 0.035, 0.045])
    t, p = paired_t(d)
    tc, pc = corrected_t(d, ratio)
    assert abs(tc - t * factor) < 1e-9 and pc > p
    assert abs(sign_flip_p(d) - 1 / 32) < 1e-12                  # one-sided, as in the thesis
    assert abs(sign_flip_p(d, "two-sided") - 2 / 32) < 1e-12
