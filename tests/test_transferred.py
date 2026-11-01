"""Unit tests for Eq. 9-12."""

from __future__ import annotations

import numpy as np

from ebba.energy import softmax
from ebba.transferred import (
    mean_difference_check,
    max_nte,
    nte,
    recover_labels,
    temperature_sensitivity,
)


def _toy(seed: int = 0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(12, 10))
    corrupt = logits[None] + rng.normal(scale=0.5, size=(5, 12, 10))
    return logits, corrupt


def test_nte_matches_literal_formula():
    logits, corrupt = _toy()
    s = softmax(logits)
    sc = softmax(corrupt.reshape(-1, 10)).reshape(corrupt.shape)
    literal = (np.exp(sc - s[None]) - 1.0).sum(axis=0)
    got = nte(logits, corrupt, temperature=1.0, cap=None)
    assert np.allclose(got, literal)


def test_nte_cap_is_applied_per_corruption():
    logits, corrupt = _toy()
    uncapped = nte(logits, corrupt, cap=None)
    capped = nte(logits, corrupt, cap=4.0)
    assert (capped <= 4.0 * corrupt.shape[0] + 1e-9).all()
    assert (capped <= uncapped + 1e-9).all()


def test_softmax_mean_difference_is_zero():
    logits, corrupt = _toy(1)
    out = mean_difference_check(logits, corrupt)
    assert out["max_abs_mean_difference"] < 1e-9
    assert out["softmax_sums"] < 1e-9


def test_recover_labels_is_argmax():
    scores = np.array([[0.0, 3.0, 1.0], [5.0, 0.0, 0.0]])
    assert recover_labels(scores).tolist() == [1, 0]
    assert max_nte(scores).tolist() == [3.0, 5.0]


def test_temperature_sensitivity_is_bounded():
    logits, corrupt = _toy(2)
    out = temperature_sensitivity(logits, corrupt)
    assert set(out) == {"0.5", "1.0", "2.0", "3.0"}
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_nte_dtype_and_shape():
    logits, corrupt = _toy(3)
    scores = nte(logits, corrupt)
    assert scores.shape == (12, 10)
    assert np.isfinite(scores).all()
