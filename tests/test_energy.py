"""Unit tests for Eq. 3-8."""

from __future__ import annotations

import numpy as np
import pytest

from ebba.energy import (
    clamp_max_logits,
    detect,
    energy_stats,
    label_energy,
    per_sample_normalized_energy,
    softmax,
    sweep_lambda,
)


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(17, 10))
    s = softmax(logits)
    assert np.allclose(s.sum(axis=1), 1.0)
    assert (s >= 0).all()


def test_clamp_sets_max_to_zero_not_subtract():
    z = np.array([[3.0, 8.0, -2.0]])
    assert np.array_equal(clamp_max_logits(z)[0], np.array([3.0, 0.0, -2.0]))
    # The input must not be modified when inplace=False.
    assert np.array_equal(z[0], np.array([3.0, 8.0, -2.0]))


def test_clamp_does_not_preserve_softmax():
    z = np.array([[3.0, 8.0, -2.0]])
    assert not np.allclose(softmax(clamp_max_logits(z)), softmax(z))


def test_appendix_b_energies():
    """Exact recomputation from the logits printed in Appendix B."""
    Z = np.array([[9.0, 1.0, -1.0], [3.0, 8.0, -2.0], [-1.0, 3.0, 7.0]])
    energies = np.exp(softmax(Z)).sum(axis=0)
    assert np.allclose(energies, [4.724291496758145, 4.71850687291136, 4.669054507269333])
    # The paper's printed order E0 > E2 > E1 does not hold.
    assert list(np.argsort(-energies)) == [0, 1, 2]


def test_clamped_appendix_b_energies():
    Z = np.array([[9.0, 1.0, -1.0], [3.0, 8.0, -2.0], [-1.0, 3.0, 7.0]])
    energies = np.exp(softmax(clamp_max_logits(Z))).sum(axis=0)
    assert np.allclose(energies, [4.8712, 5.5436, 3.1483], atol=5e-4)
    # After clamping, the target (label 1) is the highest, which is the
    # direction Eq. 8 uses to flag a backdoor.
    assert int(np.argmax(energies)) == 1
    assert list(np.argsort(-energies)) == [1, 0, 2]


def test_normalization_variants_differ():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(50, 10))
    paper = per_sample_normalized_energy(logits, 2.0, "paper")
    readme = per_sample_normalized_energy(logits, 2.0, "readme")
    official = per_sample_normalized_energy(logits, 2.0, "official")
    assert paper.shape == readme.shape == official.shape == (50, 10)
    assert not np.allclose(paper, readme)
    assert not np.allclose(paper, official)


def test_energy_stats_uses_population_std():
    e = np.array([10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    st = energy_stats(e)
    assert st.mean == pytest.approx(5.5)
    assert st.std == pytest.approx(1.5)
    assert st.deviation.max() == pytest.approx(4.5)


def test_detect_one_and_two_sided():
    e = np.array([10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    one = detect(e, lam=1.0)
    assert one["is_backdoored"] and one["target"] == 0
    assert one["margin"] == pytest.approx(3.0)

    shifted = e.copy()
    shifted[0] = 1.0                      # target pushed 3 sigma below the mean
    assert not detect(shifted, lam=1.0)["is_backdoored"]
    two = detect(shifted, lam=1.0, two_sided=True)
    assert two["is_backdoored"] and two["target"] == 0
    assert two["direction"] == "low"


def test_sweep_lambda_shapes():
    rng = np.random.default_rng(3)
    bd = rng.normal(size=10)
    bd[0] += 8.0
    clean = rng.normal(size=10)
    out = sweep_lambda(bd, clean)
    assert out["lambdas"].shape == (81,)
    assert out["backdoored_curve"].shape == (81,)
    assert out["chosen_lambda"] == 3.0
