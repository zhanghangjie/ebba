"""Unit tests for the P0 diagnostics in ``scripts/full_eval.py``.

The evaluator is a script, but these helpers are pure functions and are worth
pinning: a silent change to the refinement gate or the NTE sweep would alter
the report without any test failing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "full_eval_under_test", ROOT / "scripts" / "full_eval.py"
)
assert _SPEC is not None and _SPEC.loader is not None
full_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(full_eval)


def test_refine_cluster_drops_the_low_peak_component():
    rng = np.random.default_rng(0)
    clean_like = rng.normal(0.5, 0.02, 400)
    poison_like = rng.normal(3.0, 0.20, 200)
    peak = np.concatenate([clean_like, poison_like])
    base = np.ones(len(peak), dtype=bool)

    pred, stats = full_eval._refine_cluster(peak, base)

    assert stats["applied"] is True
    assert pred.dtype == base.dtype
    assert pred[400:].mean() > 0.95          # poisoned peak kept
    assert pred[:400].mean() < 0.05          # clean-like peak dropped
    assert stats["kept_fraction"] < 0.45


def test_refine_cluster_leaves_a_homogeneous_cluster_alone():
    rng = np.random.default_rng(1)
    peak = rng.normal(2.0, 0.001, 300)
    base = np.ones(len(peak), dtype=bool)

    pred, stats = full_eval._refine_cluster(peak, base)

    assert stats["applied"] is False
    assert np.array_equal(pred, base)


def test_refine_cluster_skips_tiny_clusters():
    base = np.zeros(100, dtype=bool)
    base[:5] = True
    pred, stats = full_eval._refine_cluster(np.arange(100, dtype=float), base)
    assert stats["applied"] is False
    assert np.array_equal(pred, base)


def test_nte_cap_is_inert_at_t1_and_binds_at_t2():
    """exp(s' - s) - 1 <= e - 1 at T=1, so cap=5 cannot bind there."""
    from ebba.transferred import nte

    rng = np.random.default_rng(2)
    source = rng.normal(size=(6, 10)).astype(np.float32)
    corrupt = rng.normal(size=(4, 6, 10)).astype(np.float32) * 4

    cap5_t1 = nte(source, corrupt, temperature=1.0, cap=5.0)
    unc_t1 = nte(source, corrupt, temperature=1.0, cap=None)
    cap5_t2 = nte(source, corrupt, temperature=2.0, cap=5.0)
    unc_t2 = nte(source, corrupt, temperature=2.0, cap=None)

    assert cap5_t1.shape == unc_t1.shape == cap5_t2.shape == unc_t2.shape == (6, 10)
    assert np.allclose(cap5_t1, unc_t1)
    assert np.all(unc_t2 >= cap5_t2)

    # Force one label from probability ~0 to ~1: the T=2 value reaches
    # exp(2) - 1, so the cap of 5 must bind there.
    src = np.zeros_like(source)
    src[:, 0] = -30.0
    cor = np.zeros_like(corrupt)
    cor[:, :, 0] = 30.0
    uncapped = nte(src, cor, temperature=2.0, cap=None)
    capped = nte(src, cor, temperature=2.0, cap=5.0)
    assert uncapped.max() > 5.0
    assert capped.max() <= 5.0 * cor.shape[0] + 1e-4
    assert capped.max() < uncapped.max()
