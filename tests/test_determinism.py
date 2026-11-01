"""Seeding and per-item RNG helpers."""

from __future__ import annotations

import numpy as np

from ebba.corruptions import corrupt_one
from ebba.determinism import rng_for, set_global_seed


def test_rng_for_is_stable_and_keyed():
    a = rng_for("img", "snow", 3).random(5)
    b = rng_for("img", "snow", 3).random(5)
    c = rng_for("img", "snow", 4).random(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_rng_for_seed_argument():
    a = rng_for("x", seed=1).random(3)
    b = rng_for("x", seed=2).random(3)
    assert not np.array_equal(a, b)


def test_global_seed_reproduces_numpy():
    set_global_seed(123)
    a = np.random.rand(4)
    set_global_seed(123)
    b = np.random.rand(4)
    assert np.array_equal(a, b)


def test_corruption_order_independence():
    """The result for image 0 must not depend on what else is in the batch."""
    img = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    solo = corrupt_one(img, "fog", 2, image_id=0)
    # Re-seeding / generating other images in between must not change it.
    _ = corrupt_one(img, "spatter", 4, image_id=99)
    again = corrupt_one(img, "fog", 2, image_id=0)
    assert np.array_equal(solo, again)
