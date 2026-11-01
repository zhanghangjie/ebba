"""Corruption suite: exactly 80 deterministic cases, valid pixel output."""

from __future__ import annotations

import numpy as np

from ebba.corruptions import corrupt_all, corrupt_one, corruption_specs


def _image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)


def test_eighty_specs():
    specs = corruption_specs()
    assert len(specs) == 80
    assert len(set(specs)) == 80
    names = {n for n, _ in specs}
    assert len(names) == 20              # 19 corruptions + divide
    assert sum(1 for n, _ in specs if n == "divide") == 4
    assert {sev for n, sev in specs if n == "divide"} == {2, 3, 4, 5}


def test_corrupt_one_is_deterministic():
    img = _image(1)
    a = corrupt_one(img, "gaussian_noise", 3, image_id=7)
    b = corrupt_one(img, "gaussian_noise", 3, image_id=7)
    assert np.array_equal(a, b)


def test_rng_depends_on_image_id():
    img = _image(2)
    a = corrupt_one(img, "gaussian_noise", 4, image_id=1)
    b = corrupt_one(img, "gaussian_noise", 4, image_id=2)
    assert not np.array_equal(a, b)


def test_output_shape_dtype_range():
    img = _image(3)
    for name, sev in corruption_specs()[:10]:
        out = corrupt_one(img, name, sev, image_id=0)
        assert out.shape == img.shape
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255


def test_divide_is_integer_division():
    img = np.full((4, 4, 3), 100, dtype=np.uint8)
    out = corrupt_one(img, "divide", 4, image_id=0)
    assert out.min() == 25 and out.max() == 25


def test_corrupt_all_stacks_and_preserves_order():
    img = _image(4)
    specs = [("divide", 2), ("divide", 4), ("brightness", 1)]
    out = corrupt_all(img[None], specs=specs)
    assert out.shape == (3, 1, 32, 32, 3)
    # divide 4 < divide 2 < brightness(0.9) for a random uint8 image
    assert out[1].mean() < out[0].mean() < out[2].mean()
