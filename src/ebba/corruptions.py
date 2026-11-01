"""Deterministic image corruptions for EBBA+ (Eq. 9-11).

The paper states that 80 corruption methods are applied to each candidate
image.  The public reference notebook uses the ``imagecorruptions`` package
(15 types x 4 severities) plus four integer divisions, i.e. 64 cases.  That
package is unmaintained, so this module re-implements 19 corruption families
x 4 severities (76) plus 4 divisions, giving exactly 80 deterministic cases.

Every stochastic corruption derives its RNG from ``(image_id, name, severity)``
so the result does not depend on batch order or worker count.  These are
re-implementations, not byte-identical copies of Hendrycks' corruption suite;
the deviation is recorded in reference/AUDIT.md.
"""

from __future__ import annotations

import io
import math
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance
from scipy import ndimage

from .determinism import rng_for

# ---------------------------------------------------------------------------
# 80 corruption specifications
# ---------------------------------------------------------------------------
_CORRUPTION_NAMES = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "saturate",
    "spatter",
    "speckle_noise",
    "gaussian_blur",
)


def corruption_specs() -> list[tuple[str, int]]:
    """Return the ordered list of 80 (name, severity) pairs."""
    specs: list[tuple[str, int]] = []
    for name in _CORRUPTION_NAMES:
        for sev in (1, 2, 3, 4):
            specs.append((name, sev))
    for n in (2, 3, 4, 5):          # integer division, as in Appendix C
        specs.append(("divide", n))
    assert len(specs) == 80, len(specs)
    return specs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _resize_center(img: np.ndarray, scale: float) -> np.ndarray:
    """Scale an HWC image and centre-crop / zero-pad back to its size."""
    h, w, c = img.shape
    out_h, out_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    zoomed = ndimage.zoom(img, (out_h / h, out_w / w, 1), order=1)
    result = np.zeros_like(img, dtype=np.float32)
    y0 = max(0, (out_h - h) // 2)
    x0 = max(0, (out_w - w) // 2)
    y1 = min(out_h, y0 + h)
    x1 = min(out_w, x0 + w)
    dy0 = max(0, (h - out_h) // 2)
    dx0 = max(0, (w - out_w) // 2)
    result[dy0:dy0 + (y1 - y0), dx0:dx0 + (x1 - x0)] = zoomed[y0:y1, x0:x1]
    return result


def _motion_kernel(length: int, angle_deg: float = 45.0) -> np.ndarray:
    k = np.zeros((length, length), dtype=np.float32)
    c = (length - 1) / 2
    a = math.radians(angle_deg)
    for t in np.linspace(-c, c, length * 2):
        y = int(round(c + t * math.sin(a)))
        x = int(round(c + t * math.cos(a)))
        if 0 <= y < length and 0 <= x < length:
            k[y, x] = 1.0
    return k / max(k.sum(), 1e-6)


def _low_freq_field(shape: tuple[int, int], rng: np.random.Generator, sigma: float) -> np.ndarray:
    noise = rng.random((shape[0], shape[1])).astype(np.float32)
    field = ndimage.gaussian_filter(noise, sigma=sigma, mode="reflect")
    field -= field.min()
    return field / max(field.max(), 1e-6)


# ---------------------------------------------------------------------------
# Individual corruptions (single image, HWC uint8)
# ---------------------------------------------------------------------------
def _gaussian_noise(img, sev, rng):
    sigma = (0.04, 0.06, 0.08, 0.12)[sev - 1] * 255
    return _to_float(img) + rng.normal(0, sigma, img.shape)


def _shot_noise(img, sev, rng):
    scale = (100, 80, 60, 40)[sev - 1]
    vals = rng.poisson(np.clip(_to_float(img), 0, 255) / scale).astype(np.float32) * scale
    return vals


def _impulse_noise(img, sev, rng):
    frac = (0.01, 0.02, 0.03, 0.05)[sev - 1]
    out = _to_float(img)
    mask = rng.random(img.shape[:2])
    salt = mask < frac / 2
    pepper = (mask >= frac / 2) & (mask < frac)
    out[salt] = 255.0
    out[pepper] = 0.0
    return out


def _defocus_blur(img, sev, rng):
    return ndimage.gaussian_filter(_to_float(img), sigma=(1, 2, 3, 4)[sev - 1])


def _glass_blur(img, sev, rng):
    # Local block shuffling followed by a light blur: a cheap, deterministic
    # stand-in for the original glass-blur implementation.
    out = _to_float(img)
    block = (2, 3, 4, 5)[sev - 1]
    h, w, _ = out.shape
    for y in range(0, h, block):
        for x in range(0, w, block):
            patch = out[y:y + block, x:x + block].copy()
            flat = patch.reshape(-1, 3)
            perm = rng.permutation(len(flat))
            out[y:y + block, x:x + block] = flat[perm].reshape(patch.shape)
    return ndimage.gaussian_filter(out, sigma=0.6)


def _motion_blur(img, sev, rng):
    length = (3, 5, 7, 9)[sev - 1]
    kernel = _motion_kernel(length)
    out = _to_float(img)
    for c in range(out.shape[2]):
        out[..., c] = ndimage.convolve(out[..., c], kernel, mode="reflect")
    return out


def _zoom_blur(img, sev, rng):
    factors = (1.05, 1.10, 1.15, 1.20)[sev - 1]
    acc = _to_float(img)
    for f in (1.0 + factors * 0.5, 1.0 + factors, 1.0 + 1.5 * factors):
        acc += _resize_center(_to_float(img), f)
    return acc / 4.0


def _snow(img, sev, rng):
    out = _to_float(img)
    h, w, _ = out.shape
    density = (0.01, 0.02, 0.03, 0.05)[sev - 1]
    n = max(8, int(density * h * w / 4))
    flakes = np.zeros((h, w), dtype=np.float32)
    ys = rng.integers(0, h, n)
    xs = rng.integers(0, w, n)
    flakes[ys, xs] = 255.0
    flakes = ndimage.gaussian_filter(flakes, sigma=(0.8, 1.2, 1.8, 2.5)[sev - 1])
    flakes = np.clip(flakes / max(flakes.max(), 1e-6), 0, 1)
    return out * (1 - 0.7 * flakes[..., None]) + 255 * 0.7 * flakes[..., None]


def _frost(img, sev, rng):
    out = _to_float(img)
    h, w, _ = out.shape
    noise = rng.random((h, w)).astype(np.float32)
    high = noise - ndimage.gaussian_filter(noise, sigma=3)
    high = np.abs(high)
    high /= max(high.max(), 1e-6)
    strength = (0.10, 0.20, 0.30, 0.40)[sev - 1]
    frost = np.clip(high * 2.0, 0, 1) * strength
    return out * (1 - frost[..., None]) + 255 * frost[..., None]


def _fog(img, sev, rng):
    out = _to_float(img)
    h, w, _ = out.shape
    field = _low_freq_field((h, w), rng, sigma=(2, 4, 6, 8)[sev - 1])
    strength = (0.10, 0.20, 0.30, 0.40)[sev - 1]
    alpha = (0.4 + 0.6 * field) * strength
    return out * (1 - alpha[..., None]) + 255 * alpha[..., None]


def _brightness(img, sev, rng):
    factor = (0.90, 0.80, 0.70, 0.60)[sev - 1]
    return _to_float(img) * factor


def _contrast(img, sev, rng):
    factor = (0.90, 0.80, 0.70, 0.60)[sev - 1]
    out = _to_float(img)
    mean = out.mean(axis=(0, 1), keepdims=True)
    return (out - mean) * factor + mean


def _elastic_transform(img, sev, rng):
    out = _to_float(img)
    h, w, _ = out.shape
    alpha = (5, 10, 20, 30)[sev - 1]
    dx = ndimage.gaussian_filter(rng.random((h, w)) * 2 - 1, sigma=4) * alpha
    dy = ndimage.gaussian_filter(rng.random((h, w)) * 2 - 1, sigma=4) * alpha
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.stack([np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1)])
    warped = np.stack(
        [ndimage.map_coordinates(out[..., c], coords, order=1, mode="reflect") for c in range(out.shape[2])],
        axis=-1,
    )
    return warped


def _pixelate(img, sev, rng):
    factor = (2, 3, 4, 6)[sev - 1]
    h, w, _ = img.shape
    small = Image.fromarray(img).resize(
        (max(1, w // factor), max(1, h // factor)), Image.Resampling.BILINEAR
    )
    return np.asarray(small.resize((w, h), Image.Resampling.NEAREST), dtype=np.float32)


def _jpeg_compression(img, sev, rng):
    quality = (50, 35, 25, 15)[sev - 1]
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)


def _saturate(img, sev, rng):
    factor = (0.80, 0.60, 0.40, 0.20)[sev - 1]
    im = Image.fromarray(img)
    return np.asarray(ImageEnhance.Color(im).enhance(factor), dtype=np.float32)


def _spatter(img, sev, rng):
    out = _to_float(img)
    h, w, _ = out.shape
    count = (6, 10, 16, 22)[sev - 1]
    radius = (1.0, 1.5, 2.0, 2.5)[sev - 1]
    for _ in range(count):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        r = max(1, int(round(radius * rng.uniform(0.6, 1.4))))
        yy, xx = np.ogrid[max(0, cy - r):min(h, cy + r + 1), max(0, cx - r):min(w, cx + r + 1)]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        color = 255.0 if rng.random() < 0.5 else 0.0
        out[max(0, cy - r):min(h, cy + r + 1), max(0, cx - r):min(w, cx + r + 1)][mask] = color
    return out


def _speckle_noise(img, sev, rng):
    sigma = (0.05, 0.10, 0.15, 0.20)[sev - 1]
    return _to_float(img) * (1.0 + rng.normal(0, sigma, img.shape))


def _gaussian_blur(img, sev, rng):
    return ndimage.gaussian_filter(_to_float(img), sigma=(0.5, 1.0, 1.5, 2.0)[sev - 1])


_DISPATCH = {
    "gaussian_noise": _gaussian_noise,
    "shot_noise": _shot_noise,
    "impulse_noise": _impulse_noise,
    "defocus_blur": _defocus_blur,
    "glass_blur": _glass_blur,
    "motion_blur": _motion_blur,
    "zoom_blur": _zoom_blur,
    "snow": _snow,
    "frost": _frost,
    "fog": _fog,
    "brightness": _brightness,
    "contrast": _contrast,
    "elastic_transform": _elastic_transform,
    "pixelate": _pixelate,
    "jpeg_compression": _jpeg_compression,
    "saturate": _saturate,
    "spatter": _spatter,
    "speckle_noise": _speckle_noise,
    "gaussian_blur": _gaussian_blur,
}


def corrupt_one(img: np.ndarray, name: str, severity: int, image_id: int) -> np.ndarray:
    """Apply one corruption to one HWC uint8 image."""
    if name == "divide":
        return _to_uint8(_to_float(img) / float(severity))
    rng = rng_for(image_id, name, severity)
    return _to_uint8(_DISPATCH[name](img, severity, rng))


def corrupt_batch(
    images: np.ndarray,
    name: str,
    severity: int,
    image_ids: Iterable[int] | None = None,
) -> np.ndarray:
    """Apply a corruption to a batch of HWC uint8 images."""
    ids = list(range(len(images))) if image_ids is None else list(image_ids)
    assert len(ids) == len(images)
    return np.stack(
        [corrupt_one(img, name, severity, iid) for img, iid in zip(images, ids)],
        axis=0,
    )


def corrupt_all(
    images: np.ndarray,
    specs: list[tuple[str, int]] | None = None,
    image_ids: Iterable[int] | None = None,
    progress: bool = False,
) -> np.ndarray:
    """Return an array of shape (J, N, H, W, C) for all corruption specs."""
    specs = corruption_specs() if specs is None else specs
    ids = list(range(len(images))) if image_ids is None else list(image_ids)
    out = []
    iterator = enumerate(specs)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(list(iterator), desc="corruptions")
        except ImportError:
            pass
    for j, (name, sev) in iterator:
        out.append(corrupt_batch(images, name, sev, ids))
    return np.stack(out, axis=0)
