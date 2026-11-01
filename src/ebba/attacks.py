"""Backdoor trigger implementations used by the scaled proxy experiments.

Images are uint8 arrays in HWC layout.  Triggers are applied in pixel space
before normalization, matching the usual BadNets / Blend protocol.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


def blend_pattern(size: int = 32, path: str | Path | None = None) -> np.ndarray:
    """Return the Blend trigger image.

    The paper uses the default "hello kitty" image.  It is not redistributed
    with this repository; set ``EBBA_BLEND_PATTERN`` to a 32x32x3 ``.npy``
    produced by ``scripts/prepare_trigger.py`` to use it.  Without that file we
    fall back to a deterministic pink surrogate and document the deviation.
    """
    path = Path(path) if path else (Path(config.BLEND_PATTERN_FILE) if config.BLEND_PATTERN_FILE else None)
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(
                f"EBBA_BLEND_PATTERN points to {path}, but the file does not exist"
            )
        array = np.load(path)
        if array.shape != (size, size, 3):
            raise ValueError(
                f"Blend pattern {path} has shape {array.shape}; expected {(size, size, 3)}"
            )
        return array.astype(np.float32)

    pattern = np.zeros((size, size, 3), dtype=np.float32)
    pattern[..., 0] = 255.0            # red
    pattern[..., 1] = 105.0            # green
    pattern[..., 2] = 180.0            # blue
    # White "face" disk in the centre.
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2
    face = (yy - cy) ** 2 + (xx - cx) ** 2 < (size * 0.30) ** 2
    pattern[face] = 255.0
    # Black eyes.
    for dy in (-0.10, 0.10):
        eye = (yy - (cy + dy * size)) ** 2 + (xx - (cx - 0.12 * size)) ** 2 < (size * 0.035) ** 2
        eye |= (yy - (cy + dy * size)) ** 2 + (xx - (cx + 0.12 * size)) ** 2 < (size * 0.035) ** 2
        pattern[eye] = 0.0
    return pattern


def apply_badnets(
    images: np.ndarray,
    patch: int = config.BADNETS_PATCH,
    value: int = config.BADNETS_PATCH_VALUE,
) -> np.ndarray:
    """Square in the bottom-right corner (classic BadNets trigger).

    The paper's implementation section specifies a black block; the proxy
    artifacts were trained with white, so the value is profile-dependent.
    """
    out = images.copy()
    out[..., -patch:, -patch:, :] = value
    return out


def apply_blend(
    images: np.ndarray,
    alpha: float = config.BLEND_ALPHA,
    pattern: np.ndarray | None = None,
) -> np.ndarray:
    """Blend a fixed pattern into every image: x' = (1-a)x + a*p."""
    if images.ndim == 3:
        images = images[None, ...]
    pat = blend_pattern(images.shape[1]) if pattern is None else pattern
    mixed = (1.0 - alpha) * images.astype(np.float32) + alpha * pat[None]
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def apply_trigger(images: np.ndarray, attack: str) -> np.ndarray:
    if attack == "badnets":
        return apply_badnets(images)
    if attack == "blend":
        return apply_blend(images)
    raise ValueError(f"unknown attack: {attack}")


def poison_indices(
    labels: np.ndarray,
    target: int = config.TARGET_LABEL,
    rate: float | None = None,
    seed: int = config.SEED,
) -> np.ndarray:
    """Select a class-stratified subset of non-target training images."""
    rate = config.POISON_RATE if rate is None else rate
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for cls in range(config.NUM_CLASSES):
        if cls == target:
            continue
        idx = np.flatnonzero(labels == cls)
        n = int(round(rate * len(idx)))
        if n > 0:
            chosen.extend(rng.choice(idx, size=n, replace=False).tolist())
    return np.sort(np.asarray(chosen, dtype=np.int64))


def make_poisoned_training_set(
    images: np.ndarray,
    labels: np.ndarray,
    attack: str,
    target: int = config.TARGET_LABEL,
    rate: float | None = None,
    seed: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (images, labels, poisoned_mask).

    The original array is not modified.  Poisoned samples receive the target
    label (dirty-label protocol), as in the paper's main experiments.
    """
    idx = poison_indices(labels, target=target, rate=rate, seed=seed)
    x = images.copy()
    y = labels.copy()
    mask = np.zeros(len(labels), dtype=bool)
    if len(idx):
        x[idx] = apply_trigger(x[idx], attack)
        y[idx] = target
        mask[idx] = True
    return x, y, mask


def evaluation_sets(
    images: np.ndarray,
    labels: np.ndarray,
    attack: str,
) -> dict[str, np.ndarray]:
    """Clean and fully poisoned test sets used for BA / ASR / PDR."""
    return {
        "clean_images": images,
        "clean_labels": labels,
        "poisoned_images": apply_trigger(images, attack),
        "poisoned_labels": labels,  # ground truth is needed for recovery
    }
