"""Seed handling and deterministic RNG helpers.

The verification never relies on global RNG state for corruption sampling:
each stochastic transform derives its own generator from a stable hash of
(image id, transform name, severity).  This makes results independent of batch
order and worker count.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

try:  # torch is optional for the corruption / formula unit tests
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal envs
    torch = None

from . import config


def set_global_seed(seed: int = config.SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for this operation")
    return torch


def configure_torch(threads: int = config.NUM_THREADS) -> None:
    """Pin thread counts and request deterministic kernels.

    Full bitwise determinism across different CPU models is not guaranteed by
    PyTorch; the report only claims metric-level reproducibility on the same
    image.  ``warn_only=True`` keeps unsupported ops usable while recording
    that they may be non-deterministic.
    """
    torch = _require_torch()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_for(*parts: object, seed: int = config.SEED) -> np.random.Generator:
    """Return a NumPy generator keyed by a stable hash of ``parts``."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8, person=b"ebba").digest()
    sub_seed = int.from_bytes(digest, "little")
    return np.random.default_rng(seed ^ sub_seed)


def get_device(preference: str | None = None) -> torch.device:
    """Pick a device from ``cpu`` / ``cuda`` / ``auto``.

    Falls back to CPU with a printed notice when CUDA is unavailable, so the
    same notebook works in both container variants.
    """
    if torch is None:
        return "cpu"
    pref = (preference or os.environ.get("EBBA_DEVICE", "auto")).lower()
    if pref == "cuda" or pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "cuda":
            print("[ebba] CUDA requested but unavailable; falling back to CPU")
    return torch.device("cpu")
