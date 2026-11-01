"""Artifact paths, manifest validation and convenience loaders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import config, data, models


class ArtifactError(RuntimeError):
    pass


REQUIRED_FILES = (
    "data/cifar10_train.npz",
    "data/cifar10_test.npz",
    "data/cifar100_ood.npz",
    "checkpoints/clean.pt",
    "checkpoints/badnets.pt",
    "checkpoints/blend.pt",
    "manifest.json",
)


def ensure_artifacts(artifacts_dir: Path = config.ARTIFACTS_DIR) -> dict:
    artifacts_dir = Path(artifacts_dir)
    missing = [p for p in REQUIRED_FILES if not (artifacts_dir / p).exists()]
    if missing:
        raise ArtifactError(
            "Missing artifacts: " + ", ".join(missing) +
            "\nRun `python scripts/build_artifacts.py` (inside the image this is done at build time)."
        )
    manifest = json.loads((artifacts_dir / "manifest.json").read_text(encoding="utf-8"))
    recorded_profile = manifest.get("profile")
    if recorded_profile is not None and recorded_profile != config.PROFILE:
        raise ArtifactError(
            f"Artifact profile mismatch: manifest={recorded_profile}, "
            f"EBBA_PROFILE={config.PROFILE}. Use the matching profile/image."
        )
    for rel, expected in manifest.get("sha256", {}).items():
        path = artifacts_dir / rel
        actual = models.sha256_file(path)
        if actual != expected:
            raise ArtifactError(f"Checksum mismatch for {rel}: {actual} != {expected}")
    return manifest


def load_model(
    attack: str,
    device: torch.device | str | None = None,
    artifacts_dir: Path = config.ARTIFACTS_DIR,
):
    from .determinism import get_device

    ensure_artifacts(artifacts_dir)
    path = Path(artifacts_dir) / "checkpoints" / config.CHECKPOINTS[attack]
    model, meta = models.load_checkpoint(path, map_location="cpu")
    model.to(device if device is not None else get_device())
    return model, meta


def load_data(artifacts_dir: Path = config.ARTIFACTS_DIR) -> dict[str, np.ndarray]:
    data_dir = Path(artifacts_dir) / "data"
    x_train, y_train = data.load_split(data_dir / "cifar10_train.npz")
    x_test, y_test = data.load_split(data_dir / "cifar10_test.npz")
    x_ood, _ = data.load_split(data_dir / "cifar100_ood.npz")
    assert y_train is not None and y_test is not None
    return {
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "y_test": y_test,
        "x_ood": x_ood,
    }


def model_metrics(
    model,
    x_clean: np.ndarray,
    y_clean: np.ndarray,
    x_poisoned: np.ndarray,
    target: int = config.TARGET_LABEL,
    device: torch.device | str = "cpu",
) -> dict:
    clean_logits = models.predict_logits(model, x_clean, device)
    poison_logits = models.predict_logits(model, x_poisoned, device)
    return {
        "ba": float((clean_logits.argmax(1) == y_clean).mean()),
        "asr": float((poison_logits.argmax(1) == target).mean()),
        "clean_logits": clean_logits,
        "poison_logits": poison_logits,
    }


def write_manifest(
    artifacts_dir: Path,
    metrics: dict,
    extra: dict | None = None,
) -> dict:
    artifacts_dir = Path(artifacts_dir)
    sha = {}
    for rel in REQUIRED_FILES:
        if rel == "manifest.json":
            continue
        path = artifacts_dir / rel
        if path.exists():
            sha[rel] = models.sha256_file(path)
    manifest = {
        "profile": config.PROFILE,
        "model_name": config.MODEL_NAME,
        "seed": config.SEED,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "protocol": {
            "train_per_class": config.TRAIN_PER_CLASS,
            "test_per_class": config.TEST_PER_CLASS,
            "ood_per_class": config.OOD_PER_CLASS,
            "epochs": config.EPOCHS,
            "poison_rate": config.POISON_RATE,
            "target": config.TARGET_LABEL,
            "temperature": config.TEMPERATURE,
            "lambda": config.LAMBDA,
            "proxy_lambda": config.PROXY_LAMBDA,
            "robust_augment": config.ROBUST_AUG,
        },
        "metrics": metrics,
        "sha256": sha,
        "extra": extra or {},
    }
    (artifacts_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
