"""Tests for the Plan A full profile: ResNet-18, config, checkpoints, aug."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch

from ebba import models


ROOT = Path(__file__).resolve().parents[1]


def test_resnet18_cifar_forward_and_size():
    model = models.build_model("resnet18")
    assert isinstance(model, models.ResNet18CIFAR)
    n_params = sum(p.numel() for p in model.parameters())
    assert 11_000_000 < n_params < 11_300_000
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_build_model_unknown_rejected():
    try:
        models.build_model("not-a-model")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown model name must raise")


def test_vectorized_augmentation_shape_and_determinism():
    x = torch.randn(7, 3, 32, 32)
    a = models.augment_batch_vectorized(x, torch.Generator().manual_seed(11))
    b = models.augment_batch_vectorized(x, torch.Generator().manual_seed(11))
    assert a.shape == x.shape
    assert a.dtype == x.dtype
    assert torch.equal(a, b)


def test_checkpoint_roundtrip_records_model_name(tmp_path):
    model = models.build_model("resnet18")
    path = tmp_path / "resnet18.pt"
    models.save_checkpoint(
        path, model, {"model_name": "resnet18", "attack": "clean"}, history=[]
    )
    loaded, meta = models.load_checkpoint(path)
    assert meta["model_name"] == "resnet18"
    assert isinstance(loaded, models.ResNet18CIFAR)
    for (name_a, param_a), (name_b, param_b) in zip(
        model.state_dict().items(), loaded.state_dict().items()
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)


def test_full_profile_config_paths_and_hyperparameters():
    env = dict(os.environ)
    env["EBBA_PROFILE"] = "full"
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import json; from ebba import config; "
        "print(json.dumps({'profile': config.PROFILE, 'full': config.IS_FULL, "
        "'model': config.MODEL_NAME, 'epochs': config.EPOCHS, "
        "'train_per_class': config.TRAIN_PER_CLASS, "
        "'test_per_class': config.TEST_PER_CLASS, "
        "'artifacts': config.ARTIFACTS_DIR.name, "
        "'reports': config.REPORTS_DIR.name, "
        "'work': config.WORK_DIR.name, "
        "'batch': config.BATCH_SIZE, 'lr': config.LEARNING_RATE, "
        "'milestones': list(config.LR_MILESTONES)}))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], env=env, cwd=ROOT)
    import json

    rec = json.loads(out)
    assert rec == {
        "profile": "full",
        "full": True,
        "model": "resnet18",
        "epochs": 200,
        "train_per_class": 5000,
        "test_per_class": 1000,
        "artifacts": "artifacts-full",
        "reports": "reports-full",
        "work": "checkpoints-full",
        "batch": 64,
        "lr": 0.01,
        "milestones": [50, 100, 150],
    }


def test_full_profile_uses_paper_triggers():
    """Black BadNets block and alpha 0.2 for the full profile (P1)."""
    env = dict(os.environ)
    env["EBBA_PROFILE"] = "full"
    env["PYTHONPATH"] = str(ROOT / "src")
    env.pop("EBBA_BADNETS_VALUE", None)
    env.pop("EBBA_BLEND_PATTERN", None)
    code = (
        "import json, numpy as np; from ebba import config, attacks; "
        "pat = attacks.blend_pattern(32); "
        "dirty = attacks.apply_badnets(np.full((1, 32, 32, 3), 128, np.uint8)); "
        "print(json.dumps({'value': config.BADNETS_PATCH_VALUE, "
        "'alpha': config.BLEND_ALPHA, "
        "'pattern_shape': list(pat.shape), "
        "'pattern_file': config.BLEND_PATTERN_FILE, "
        "'patch_sum': int(dirty[0, -3:, -3:, :].sum()), "
        "'outside_sum': int(dirty[0, :-3, :-3, :].sum())}))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], env=env, cwd=ROOT)
    import json

    rec = json.loads(out)
    assert rec["value"] == 0
    assert rec["patch_sum"] == 0                    # black block, not white
    assert rec["outside_sum"] > 0                   # only the corner is changed
    assert rec["alpha"] == 0.20
    assert rec["pattern_shape"] == [32, 32, 3]
    if rec["pattern_file"]:
        assert rec["pattern_file"].endswith("blend_hello_kitty.npy")


def test_blend_pattern_loads_explicit_npy(tmp_path):
    import numpy as np

    from ebba import attacks

    array = np.full((32, 32, 3), 7, dtype=np.uint8)
    path = tmp_path / "pattern.npy"
    np.save(path, array)
    got = attacks.blend_pattern(32, path=path)
    assert got.shape == (32, 32, 3)
    assert np.allclose(got, 7)

    np.save(path, np.zeros((16, 16, 3), dtype=np.uint8))
    try:
        attacks.blend_pattern(32, path=path)
    except ValueError:
        pass
    else:
        raise AssertionError("a wrong-sized pattern must raise")


def test_missing_blend_pattern_raises(tmp_path):
    from ebba import attacks

    try:
        attacks.blend_pattern(32, path=tmp_path / "nope.npy")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a missing pattern file must raise")
