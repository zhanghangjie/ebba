"""Global configuration for the EBBA verification project.

All paths and experiment constants live here so notebooks and scripts share a
single source of truth.  Values can be overridden with environment variables
when the code is executed outside the container.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Profile: the default "proxy" tier is the fast CIFAR-10 subset + SmallCNN.
# "full" adds the paper backbone (ResNet-18), the full CIFAR-10 splits and a
# 200-epoch paper-style recipe.  It is opt-in through EBBA_PROFILE=full and
# uses separate artifacts/report directories so the two runs never overwrite
# each other.
# ---------------------------------------------------------------------------
PROFILE = os.environ.get("EBBA_PROFILE", "proxy").strip().lower()
if PROFILE not in ("proxy", "full"):
    raise ValueError(f"EBBA_PROFILE must be 'proxy' or 'full', got {PROFILE!r}")
IS_FULL = PROFILE == "full"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("EBBA_ROOT", Path(__file__).resolve().parents[2]))
_DEFAULT_ARTIFACTS = PROJECT_ROOT / ("artifacts-full" if IS_FULL else "artifacts")
_DEFAULT_REPORTS = PROJECT_ROOT / ("reports-full" if IS_FULL else "reports")
ARTIFACTS_DIR = Path(os.environ.get("EBBA_ARTIFACTS", _DEFAULT_ARTIFACTS))
REPORTS_DIR = Path(os.environ.get("EBBA_REPORTS", _DEFAULT_REPORTS))
DATA_DIR = ARTIFACTS_DIR / "data"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
# Training workspaces (host-mounted) are deliberately outside ARTIFACTS_DIR so
# a half-written checkpoint is never hashed as a final artifact.
WORK_DIR = Path(os.environ.get("EBBA_WORK", PROJECT_ROOT / ("checkpoints-full" if IS_FULL else "checkpoints")))

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 2024
NUM_THREADS = int(os.environ.get("EBBA_THREADS", "8" if IS_FULL else "4"))

# ---------------------------------------------------------------------------
# Data protocol
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
# Proxy: fixed balanced subsets.  Full: every training/test image.
TRAIN_PER_CLASS = 2000 if not IS_FULL else 5000
TEST_PER_CLASS = 200 if not IS_FULL else 1000
OOD_PER_CLASS = 50 if not IS_FULL else 1000
OOD_MIN_PER_CLASS = 50 if not IS_FULL else 100
FULL_TRAIN_SIZE = 50_000
FULL_TEST_SIZE = 10_000
OOD_GTSRB = os.environ.get("EBBA_OOD_GTSRB", "0") == "1"
# CIFAR-100 test is 10k.  With OOD_GTSRB the full OOD file becomes 22,630
# (10k CIFAR-100 + 12,630 GTSRB test images resized to 32x32, labels +100).
FULL_OOD_SIZE = 22_630 if OOD_GTSRB else 10_000

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# ---------------------------------------------------------------------------
# Model / training
# ---------------------------------------------------------------------------
MODEL_NAME = "resnet18" if IS_FULL else "smallcnn"
EPOCHS = 200 if IS_FULL else 30
# The proxy uses a light corruption augmentation to make the small model
# robust enough for the NTE signal.  The full run follows the paper recipe:
# standard random crop + horizontal flip only.
ROBUST_AUG = not IS_FULL
BATCH_SIZE = 64 if IS_FULL else 128
LEARNING_RATE = 0.01 if IS_FULL else 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LR_MILESTONES = (50, 100, 150) if IS_FULL else ()
LR_GAMMA = 0.1
AMP = IS_FULL              # mixed precision is used for the GPU full run

# ---------------------------------------------------------------------------
# Backdoor attacks
# ---------------------------------------------------------------------------
TARGET_LABEL = 0
POISON_RATE = 0.10
BADNETS_PATCH = 3
# The paper's implementation detail section specifies a *black* block for
# BadNets and the "hello kitty" image for Blend (alpha=0.2).  The proxy
# checkpoints were trained with the earlier white 3x3 patch and a procedural
# Blend pattern, so the defaults keep those values for the proxy profile and
# use the paper's trigger for the full profile.  Both can be overridden for an
# explicit experiment; the training config hash records the choice.
BADNETS_PATCH_VALUE = int(
    os.environ.get("EBBA_BADNETS_VALUE", "255" if not IS_FULL else "0")
)
_PATTERN_ENV = os.environ.get("EBBA_BLEND_PATTERN", "").strip()
# Ship-ready default: if the 32x32 hello-kitty array produced by
# scripts/prepare_trigger.py sits next to the full artifacts, use it without
# requiring an environment variable.  A missing file is not an error for the
# proxy profile, which keeps its procedural trigger.
_PATTERN_DEFAULT = DATA_DIR / "blend_hello_kitty.npy" if IS_FULL else None
BLEND_PATTERN_FILE = _PATTERN_ENV or (
    str(_PATTERN_DEFAULT) if _PATTERN_DEFAULT is not None and _PATTERN_DEFAULT.exists() else ""
)
BLEND_ALPHA = 0.20

ATTACKS = ("badnets", "blend")
CHECKPOINTS = {
    "clean": "clean.pt",
    "badnets": "badnets.pt",
    "blend": "blend.pt",
}

# ---------------------------------------------------------------------------
# EBBA hyper-parameters (paper defaults)
# ---------------------------------------------------------------------------
TEMPERATURE = 2.0
LAMBDA = 3.0            # paper value (Eq. 8 / Table 2)
# The proxy uses a separate threshold because lambda=3 is unreachable on
# CIFAR-10: for any K-vector, max_k (E_k - mu)/sigma <= sqrt(K - 1), so with
# K=10 the supremum is exactly 3 and Eq. 8's *strict* margin cannot be met
# (claim A20).  The paper's [1.8, 5.7] interval matches GTSRB's K=43 bound.
# PROXY_LAMBDA sits between the measured clean z (1.77) and backdoored z
# (2.66); the exact-tier claims still use the paper value LAMBDA=3.
PROXY_LAMBDA = 2.2
BIRCH_THRESHOLD = 0.5
CORRUPTION_SEVERITIES = (1, 2, 3, 4)

# Fast mode only affects the proxy.  The full profile uses larger samples so
# PDR/F1 and removal estimates are closer to the paper's test-set scale.
FAST = (not IS_FULL) and os.environ.get("EBBA_FAST", "0") == "1"
if IS_FULL:
    # The paper defines PDR over all poisoned samples, not a 2:1 subsample.
    # Using 5000 poisoned test images plus all 1000 clean target-class test
    # images keeps F1's prevalence closer to the paper than the earlier
    # 1000/500 pool, while staying within one GPU pass.
    NTE_NUM_POISONED = 5000
    NTE_NUM_CLEAN_TARGET = 1000
    LEMMA2_IMAGES = 500
    CLUSTER_SAMPLES = 6000
else:
    NTE_NUM_POISONED = 120 if FAST else 200
    NTE_NUM_CLEAN_TARGET = 60 if FAST else 100
    LEMMA2_IMAGES = 60 if FAST else 100
    CLUSTER_SAMPLES = 240 if FAST else 300
