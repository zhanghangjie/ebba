"""Small CNN and inference/training helpers.

The paper uses ResNet18 for 200 epochs.  A scaled proxy is necessary to keep
container runtime in minutes; the architecture below is deliberately small but
strong enough on CIFAR-10 to exhibit the energy anomaly the paper describes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


class BasicBlock(nn.Module):
    """CIFAR ResNet basic block (two 3x3 convolutions + identity shortcut)."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet18CIFAR(nn.Module):
    """ResNet-18 with the CIFAR stem (3x3/stride 1, no max-pool).

    This is the standard CIFAR variant used by the paper's 32x32 experiments.
    It has the same 11.2M-parameter backbone as torchvision's ResNet-18 but
    keeps 32x32 spatial resolution through layer1.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out).flatten(1)
        return self.fc(out)


def build_model(name: str | None = None, num_classes: int = config.NUM_CLASSES) -> nn.Module:
    """Construct a model by name (checkpoint metadata uses this)."""
    name = (name or config.MODEL_NAME).lower()
    if name == "smallcnn":
        return SmallCNN(num_classes=num_classes)
    if name == "resnet18":
        return ResNet18CIFAR(num_classes=num_classes)
    raise ValueError(f"unknown model: {name}")


_MEAN = torch.tensor(config.CIFAR10_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(config.CIFAR10_STD).view(1, 3, 1, 1)


def to_tensor(images: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> normalized NCHW float32."""
    x = torch.from_numpy(np.ascontiguousarray(images)).float().permute(0, 3, 1, 2) / 255.0
    return (x - _MEAN) / _STD


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    images: np.ndarray,
    device: torch.device | str | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    if device is None:
        from .determinism import get_device

        device = get_device()
    device = torch.device(device)
    model.to(device)
    out: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch = to_tensor(images[start:start + batch_size]).to(device)
        out.append(model(batch).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, config.NUM_CLASSES), np.float32)


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def _augment_batch(xb: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Random 4-px-pad crop + horizontal flip, driven by a private generator.

    Without augmentation an 8k-image subset overfits badly (train 0.85 /
    test 0.71) and the proxy model is too weak to be a meaningful stand-in.
    Both transforms use ``g`` so runs stay reproducible.
    """
    n = xb.shape[0]
    padded = F.pad(xb, (4, 4, 4, 4), mode="reflect")
    offsets = torch.randint(0, 9, (n, 2), generator=g)
    out = torch.empty_like(xb)
    for i in range(n):
        dy = int(offsets[i, 0])
        dx = int(offsets[i, 1])
        out[i] = padded[i, :, dy:dy + 32, dx:dx + 32]
    flip = torch.rand(n, generator=g) < 0.5
    if bool(flip.any()):
        out[flip] = torch.flip(out[flip], dims=[3])
    return out


def augment_batch_vectorized(xb: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Random 4-px-pad crop + horizontal flip for a GPU-resident batch.

    The per-sample Python loop in :func:`_augment_batch` is fine for the
    SmallCNN proxy (batch 128, 20k images).  The full run has 782 batches per
    epoch and 200 epochs, so this version uses advanced indexing to build all
    crops at once.  The random offsets and flips still come from the caller's
    private CPU generator, which keeps checkpoint/resume deterministic.
    """
    n = xb.shape[0]
    padded = F.pad(xb, (4, 4, 4, 4), mode="reflect")
    offsets = torch.randint(0, 9, (n, 2), generator=g)
    ar = torch.arange(32, device=xb.device)
    ys = offsets[:, 0, None].to(xb.device) + ar[None, :]
    xs = offsets[:, 1, None].to(xb.device) + ar[None, :]
    idx_n = torch.arange(n, device=xb.device)[:, None, None, None]
    idx_c = torch.arange(xb.shape[1], device=xb.device)[None, :, None, None]
    out = padded[idx_n, idx_c, ys[:, None, :, None], xs[:, None, None, :]]
    flip = (torch.rand(n, generator=g) < 0.5).to(xb.device)
    if bool(flip.any()):
        out[flip] = torch.flip(out[flip], dims=[3])
    return out


def _corruption_augment_batch(xb: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Mild photometric augmentations that mimic low-severity corruptions.

    EBBA+ relies on corruptions destroying the trigger.  A model trained
    without ever seeing noisy/blurred inputs is not robust enough for the NTE
    signal to recover labels (measured: 12% top-1 on corrupted test images),
    so the proxy trains with a light version of the same distribution shift.
    """
    out = xb
    if torch.rand(1, generator=g).item() < 0.5:
        sigma = 0.02 + 0.08 * torch.rand(1, generator=g).item()
        noise = torch.randn(out.shape, generator=g).to(out.device)
        out = out + sigma * noise
    if torch.rand(1, generator=g).item() < 0.25:
        k = torch.tensor([1.0, 2.0, 1.0])
        k = (k[:, None] * k[None, :]) / 16.0
        k = k.expand(3, 1, 3, 3).to(out.device)
        out = F.conv2d(F.pad(out, (1, 1, 1, 1), mode="reflect"), k, groups=3)
    if torch.rand(1, generator=g).item() < 0.30:
        contrast = 0.7 + 0.5 * torch.rand(1, generator=g).item()
        mean = out.mean(dim=(1, 2, 3), keepdim=True)
        out = mean + contrast * (out - mean)
    if torch.rand(1, generator=g).item() < 0.30:
        shift = 0.2 * (2.0 * torch.rand(1, generator=g).item() - 1.0)
        out = out + shift
    return out


def train_model(
    images: np.ndarray,
    labels: np.ndarray,
    epochs: int = config.EPOCHS,
    seed: int = config.SEED,
    device: torch.device | str = "cpu",
    verbose: bool = True,
    augment: bool = True,
    robust: bool = False,
) -> tuple[SmallCNN, list[dict[str, float]]]:
    """Deterministic SGD training with a fixed per-epoch permutation."""
    device = torch.device(device)
    torch.manual_seed(seed)
    model = SmallCNN().to(device)
    x = to_tensor(images).to(device)
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(device)
    g = torch.Generator(device="cpu").manual_seed(seed)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, float]] = []
    n = len(x)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        total_loss = 0.0
        correct = 0
        for start in range(0, n, config.BATCH_SIZE):
            idx = perm[start:start + config.BATCH_SIZE]
            xb = _augment_batch(x[idx], g) if augment else x[idx]
            if robust:
                xb = _corruption_augment_batch(xb, g)
            logits = model(xb)
            loss = F.cross_entropy(logits, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(idx)
            correct += int((logits.argmax(1) == y[idx]).sum())
        scheduler.step()
        rec = {
            "epoch": epoch + 1,
            "loss": total_loss / n,
            "train_acc": correct / n,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(rec)
        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs):
            print(f"[train] epoch {epoch + 1:2d}/{epochs} loss={rec['loss']:.4f} acc={rec['train_acc']:.4f}")
    return model, history


def save_checkpoint(
    path: Path,
    model: nn.Module,
    meta: dict,
    history: list[dict[str, float]] | None = None,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta.setdefault("model_name", config.MODEL_NAME)
    torch.save({"state_dict": model.state_dict(), "meta": meta, "history": history or []}, path)
    return sha256_file(path)


def load_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> tuple[nn.Module, dict]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    meta = payload.get("meta", {})
    # Older proxy checkpoints predate the model_name field; they are SmallCNN.
    model = build_model(meta.get("model_name", "smallcnn"))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, meta


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
