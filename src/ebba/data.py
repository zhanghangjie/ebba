"""Dataset preparation: fixed CIFAR-10 subsets and a task-agnostic proxy set.

The paper's "Internet images" are replaced by CIFAR-100 test images, which
belong to classes disjoint from CIFAR-10.  This is a proxy: it removes
task-specific supervision but shares the natural-image domain.  The limitation
is stated in every notebook and in the final report.
"""

from __future__ import annotations

import json
import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

# Two interchangeable sources of the same CIFAR images:
#   * "torchvision": the canonical pickled batches from cs.toronto.edu
#   * "fastai"     : PNG re-encodings mirrored on S3 (much faster in CN networks)
# Both are lossless with respect to pixel values.  The selected subset differs
# because filenames and original ordering differ; the manifest records the
# source, and no claim depends on a particular image order.
CIFAR_SOURCES = ("fastai", "torchvision")
FASTAI_URLS = {
    "cifar10": "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz",
    "cifar100": "https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz",
}


def _per_class_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        chosen.append(idx[:per_class])
    return np.sort(np.concatenate(chosen))


def _download(url: str, dest: Path, retries: int = 3) -> None:
    """Stream ``url`` to ``dest`` atomically (``.part`` then rename)."""
    dest = Path(dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            # A CDN can close a 200 response early; urllib then reports no
            # error.  Read the gzip stream to EOF so truncation triggers a
            # retry instead of failing later during extraction.
            if dest.name.endswith((".tgz", ".tar.gz", ".gz")):
                import gzip

                with gzip.open(tmp, "rb") as check:
                    while check.read(1 << 20):
                        pass
            tmp.replace(dest)
            return
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise
            print(f"[data] retry {attempt}/{retries - 1} for {url}")


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract only regular files that stay inside ``dest``.

    The archive is third-party data, not code.  Reject absolute names, ``..``
    components, links and device nodes so a tampered mirror cannot write
    outside the data directory.
    """
    dest = Path(dest).resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            target = (dest / member.name).resolve()
            if dest != target and dest not in target.parents:
                raise RuntimeError(f"unsafe path in {archive.name}: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                raise RuntimeError(f"unreadable member: {member.name}")
            with src, open(target, "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)


def _fastai_subset(
    root: Path,
    dataset: str,
    split: str,
    per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a per-class subset from an extracted fast.ai CIFAR tree."""
    split_dir = Path(root) / dataset / split
    # CIFAR-10 is <split>/<class>/*.png; the CIFAR-100 mirror adds a
    # superclass level, so group by the directory that actually holds files.
    groups: dict[Path, list[Path]] = {}
    for png in split_dir.rglob("*.png"):
        groups.setdefault(png.parent, []).append(png)
    class_dirs = sorted(groups)
    if not class_dirs:
        raise FileNotFoundError(f"no PNG class folders under {split_dir}")

    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for label, cls_dir in enumerate(class_dirs):
        files = sorted(groups[cls_dir])
        take = min(per_class, len(files))
        chosen = np.sort(rng.choice(len(files), size=take, replace=False))
        for i in chosen:
            with Image.open(files[int(i)]) as im:
                xs.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
            ys.append(label)
    if not xs:
        raise RuntimeError(f"empty subset for {dataset}/{split}")
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def _build_fastai_subsets(data_dir: Path, download: bool) -> dict[str, Path]:
    """Download the S3 PNG mirrors and build the same three .npz files."""
    raw = data_dir / "raw" / "fastai"
    raw.mkdir(parents=True, exist_ok=True)
    extracted = data_dir / "raw" / "fastai_extracted"

    for name, url in FASTAI_URLS.items():
        archive = raw / f"{name}.tgz"
        if not archive.exists():
            if not download:
                raise FileNotFoundError(f"{archive} missing and download=False")
            print(f"[data] downloading {url}")
            _download(url, archive)
        marker = extracted / f".{name}.ok"
        if not marker.exists():
            print(f"[data] extracting {archive.name}")
            _safe_extract(archive, extracted)
            marker.write_text("ok\n", encoding="utf-8")

    x_tr, y_tr = _fastai_subset(extracted, "cifar10", "train", config.TRAIN_PER_CLASS, config.SEED)
    x_te, y_te = _fastai_subset(extracted, "cifar10", "test", config.TEST_PER_CLASS, config.SEED + 1)
    x_ood, y_ood = _fastai_subset(extracted, "cifar100", "test", config.OOD_PER_CLASS, config.SEED + 2)

    paths = {
        "train": data_dir / "cifar10_train.npz",
        "test": data_dir / "cifar10_test.npz",
        "ood": data_dir / "cifar100_ood.npz",
    }
    np.savez_compressed(paths["train"], x=x_tr, y=y_tr, indices=np.arange(len(x_tr)))
    np.savez_compressed(paths["test"], x=x_te, y=y_te, indices=np.arange(len(x_te)))
    np.savez_compressed(paths["ood"], x=x_ood, y=y_ood, indices=np.arange(len(x_ood)))
    return paths


def _fastai_all(root: Path, dataset: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Read every image from a fast.ai CIFAR tree (full-split builder).

    The proxy subset functions deliberately take a fixed number per class.
    This variant keeps all images, which is what the paper's 50k/10k CIFAR-10
    protocol uses.  Class order follows the sorted directory names, as in the
    proxy path; CIFAR-100's extra superclass level is handled by grouping on
    the directory that actually contains the PNGs.
    """
    split_dir = Path(root) / dataset / split
    groups: dict[Path, list[Path]] = {}
    for png in split_dir.rglob("*.png"):
        groups.setdefault(png.parent, []).append(png)
    class_dirs = sorted(groups)
    if not class_dirs:
        raise FileNotFoundError(f"no PNG class folders under {split_dir}")

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for label, cls_dir in enumerate(class_dirs):
        for path in sorted(groups[cls_dir]):
            with Image.open(path) as im:
                xs.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
            ys.append(label)
    if not xs:
        raise RuntimeError(f"empty split for {dataset}/{split}")
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def _gtsrb_ood(raw: Path, download: bool) -> tuple[np.ndarray, np.ndarray] | None:
    """Load the GTSRB test split as an additional task-agnostic image source.

    GTSRB is a paper dataset and provides a genuinely different visual domain
    from CIFAR-100.  torchvision downloads it on demand; the image sizes vary,
    so every image is resized to 32x32 to match the CIFAR input geometry.  If
    the download is unavailable the caller simply keeps CIFAR-100 and records
    that fact in ``full_meta.json``.
    """
    try:
        import torchvision

        dataset = torchvision.datasets.GTSRB(root=raw, split="test", download=download)
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for image, label in dataset:
            image = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
            xs.append(np.asarray(image, dtype=np.uint8))
            ys.append(int(label))
        return np.stack(xs), np.asarray(ys, dtype=np.int64)
    except Exception as exc:  # network / mirror / torchvision version
        print(f"[data] GTSRB unavailable ({type(exc).__name__}: {exc}); using CIFAR-100 only")
        return None


def build_full_subsets(
    data_dir: Path = config.DATA_DIR,
    download: bool = True,
    source: str | None = None,
    include_gtsrb: bool | None = None,
) -> dict[str, Path]:
    """Create the full CIFAR-10 50k/10k splits plus a task-agnostic OOD set.

    Layout is intentionally identical to the proxy builder (same filenames) so
    the existing loaders and claim code work unchanged.  The full profile
    differs only in the *contents*: all training/test images, and CIFAR-100
    test (optionally concatenated with GTSRB test) for OOD.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    source = (source or os.environ.get("EBBA_CIFAR_SOURCE", "fastai")).lower()
    if source not in CIFAR_SOURCES:
        raise ValueError(f"unknown CIFAR source: {source}")
    include_gtsrb = config.OOD_GTSRB if include_gtsrb is None else bool(include_gtsrb)

    if source == "fastai":
        raw = data_dir / "raw" / "fastai"
        raw.mkdir(parents=True, exist_ok=True)
        extracted = data_dir / "raw" / "fastai_extracted"
        for name, url in FASTAI_URLS.items():
            archive = raw / f"{name}.tgz"
            if not archive.exists():
                if not download:
                    raise FileNotFoundError(f"{archive} missing and download=False")
                print(f"[data] downloading {url}")
                _download(url, archive)
            marker = extracted / f".{name}.ok"
            if not marker.exists():
                print(f"[data] extracting {archive.name}")
                _safe_extract(archive, extracted)
                marker.write_text("ok\n", encoding="utf-8")
        print("[data] reading full CIFAR-10 train (50k PNGs) ...")
        x_tr, y_tr = _fastai_all(extracted, "cifar10", "train")
        print("[data] reading full CIFAR-10 test (10k PNGs) ...")
        x_te, y_te = _fastai_all(extracted, "cifar10", "test")
        print("[data] reading full CIFAR-100 test (10k PNGs) ...")
        x_ood, y_ood = _fastai_all(extracted, "cifar100", "test")
        raw_for_gtsrb = data_dir / "raw"
    else:
        import torchvision

        raw_for_gtsrb = data_dir / "raw"
        train = torchvision.datasets.CIFAR10(root=raw_for_gtsrb, train=True, download=download)
        test = torchvision.datasets.CIFAR10(root=raw_for_gtsrb, train=False, download=download)
        ood = torchvision.datasets.CIFAR100(root=raw_for_gtsrb, train=False, download=download)
        x_tr = np.asarray(train.data, dtype=np.uint8)
        y_tr = np.asarray(train.targets, dtype=np.int64)
        x_te = np.asarray(test.data, dtype=np.uint8)
        y_te = np.asarray(test.targets, dtype=np.int64)
        x_ood = np.asarray(ood.data, dtype=np.uint8)
        y_ood = np.asarray(ood.targets, dtype=np.int64)

    ood_sources = ["cifar100_test"]
    if include_gtsrb:
        gtsrb = _gtsrb_ood(raw_for_gtsrb, download)
        if gtsrb is not None:
            x_g, y_g = gtsrb
            x_ood = np.concatenate([x_ood, x_g], axis=0)
            y_ood = np.concatenate([y_ood, y_g + 100], axis=0)  # disjoint labels
            ood_sources.append("gtsrb_test")

    paths = {
        "train": data_dir / "cifar10_train.npz",
        "test": data_dir / "cifar10_test.npz",
        "ood": data_dir / "cifar100_ood.npz",
    }
    np.savez_compressed(paths["train"], x=x_tr, y=y_tr, indices=np.arange(len(x_tr)))
    np.savez_compressed(paths["test"], x=x_te, y=y_te, indices=np.arange(len(x_te)))
    np.savez_compressed(paths["ood"], x=x_ood, y=y_ood, indices=np.arange(len(x_ood)))
    meta = {
        "profile": "full",
        "source": source,
        "train": int(len(x_tr)),
        "test": int(len(x_te)),
        "ood": int(len(x_ood)),
        "ood_sources": ood_sources,
        "seed": config.SEED,
    }
    (data_dir / "full_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[data] full splits written to {data_dir}: {meta}")
    return paths


def build_subsets(
    data_dir: Path = config.DATA_DIR,
    download: bool = True,
    source: str | None = None,
) -> dict[str, Path]:
    """Create deterministic CIFAR-10 / CIFAR-100 subsets as .npz files.

    ``source`` defaults to ``EBBA_CIFAR_SOURCE`` or ``fastai``.  ``torchvision``
    downloads the canonical pickles from cs.toronto.edu, which is frequently
    rate-limited; the fast.ai mirror holds the same images as lossless PNGs.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    source = (source or os.environ.get("EBBA_CIFAR_SOURCE", "fastai")).lower()
    if source not in CIFAR_SOURCES:
        raise ValueError(f"unknown CIFAR source: {source}")
    if source == "fastai":
        return _build_fastai_subsets(data_dir, download)

    import torchvision  # only needed for the canonical pickled batches

    raw = data_dir / "raw"
    train = torchvision.datasets.CIFAR10(root=raw, train=True, download=download)
    test = torchvision.datasets.CIFAR10(root=raw, train=False, download=download)
    ood = torchvision.datasets.CIFAR100(root=raw, train=False, download=download)

    x_tr = np.asarray(train.data, dtype=np.uint8)
    y_tr = np.asarray(train.targets, dtype=np.int64)
    x_te = np.asarray(test.data, dtype=np.uint8)
    y_te = np.asarray(test.targets, dtype=np.int64)
    x_ood = np.asarray(ood.data, dtype=np.uint8)
    y_ood = np.asarray(ood.targets, dtype=np.int64)

    tr_idx = _per_class_indices(y_tr, config.TRAIN_PER_CLASS, config.SEED)
    te_idx = _per_class_indices(y_te, config.TEST_PER_CLASS, config.SEED + 1)
    ood_idx = _per_class_indices(y_ood, config.OOD_PER_CLASS, config.SEED + 2)

    paths = {
        "train": data_dir / "cifar10_train.npz",
        "test": data_dir / "cifar10_test.npz",
        "ood": data_dir / "cifar100_ood.npz",
    }
    np.savez_compressed(paths["train"], x=x_tr[tr_idx], y=y_tr[tr_idx], indices=tr_idx)
    np.savez_compressed(paths["test"], x=x_te[te_idx], y=y_te[te_idx], indices=te_idx)
    np.savez_compressed(paths["ood"], x=x_ood[ood_idx], y=y_ood[ood_idx], indices=ood_idx)
    return paths


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    payload = np.load(Path(path))
    x = payload["x"]
    y = payload["y"] if "y" in payload else None
    return x, y


def load_training_data(attack: str, data_dir: Path = config.DATA_DIR) -> dict:
    """Load the fixed subset and apply the requested poisoning protocol."""
    from .attacks import make_poisoned_training_set

    x, y = load_split(Path(data_dir) / "cifar10_train.npz")
    assert y is not None
    if attack == "clean":
        return {"x": x, "y": y, "mask": np.zeros(len(y), dtype=bool)}
    xp, yp, mask = make_poisoned_training_set(x, y, attack)
    return {"x": xp, "y": yp, "mask": mask}


def load_test_data(data_dir: Path = config.DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    x, y = load_split(Path(data_dir) / "cifar10_test.npz")
    assert y is not None
    return x, y


def load_ood_data(data_dir: Path = config.DATA_DIR) -> np.ndarray:
    x, _ = load_split(Path(data_dir) / "cifar100_ood.npz")
    return x


def balanced_select_by_pseudo(
    logits: np.ndarray,
    n_per_class: int = config.OOD_MIN_PER_CLASS,
    seed: int = config.SEED,
    max_available: int | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    """Select up to ``n_per_class`` images per pseudo-label (paper's step 1)."""
    pred = logits.argmax(axis=1)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    counts: dict[int, int] = {}
    for cls in range(config.NUM_CLASSES):
        idx = np.flatnonzero(pred == cls)
        rng.shuffle(idx)
        take = min(n_per_class, len(idx))
        chosen.extend(idx[:take].tolist())
        counts[cls] = take
    chosen = np.asarray(sorted(chosen), dtype=np.int64)
    if max_available is not None:
        chosen = chosen[:max_available]
    return chosen, counts


def imbalanced_select(logits: np.ndarray, total: int = 500, seed: int = config.SEED) -> np.ndarray:
    """A deliberately biased selection used for the Fig. 78/79 ablation."""
    pred = logits.argmax(axis=1)
    # Keep only the three most frequent pseudo-labels to create skew.
    values, counts = np.unique(pred, return_counts=True)
    keep = values[np.argsort(counts)[-3:]]
    idx = np.flatnonzero(np.isin(pred, keep))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    return np.sort(idx[: min(total, len(idx))])


def make_candidate_set(
    clean_images: np.ndarray,
    clean_labels: np.ndarray,
    poisoned_images: np.ndarray,
    target: int,
    n_poisoned: int = config.NTE_NUM_POISONED,
    n_clean_target: int = config.NTE_NUM_CLEAN_TARGET,
    seed: int = config.SEED,
) -> dict:
    """Build the EBBA+ candidate pool: poisoned images + clean target images.

    Ground truth is retained for PDR/F1; the method itself never sees it.
    """
    rng = np.random.default_rng(seed)
    non_target = np.flatnonzero(clean_labels != target)
    target_idx = np.flatnonzero(clean_labels == target)
    p_idx = rng.choice(non_target, size=min(n_poisoned, len(non_target)), replace=False)
    c_idx = rng.choice(target_idx, size=min(n_clean_target, len(target_idx)), replace=False)

    images = np.concatenate([poisoned_images[p_idx], clean_images[c_idx]], axis=0)
    is_poisoned = np.concatenate(
        [np.ones(len(p_idx), dtype=bool), np.zeros(len(c_idx), dtype=bool)]
    )
    gt = np.concatenate([clean_labels[p_idx], clean_labels[c_idx]])
    return {
        "images": images,
        "is_poisoned": is_poisoned,
        "ground_truth": gt,
        "source_index": np.concatenate([p_idx, c_idx]),
    }
