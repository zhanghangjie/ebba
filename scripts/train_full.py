"""Train the paper-backbone profile (ResNet-18, full CIFAR-10, 200 epochs).

This script is the "full" counterpart of ``scripts/build_artifacts.py``.  It is
deliberately separate from the fast proxy path so a mistake here can never
change the one-command CPU verification (``ebba:cpu``).

What it does
------------
* Reads the full CIFAR-10 50k/10k splits from ``artifacts-full/data``.
* Applies BadNets / Blend poisoning with the same trigger code as the proxy.
* Trains ``ResNet18CIFAR`` with the paper recipe: SGD lr=0.01,
  momentum=0.9, weight decay=5e-4, batch=64, milestones 50/100/150,
  standard random crop + horizontal flip.
* Saves a resumable checkpoint after every epoch (and optionally every N
  steps).  The checkpoint contains model, optimizer, scheduler, AMP scaler,
  epoch/step, the current permutation, and all RNG states.
* Writes the final inference checkpoint and metrics into
  ``artifacts-full/checkpoints`` / ``artifacts-full/metrics``.

Resume semantics
----------------
``--resume auto`` loads ``<work-dir>/<attack>/last.pt`` (or ``last_prev.pt`` if
the primary file is missing) and continues at the next batch.  A checkpoint
written at an epoch boundary resumes at the start of the next epoch; a
checkpoint written mid-epoch resumes at the saved batch and does not replay
already-processed batches.  The config hash is checked, so changing the data,
recipe or attack refuses to resume silently.

Example
-------
    # First run (detached, survives closing the terminal)
    docker compose --profile train up -d ebba-train

    # Stop before sleeping: SIGTERM -> save at the next batch boundary
    docker compose stop ebba-train

    # Resume after waking / rebooting
    docker compose --profile train run --rm ebba-train train-full --all --resume auto
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebba import artifacts, attacks, config, data, determinism, models  # noqa: E402

_STOP = False
_LOG_PATH: Path | None = None


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _handle_signal(signum, frame) -> None:  # noqa: ARG001
    """First SIGINT/SIGTERM asks the loop to checkpoint; second forces exit."""
    global _STOP
    if _STOP:
        raise KeyboardInterrupt
    _STOP = True
    print(
        f"[train-full] signal {signum} received; saving at the next batch boundary "
        "(press again to force quit)",
        flush=True,
    )


def _log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if _LOG_PATH is not None:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _to_cpu(value):
    """Recursively move tensors in optimizer/RNG payloads to CPU."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    return value


def _atomic_save(payload: dict, path: Path) -> None:
    """Write via a temporary file and atomically replace the destination.

    The previous ``last.pt`` is kept as ``last_prev.pt``.  A sleep, power loss
    or container kill therefore leaves either the old checkpoint or the new
    one, never a half-written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    try:
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
    except OSError:
        pass
    if path.exists():
        # last.pt -> last_prev.pt; best.pt -> best_prev.pt.  Keeping the
        # rotation path-specific prevents a best-checkpoint write from
        # clobbering the emergency fallback used by --resume auto.
        prev = path.with_name(path.stem + "_prev" + path.suffix)
        try:
            os.replace(path, prev)
        except OSError:
            pass
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _subset(images: np.ndarray, labels: np.ndarray, limit: int | None, seed: int):
    if limit is None or limit >= len(images):
        return images, labels
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(images), size=limit, replace=False))
    return images[idx], labels[idx]


def _trigger_fingerprint(attack: str) -> dict:
    """Describe the trigger so a changed patch/pattern cannot resume silently."""
    if attack == "badnets":
        return {"kind": "badnets", "patch": config.BADNETS_PATCH,
                "value": config.BADNETS_PATCH_VALUE}
    if attack == "blend":
        path = Path(config.BLEND_PATTERN_FILE) if config.BLEND_PATTERN_FILE else None
        return {
            "kind": "blend",
            "alpha": config.BLEND_ALPHA,
            "pattern": str(path) if path else "surrogate",
            "pattern_sha256": _sha256(path) if path and path.exists() else None,
        }
    return {"kind": attack}


def _config_hash(
    attack: str,
    args: argparse.Namespace,
    data_sha: dict[str, str],
    amp: bool,
) -> str:
    payload = {
        "profile": config.PROFILE,
        "model": "resnet18",
        "attack": attack,
        "data_sha": data_sha,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "milestones": list(args.milestones),
        "gamma": args.gamma,
        "seed": args.seed,
        "amp": amp,
        "augment": args.augment,
        "limit_train": args.limit_train,
        "limit_test": args.limit_test,
        "poison_rate": config.POISON_RATE if attack != "clean" else 0.0,
        "target": config.TARGET_LABEL,
        "trigger": _trigger_fingerprint(attack),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Data / model
# ---------------------------------------------------------------------------
def _ensure_data(data_dir: Path, args: argparse.Namespace) -> dict:
    needed = ("cifar10_train.npz", "cifar10_test.npz", "cifar100_ood.npz")
    if all((data_dir / name).exists() for name in needed):
        meta_path = data_dir / "full_meta.json"
        return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if args.no_download:
        raise FileNotFoundError(
            f"full data missing under {data_dir} and --no-download was given"
        )
    _log(f"full data missing under {data_dir}; building it now")
    data.build_full_subsets(
        data_dir,
        download=True,
        source=args.data_source,
        include_gtsrb=args.ood_gtsrb,
    )
    meta_path = data_dir / "full_meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x, y = data.load_split(path)
    assert y is not None
    return x, y


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    x_clean: torch.Tensor,
    y_clean: torch.Tensor,
    x_poison: torch.Tensor | None,
    target: int,
    batch_size: int = 512,
) -> dict:
    model.eval()
    correct = 0
    asr = 0
    n = len(x_clean)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        logits = model(x_clean[start:end])
        correct += int((logits.argmax(1) == y_clean[start:end]).sum())
        if x_poison is not None:
            p_logits = model(x_poison[start:end])
            asr += int((p_logits.argmax(1) == target).sum())
    return {
        "ba": correct / max(n, 1),
        "asr": (asr / max(n, 1)) if x_poison is not None else None,
    }


def _save_workspace_checkpoint(
    path: Path,
    *,
    attack: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    epoch: int,
    step: int,
    global_step: int,
    perm: torch.Tensor | None,
    generator: torch.Generator,
    history: list[dict],
    best_ba: float,
    best_epoch: int,
    epoch_loss_sum: float,
    epoch_correct: int,
    epoch_seen: int,
    config_hash: str,
    args: argparse.Namespace,
    resume_count: int,
) -> None:
    cuda_rng = None
    if torch.cuda.is_available():
        cuda_rng = _to_cpu(torch.cuda.get_rng_state_all())
    payload = {
        "format": 2,
        "profile": config.PROFILE,
        "model_name": "resnet18",
        "attack": attack,
        "config_hash": config_hash,
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "model": _to_cpu(model.state_dict()),
        "optimizer": _to_cpu(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "perm": perm.detach().cpu() if perm is not None else None,
        "generator": generator.get_state(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": cuda_rng,
        },
        "history": history,
        "best_ba": best_ba,
        "best_epoch": best_epoch,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_correct": epoch_correct,
        "epoch_seen": epoch_seen,
        "resume_count": resume_count,
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "saved_at": time.time(),
    }
    _atomic_save(payload, path)


def _restore_workspace_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    generator: torch.Generator,
    expected_hash: str,
) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("config_hash") != expected_hash:
        raise RuntimeError(
            f"Refusing to resume {path}: checkpoint config hash "
            f"{payload.get('config_hash')} != current {expected_hash}. "
            "Use --resume none or a different --work-dir."
        )
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    generator.set_state(payload["generator"])
    rng = payload.get("rng", {})
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception as exc:  # different GPU count is non-fatal
            print(f"[train-full] could not restore CUDA RNG state: {exc}", flush=True)
    return payload


def _resolve_resume(work_dir: Path, attack: str, mode: str) -> Path | None:
    if mode == "none":
        return None
    if mode == "auto":
        for name in ("last.pt", "last_prev.pt"):
            candidate = work_dir / attack / name
            if candidate.exists():
                return candidate
        return None
    path = Path(mode)
    if path.is_dir():
        path = path / attack / "last.pt"
    if not path.exists():
        raise FileNotFoundError(f"--resume path does not exist: {path}")
    return path


def _train_attack(
    attack: str,
    args: argparse.Namespace,
    data_dir: Path,
    data_sha: dict[str, str],
    device: torch.device,
) -> dict:
    _log(f"=== attack={attack} ===")
    x_train_np, y_train_np = _load_split(data_dir / "cifar10_train.npz")
    x_test_np, y_test_np = _load_split(data_dir / "cifar10_test.npz")
    x_train_np, y_train_np = _subset(x_train_np, y_train_np, args.limit_train, args.seed)
    x_test_np, y_test_np = _subset(x_test_np, y_test_np, args.limit_test, args.seed)

    poison_mask = None
    if attack != "clean":
        x_train_np, y_train_np, poison_mask = attacks.make_poisoned_training_set(
            x_train_np, y_train_np, attack,
            target=config.TARGET_LABEL, rate=config.POISON_RATE, seed=config.SEED,
        )
        x_test_poison_np = attacks.apply_trigger(x_test_np, attack)
    else:
        x_test_poison_np = None

    amp = bool(args.amp) if args.amp is not None else (device.type == "cuda")
    if device.type != "cuda":
        amp = False
    config_hash = _config_hash(attack, args, data_sha, amp)

    x_train = models.to_tensor(x_train_np).to(device)
    y_train = torch.from_numpy(np.asarray(y_train_np, dtype=np.int64)).to(device)
    x_test = models.to_tensor(x_test_np).to(device)
    y_test = torch.from_numpy(np.asarray(y_test_np, dtype=np.int64)).to(device)
    x_test_poison = (
        models.to_tensor(x_test_poison_np).to(device)
        if x_test_poison_np is not None else None
    )
    n = len(x_train)
    steps_per_epoch = math.ceil(n / args.batch_size)
    _log(
        f"train={n} test={len(x_test)} poisoned_train="
        f"{int(poison_mask.sum()) if poison_mask is not None else 0} "
        f"steps/epoch={steps_per_epoch} device={device} amp={amp}"
    )

    model = models.build_model("resnet18", config.NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(args.milestones), gamma=args.gamma
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    history: list[dict] = []
    best_ba = -1.0
    best_epoch = 0
    resume_count = 0
    start_epoch = 0
    step_start = 0
    global_step = 0
    perm: torch.Tensor | None = None
    epoch_loss_sum = 0.0
    epoch_correct = 0
    epoch_seen = 0

    resume_path = _resolve_resume(Path(args.work_dir), attack, args.resume)
    if resume_path is not None:
        payload = _restore_workspace_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if amp else None,
            generator=g,
            expected_hash=config_hash,
        )
        start_epoch = int(payload.get("epoch", 0))
        step_start = int(payload.get("step", 0))
        global_step = int(payload.get("global_step", 0))
        perm = payload.get("perm")
        history = list(payload.get("history", []))
        best_ba = float(payload.get("best_ba", -1.0))
        best_epoch = int(payload.get("best_epoch", 0))
        epoch_loss_sum = float(payload.get("epoch_loss_sum", 0.0))
        epoch_correct = int(payload.get("epoch_correct", 0))
        epoch_seen = int(payload.get("epoch_seen", 0))
        resume_count = int(payload.get("resume_count", 0)) + 1
        _log(
            f"resumed from {resume_path}: epoch={start_epoch} step={step_start} "
            f"global_step={global_step} resume_count={resume_count}"
        )
    else:
        _log(f"starting fresh: config_hash={config_hash}")

    work_path = Path(args.work_dir) / attack / "last.pt"
    best_path = Path(args.work_dir) / attack / "best.pt"

    if start_epoch >= args.epochs:
        _log("checkpoint is already at the requested epoch count; evaluating final model")
        metrics = _evaluate(model, x_test, y_test, x_test_poison, config.TARGET_LABEL)
        metrics.update({"n_train": n, "n_test": len(x_test), "amp": amp})
        _write_final_artifact(attack, model, metrics, history, args, resume_count)
        return metrics

    for epoch in range(start_epoch, args.epochs):
        if perm is None:
            perm = torch.randperm(n, generator=g)
        model.train()
        epoch_start = time.time()
        first_step = step_start if epoch == start_epoch else 0
        if first_step == 0:
            epoch_loss_sum = 0.0
            epoch_correct = 0
            epoch_seen = 0

        for step in range(first_step, steps_per_epoch):
            start = step * args.batch_size
            idx = perm[start:start + args.batch_size]
            xb = x_train[idx]
            if args.augment:
                xb = models.augment_batch_vectorized(xb, g)
            yb = y_train[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            bs = len(idx)
            epoch_loss_sum += float(loss.detach().float().item()) * bs
            epoch_correct += int((logits.detach().argmax(1) == yb).sum())
            epoch_seen += bs
            global_step += 1

            if _STOP or (
                args.checkpoint_every_steps > 0
                and global_step % args.checkpoint_every_steps == 0
            ):
                _save_workspace_checkpoint(
                    work_path,
                    attack=attack, model=model, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler if amp else None,
                    epoch=epoch, step=step + 1, global_step=global_step,
                    perm=perm, generator=g, history=history,
                    best_ba=best_ba, best_epoch=best_epoch,
                    epoch_loss_sum=epoch_loss_sum, epoch_correct=epoch_correct,
                    epoch_seen=epoch_seen, config_hash=config_hash,
                    args=args, resume_count=resume_count,
                )
                _log(
                    f"checkpoint saved at epoch={epoch} step={step + 1}/{steps_per_epoch} "
                    f"global_step={global_step}"
                )
                if _STOP:
                    _log("stop requested; exiting after checkpoint")
                    return {"ba": float("nan"), "asr": None, "stopped": True}

        scheduler.step()
        train_loss = epoch_loss_sum / max(epoch_seen, 1)
        train_acc = epoch_correct / max(epoch_seen, 1)
        should_eval = (
            args.eval_every > 0
            and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs)
        )
        metrics = {"ba": None, "asr": None}
        if should_eval:
            metrics = _evaluate(model, x_test, y_test, x_test_poison, config.TARGET_LABEL)
            if metrics["ba"] is not None and metrics["ba"] > best_ba:
                best_ba = float(metrics["ba"])
                best_epoch = epoch + 1
                _atomic_save(
                    {
                        "format": 2,
                        "model_name": "resnet18",
                        "attack": attack,
                        "epoch": epoch + 1,
                        "model": _to_cpu(model.state_dict()),
                        "meta": {
                            "attack": attack,
                            "metrics": metrics,
                            "epochs": args.epochs,
                            "seed": args.seed,
                            "profile": config.PROFILE,
                            "model_name": "resnet18",
                            "poison_rate": config.POISON_RATE if attack != "clean" else 0.0,
                            "target": config.TARGET_LABEL,
                        },
                    },
                    best_path,
                )
        rec = {
            "epoch": epoch + 1,
            "loss": train_loss,
            "train_acc": train_acc,
            "lr": float(scheduler.get_last_lr()[0]),
            "test_ba": metrics.get("ba"),
            "test_asr": metrics.get("asr"),
            "seconds": round(time.time() - epoch_start, 2),
            "global_step": global_step,
        }
        history.append(rec)
        _log(
            f"epoch {epoch + 1:3d}/{args.epochs} loss={train_loss:.4f} "
            f"acc={train_acc:.4f} lr={rec['lr']:.5f} "
            f"ba={metrics.get('ba')} asr={metrics.get('asr')} "
            f"({rec['seconds']}s)"
        )

        # Epoch-boundary checkpoint: next epoch starts at step 0 with a fresh
        # permutation.  Save after evaluation so history is complete.
        _save_workspace_checkpoint(
            work_path,
            attack=attack, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler if amp else None,
            epoch=epoch + 1, step=0, global_step=global_step,
            perm=None, generator=g, history=history,
            best_ba=best_ba, best_epoch=best_epoch,
            epoch_loss_sum=0.0, epoch_correct=0, epoch_seen=0,
            config_hash=config_hash, args=args, resume_count=resume_count,
        )
        perm = None
        step_start = 0
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_seen = 0

    metrics = _evaluate(model, x_test, y_test, x_test_poison, config.TARGET_LABEL)
    metrics.update({"n_train": n, "n_test": len(x_test), "amp": amp})
    _log(f"final {attack}: BA={metrics['ba']:.4f} ASR={metrics['asr']}")
    _write_final_artifact(attack, model, metrics, history, args, resume_count)
    return metrics


def _write_final_artifact(
    attack: str,
    model: torch.nn.Module,
    metrics: dict,
    history: list[dict],
    args: argparse.Namespace,
    resume_count: int,
) -> None:
    out_dir = Path(args.output_dir)
    checkpoint = out_dir / "checkpoints" / config.CHECKPOINTS[attack]
    meta = {
        "attack": attack,
        "metrics": metrics,
        "epochs": args.epochs,
        "seed": args.seed,
        "profile": config.PROFILE,
        "model_name": "resnet18",
        "poison_rate": config.POISON_RATE if attack != "clean" else 0.0,
        "target": config.TARGET_LABEL,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "milestones": list(args.milestones),
        "amp": metrics.get("amp"),
        "resume_count": resume_count,
        "train_size": metrics.get("n_train"),
        "test_size": metrics.get("n_test"),
        "paper_recipe": "SGD lr=0.01, milestones 50/100/150, batch 64, 200 epochs",
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tmp_checkpoint = checkpoint.with_name(checkpoint.name + ".tmp")
    sha = models.save_checkpoint(tmp_checkpoint, model, meta, history)
    os.replace(tmp_checkpoint, checkpoint)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{attack}.json").write_text(
        json.dumps({"metrics": metrics, "meta": meta, "history": history, "sha256": sha},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _log(f"final artifact {attack}: {checkpoint} sha256={sha[:12]}...")


def _write_manifest(args: argparse.Namespace, data_meta: dict) -> None:
    out_dir = Path(args.output_dir)
    metrics = {}
    for attack in ("clean", *config.ATTACKS):
        path = out_dir / "metrics" / f"{attack}.json"
        if not path.exists():
            _log(f"manifest skipped: missing metrics for {attack}")
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics[attack] = payload["metrics"]
    extra = {
        "profile": config.PROFILE,
        "model_name": "resnet18",
        "data": data_meta,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "milestones": list(args.milestones),
        "epochs": args.epochs,
        "amp": {name: rec.get("amp") for name, rec in metrics.items()},
        "recipe": "SGD lr=0.01, milestones 50/100/150, batch 64, 200 epochs",
        "ood_gtsrb": bool(args.ood_gtsrb),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    artifacts.write_manifest(out_dir, metrics, extra=extra)
    _log(f"manifest written: {out_dir / 'manifest.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attack", choices=("clean", *config.ATTACKS, "all"), default="all")
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--output-dir", default=str(config.ARTIFACTS_DIR))
    p.add_argument("--work-dir", default=str(config.WORK_DIR))
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--momentum", type=float, default=config.MOMENTUM)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--milestones", type=int, nargs="+", default=list(config.LR_MILESTONES))
    p.add_argument("--gamma", type=float, default=config.LR_GAMMA)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", default="auto", help="auto | none | path/to/last.pt")
    p.add_argument("--checkpoint-every-steps", type=int, default=0,
                   help="also checkpoint every N optimizer steps (0 = epoch only)")
    p.add_argument("--eval-every", type=int, default=5,
                   help="evaluate on the test split every N epochs (0 = only at the end)")
    p.add_argument("--limit-train", type=int, default=None, help="smoke tests only")
    p.add_argument("--limit-test", type=int, default=None, help="smoke tests only")
    p.add_argument("--data-source", default=os.environ.get("EBBA_CIFAR_SOURCE", "fastai"),
                   choices=["fastai", "torchvision"])
    p.add_argument("--ood-gtsrb", action=argparse.BooleanOptionalAction, default=config.OOD_GTSRB)
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--force", action="store_true", help="retrain even if final artifacts exist")
    p.add_argument("--no-manifest", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--threads", type=int, default=config.NUM_THREADS)
    return p


def main() -> int:
    global _LOG_PATH
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = work_dir / "train.log"

    determinism.set_global_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
    device = determinism.get_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit(
            "CUDA is not available. Full ResNet-18/200-epoch training on CPU is "
            "not practical; pass --allow-cpu only for a deliberate smoke test."
        )
    _log(
        f"profile={config.PROFILE} torch={torch.__version__} device={device} "
        f"gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} "
        f"threads={args.threads}"
    )

    data_meta = _ensure_data(data_dir, args)
    _log(f"data meta: {data_meta}")

    if args.finalize_only:
        _write_manifest(args, data_meta)
        return 0

    # Hash the split files once.  Resume validation then catches a rebuilt or
    # swapped dataset instead of silently continuing on different images.
    data_sha = {
        name: _sha256(data_dir / name)
        for name in ("cifar10_train.npz", "cifar10_test.npz", "cifar100_ood.npz")
    }

    attacks_to_run = list(("clean", *config.ATTACKS)) if args.attack == "all" else [args.attack]
    all_metrics = {}
    for attack in attacks_to_run:
        final_artifact = Path(args.output_dir) / "checkpoints" / config.CHECKPOINTS[attack]
        metric_path = Path(args.output_dir) / "metrics" / f"{attack}.json"
        if final_artifact.exists() and metric_path.exists() and not args.force:
            _log(f"skip {attack}: final artifact already exists ({final_artifact})")
            all_metrics[attack] = json.loads(
                metric_path.read_text(encoding="utf-8")
            )["metrics"]
            continue
        result = _train_attack(attack, args, data_dir, data_sha, device)
        all_metrics[attack] = result
        if result.get("stopped"):
            # A signal stops the whole run, not just the current attack.
            # Rerunning with --resume auto continues at the saved batch.
            _log("stop requested; not starting further attacks")
            break

    if not args.no_manifest and args.attack == "all":
        _write_manifest(args, data_meta)
    _log(f"done: {json.dumps(all_metrics, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    raise SystemExit(main())
