"""Build the proxy dataset, train three checkpoints and enforce build gates.

Run inside the image at build time (offline afterwards):

    python scripts/build_artifacts.py --device cpu

The gates are deliberately strict: if the proxy model does not show the
energy anomaly, the verification would be meaningless, so the build fails
instead of producing a misleading report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebba import artifacts, attacks, config, data, determinism, models  # noqa: E402
from ebba.energy import detect, label_energy  # noqa: E402
from ebba.transferred import nte, recover_labels  # noqa: E402


def _evaluate(model, x_test, y_test, attack, device):
    poisoned = attacks.apply_trigger(x_test, attack)
    logits_clean = models.predict_logits(model, x_test, device)
    logits_poison = models.predict_logits(model, poisoned, device)
    return {
        "ba": float((logits_clean.argmax(1) == y_test).mean()),
        "asr": float((logits_poison.argmax(1) == config.TARGET_LABEL).mean()),
    }


def _ood_energy(model, x_ood, device):
    logits = models.predict_logits(model, x_ood, device)
    idx, counts = data.balanced_select_by_pseudo(logits, config.OOD_MIN_PER_CLASS)
    energy = label_energy(logits[idx], config.TEMPERATURE, "paper")
    pred_hist = np.bincount(logits.argmax(axis=1), minlength=config.NUM_CLASSES)
    print(f"[ood] predicted-label histogram: {pred_hist.tolist()}")
    print(f"[ood] selected per label: {counts}")
    print(f"[ood] energy: {np.round(energy, 2).tolist()} top3={np.argsort(-energy)[:3].tolist()}")
    det = detect(energy, config.PROXY_LAMBDA)
    det["paper_margin"] = float(det["margin"] - (config.LAMBDA - config.PROXY_LAMBDA) * det["std"])
    return energy, det, counts


def _nte_recovery(model, x_test, y_test, attack, device, n=100):
    poisoned = attacks.apply_trigger(x_test, attack)
    candidate = data.make_candidate_set(
        x_test, y_test, poisoned, config.TARGET_LABEL,
        n_poisoned=n, n_clean_target=20,
    )
    src = models.predict_logits(model, candidate["images"], device)
    # EBBA+ only processes images the model assigns to the target label.
    keep = src.argmax(1) == config.TARGET_LABEL
    src = src[keep]
    images = candidate["images"][keep]
    ground_truth = candidate["ground_truth"][keep]
    mask = candidate["is_poisoned"][keep]
    from ebba.corruptions import corrupt_all

    corrupted = corrupt_all(images)
    cor = models.predict_logits(model, corrupted.reshape(-1, *images.shape[1:]), device)
    cor = cor.reshape(len(corrupted), len(images), config.NUM_CLASSES)
    scores = nte(src, cor, temperature=1.0, cap=5.0)
    pred = recover_labels(scores)
    return float((pred[mask] == ground_truth[mask]).mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--data-dir", default=str(config.DATA_DIR))
    parser.add_argument(
        "--data-source",
        default=os.environ.get("EBBA_CIFAR_SOURCE", "fastai"),
        choices=["fastai", "torchvision"],
        help="fastai: S3 PNG mirror (default, usually much faster); "
        "torchvision: canonical cs.toronto.edu pickles",
    )
    parser.add_argument("--checkpoint-dir", default=str(config.CHECKPOINT_DIR))
    parser.add_argument("--no-gates", action="store_true")
    parser.add_argument("--robust", action=argparse.BooleanOptionalAction,
                        default=config.ROBUST_AUG,
                        help="light corruption augmentation during training")
    args = parser.parse_args()

    determinism.set_global_seed(config.SEED)
    determinism.configure_torch(config.NUM_THREADS)
    device = determinism.get_device(args.device)
    print(f"[env] device={device} torch={__import__('torch').__version__} threads={config.NUM_THREADS}")

    data.build_subsets(Path(args.data_dir), source=args.data_source)
    x_test, y_test = data.load_test_data(Path(args.data_dir))
    x_ood = data.load_ood_data(Path(args.data_dir))

    metrics: dict[str, dict] = {}
    gates: dict[str, bool] = {}
    margin: dict[str, dict] = {}
    for attack in ("clean", *config.ATTACKS):
        print(f"[train] {attack}")
        training = data.load_training_data(attack, Path(args.data_dir))
        model, history = models.train_model(
            training["x"], training["y"], epochs=args.epochs, device=device,
            verbose=True, robust=args.robust,
        )
        m = _evaluate(model, x_test, y_test, "badnets" if attack == "clean" else attack, device)
        energy, det, counts = _ood_energy(model, x_ood, device)
        margin[attack] = {
            "target": det["target"], "margin": det["margin"], "zscore": det["zscore"],
            "is_backdoored": det["is_backdoored"], "counts": counts,
        }
        checkpoint = Path(args.checkpoint_dir) / config.CHECKPOINTS[attack]
        meta = {
            "attack": attack,
            "metrics": m,
            "epochs": args.epochs,
            "seed": config.SEED,
            "poison_rate": config.POISON_RATE if attack != "clean" else 0.0,
            "target": config.TARGET_LABEL,
            "augment": True,
            "robust_augment": bool(args.robust),
        }
        models.save_checkpoint(checkpoint, model, meta, history)
        metrics[attack] = m
        print(f"[metrics] {attack} BA={m['ba']:.4f} ASR={m['asr']:.4f} "
              f"detect_margin={det['margin']:.3f} target={det['target']}")

    recovery = {}
    for attack in config.ATTACKS:
        model, _ = models.load_checkpoint(
            Path(args.checkpoint_dir) / config.CHECKPOINTS[attack], map_location="cpu"
        )
        recovery[attack] = _nte_recovery(model, x_test, y_test, attack, device)
        print(f"[gate] NTE recovery {attack}={recovery[attack]:.4f}")

    if not args.no_gates:
        # Gates certify the *testbed*, not the paper's claims: the proxy model
        # must be competent, both attacks must succeed, and the clean control
        # must stay below the proxy threshold.  NTE recovery is a measured
        # outcome (B05), so it is recorded but deliberately not gated.
        gates["clean_ba_min"] = metrics["clean"]["ba"] >= 0.70
        for attack in config.ATTACKS:
            gates[f"{attack}_ba_min"] = metrics[attack]["ba"] >= 0.68
        gates["badnets_asr"] = metrics["badnets"]["asr"] >= 0.90
        gates["blend_asr"] = metrics["blend"]["asr"] >= 0.90
        gates["clean_not_detected"] = (
            margin["clean"]["margin"] < 0 and not margin["clean"]["is_backdoored"]
        )
        # EBBA+ needs at least one attacked model whose target energy anomaly
        # yields a target pool; BadNets is that model.  Blend's detection is a
        # result (B02), not a prerequisite.
        gates["badnets_detected_target"] = (
            margin["badnets"]["margin"] > 0
            and margin["badnets"]["is_backdoored"]
            and margin["badnets"]["target"] == config.TARGET_LABEL
        )
        failed = [k for k, v in gates.items() if not v]
        print("[gates]", json.dumps(gates, indent=2))
        if failed:
            print("[gates] FAILED:", ", ".join(failed), file=sys.stderr)
            return 1

    artifacts.write_manifest(
        Path(args.checkpoint_dir).parent,
        metrics,
        extra={
            "detection": margin,
            "nte_recovery": recovery,
            "gates": gates,
            "epochs": args.epochs,
            "device": str(device),
            "cifar_source": args.data_source,
        },
    )
    print("[done] artifacts written to", Path(args.checkpoint_dir).parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
