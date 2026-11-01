"""Diagnostic: which feature space separates poisoned from clean-target images?

The paper says EBBA+ projects the *perturbed images* with t-SNE before
clustering, but does not specify the representation.  This script compares
three operationalisations on the same candidate pool:

  pixel    : average corrupted image, downsampled to 8x8x3
  embed    : average ResNet-18 penultimate feature over the corruptions
  nte      : the K-dimensional NTE vector (current full_eval behaviour)

It prints oracle PDR (best of the two Birch clusters) and the PDR obtained
with the label-free target-margin rule.  It only reads artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ebba import attacks, artifacts, clustering, config, corruptions, data, metrics, models  # noqa: E402
from ebba.transferred import nte, recover_labels  # noqa: E402


def _predict(model, x, device, batch=256):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = models.to_tensor(x[i:i + batch]).to(device)
            outs.append(model(xb).float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _embed(model, x, device, batch=256):
    captured = []
    handle = model.avgpool.register_forward_hook(
        lambda _m, _i, out: captured.append(out.flatten(1).detach())
    )
    model.eval()
    outs = []
    try:
        with torch.no_grad():
            for i in range(0, len(x), batch):
                xb = models.to_tensor(x[i:i + batch]).to(device)
                captured.clear()
                model(xb)
                outs.append(captured[0].float().cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(outs, axis=0)


def _cluster(features, margin, peak, is_poisoned, use_tsne=True):
    if use_tsne:
        emb = clustering.tsne_embed(features)
    else:
        from sklearn.preprocessing import StandardScaler
        emb = StandardScaler().fit_transform(features)
    labels, used = clustering.cluster_labels(emb, "birch")
    best = None
    for c in np.unique(labels):
        if c < 0:
            continue
        m = metrics.binary_metrics(labels == c, is_poisoned)
        if best is None or m["pdr"] > best["pdr"]:
            best = m
            best_c = int(c)
    sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    rules = {}
    for name, score in (("margin", margin), ("peak", peak), ("size", np.ones(len(labels)))):
        stats = {int(c): float(score[labels == c].mean()) for c in np.unique(labels) if c >= 0}
        if name == "size":
            stats = {int(c): int((labels == c).sum()) for c in np.unique(labels) if c >= 0}
        for direction in ("high", "low"):
            c = max(stats, key=stats.get) if direction == "high" else min(stats, key=stats.get)
            m = metrics.binary_metrics(labels == c, is_poisoned)
            rules[f"{name}_{direction}"] = {
                "pdr": m["pdr"], "f1": m["f1"], "cluster": c, "stats": stats,
            }
    return {
        "method": used,
        "sizes": sizes,
        "oracle_pdr": best["pdr"],
        "oracle_f1": best["f1"],
        "oracle_cluster": best_c,
        "oracle_precision": best["precision"],
        "rules": rules,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--attack", default="badnets", choices=["badnets", "blend"])
    p.add_argument("--n-poisoned", type=int, default=300)
    p.add_argument("--n-clean", type=int, default=150)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    bundle = artifacts.load_data()
    x_test, y_test = bundle["x_test"], bundle["y_test"]
    ckpt = config.ARTIFACTS_DIR / "checkpoints" / config.CHECKPOINTS[args.attack]
    model, meta = models.load_checkpoint(ckpt, map_location="cpu")
    model = model.to(device)
    print(f"loaded {ckpt} ({meta.get('model_name')})")

    poisoned_test = attacks.apply_trigger(x_test, args.attack)
    cand = data.make_candidate_set(
        x_test, y_test, poisoned_test, config.TARGET_LABEL,
        n_poisoned=args.n_poisoned, n_clean_target=args.n_clean,
    )
    src = _predict(model, cand["images"], device)
    keep = src.argmax(1) == config.TARGET_LABEL
    x = cand["images"][keep]
    is_p = cand["is_poisoned"][keep]
    src = src[keep]
    others = np.delete(src, config.TARGET_LABEL, axis=1)
    margin = src[:, config.TARGET_LABEL] - others.max(1)
    print(f"candidates={len(cand['images'])} kept={len(x)} poisoned={is_p.sum()} clean={(~is_p).sum()}")

    corrupt = corruptions.corrupt_all(x)
    print(f"corruptions {corrupt.shape}")
    flat = corrupt.reshape(-1, *corrupt.shape[2:])
    cl = _predict(model, flat, device).reshape(len(corrupt), len(x), -1)
    scores = nte(src, cl, temperature=1.0, cap=5.0)
    peak = np.clip(scores, 0, None).max(axis=1)
    print("nte recovery:", float((recover_labels(scores)[is_p] == cand['ground_truth'][keep][is_p]).mean()))

    # Pixel feature: mean corrupted image, 8x8 block mean.
    pix = corrupt.astype(np.float32)
    h, w = pix.shape[2], pix.shape[3]
    pix = pix.reshape(len(pix), len(x), h // 8, 8, w // 8, 8, 3).mean(axis=(3, 5))
    pix = pix.mean(axis=0).reshape(len(x), -1)

    # Embedding feature: average penultimate vector over corruptions.
    emb = np.stack([_embed(model, c, device) for c in corrupt], axis=0).mean(axis=0)
    print(f"pixel {pix.shape} embed {emb.shape} nte {scores.shape}")

    for name, feats in (("nte", scores), ("pixel", pix), ("embed", emb)):
        for use_tsne in (True, False):
            res = _cluster(feats, margin, peak, is_p, use_tsne=use_tsne)
            print(name, "tsne" if use_tsne else "raw", res)


if __name__ == "__main__":
    raise SystemExit(main())
