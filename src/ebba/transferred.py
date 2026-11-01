"""Transferred energy / normalized transferred energy (paper Eq. 9-12).

TE^k_j  = P'_k / P_k = exp(s'_k - s_k)          (Eq. 9)
NTE^k_j = exp(s'_k - s_k) - 1                  (Eq. 10)
NTE_k   = sum_j NTE^k_j                        (Eq. 11)
k1      = argmax_k NTE_k                        (Eq. 12)

Formally Eq. 4 carries a temperature T while Eq. 9/10 do not.  We expose the
temperature so the audit can quantify the discrepancy; the main experiments
use T=1, which is what the official code does.
"""

from __future__ import annotations

import numpy as np

from .energy import softmax


def nte(
    source_logits: np.ndarray,
    corrupt_logits: np.ndarray,
    temperature: float = 1.0,
    cap: float | None = None,
) -> np.ndarray:
    """Aggregate NTE over corruptions.

    Parameters
    ----------
    source_logits : (N, K)
    corrupt_logits : (J, N, K) -- logits of J corruptions for the same N images
    temperature : float
        Set to 1 to match Eq. 9/10 and the official implementation.
    cap : float or None
        The official notebook caps single-corruption NTE at 5 (and 4 in one
        helper).  ``None`` implements the paper literally.
    """
    s = softmax(source_logits)
    s_c = softmax(corrupt_logits.reshape(-1, corrupt_logits.shape[-1])).reshape(corrupt_logits.shape)
    delta = temperature * (s_c - s[None, :, :])
    values = np.exp(np.clip(delta, -50, 50)) - 1.0
    if cap is not None:
        values = np.minimum(values, cap)
    return values.sum(axis=0)


def recover_labels(nte_scores: np.ndarray) -> np.ndarray:
    """Eq. (12): the original label is the argmax of the NTE vector."""
    return np.asarray(nte_scores).argmax(axis=1)


def max_nte(nte_scores: np.ndarray) -> np.ndarray:
    """Peak transferred energy: a compact feature for poisoned/clean separation."""
    return np.asarray(nte_scores).max(axis=1)


def nte_entropy(nte_scores: np.ndarray) -> np.ndarray:
    """Entropy of the positive part of the NTE distribution."""
    x = np.clip(np.asarray(nte_scores, dtype=np.float64), 0, None)
    x = x / np.maximum(x.sum(axis=1, keepdims=True), 1e-12)
    return -(x * np.log(np.maximum(x, 1e-12))).sum(axis=1)


def features_for_clustering(nte_scores: np.ndarray) -> np.ndarray:
    """Features used by the clustering step (paper's t-SNE input analogue).

    The paper projects corrupted images into a feature space with t-SNE.  We
    use the K-dimensional NTE vector itself followed by t-SNE; this keeps the
    decision tied to the quantity defined in Eq. 11.
    """
    return np.asarray(nte_scores, dtype=np.float64)


def mean_difference_check(source_logits: np.ndarray, corrupt_logits: np.ndarray) -> dict:
    """Verify the paper's claim that mean_m(s'_m) - mean_m(s_m) = 0."""
    s = softmax(source_logits)
    sc = softmax(corrupt_logits.reshape(-1, corrupt_logits.shape[-1])).reshape(corrupt_logits.shape)
    per_corruption = (sc - s[None]).mean(axis=2)          # (J, N)
    return {
        "max_abs_mean_difference": float(np.abs(per_corruption).max()),
        "mean_abs_mean_difference": float(np.abs(per_corruption).mean()),
        "softmax_sums": float(np.abs(s.sum(axis=1) - 1).max()),
    }


def temperature_sensitivity(
    source_logits: np.ndarray,
    corrupt_logits: np.ndarray,
    temperatures: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
    cap: float | None = 5.0,
) -> dict:
    """Argmax agreement of NTE recovery across temperatures."""
    base = recover_labels(nte(source_logits, corrupt_logits, temperature=1.0, cap=cap))
    out = {}
    for t in temperatures:
        pred = recover_labels(nte(source_logits, corrupt_logits, temperature=t, cap=cap))
        out[str(t)] = float((pred == base).mean())
    return out
