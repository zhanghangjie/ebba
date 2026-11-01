"""EBBA energy statistics (paper Eq. 3-8).

Important implementation note
-----------------------------
Eq. (3) does **not** subtract the maximum logit from every entry; it sets the
maximum entry itself to zero and leaves the rest unchanged.  The paper's
Appendix-B example confirms this: ``[3, 8, -2] -> [3, 0, -2]``.  We implement
the literal operation.

Three normalisations are provided because the paper, the official README and
the official notebook disagree:
  * ``paper``    : NP = exp(s*T)/T - exp(0)/T        (Eq. 5, T=2)
  * ``readme``   : NP = exp(s*T)/T - 1               (README: P_k - 1)
  * ``official`` : NP = min(exp(s*T)/T, 2) - 0.6     (reference notebook)

The notebooks report how sensitive the detection verdict is to this choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)       # numerical stability only
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def clamp_max_logits(logits: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Eq. (3): set each sample's maximum logit to 0, keep the others."""
    z = np.asarray(logits, dtype=np.float64)
    if not inplace:
        z = z.copy()
    rows = np.arange(z.shape[0])
    z[rows, z.argmax(axis=1)] = 0.0
    return z


def per_sample_normalized_energy(
    logits: np.ndarray,
    temperature: float = config.TEMPERATURE,
    variant: str = "paper",
) -> np.ndarray:
    """Return an (N, K) matrix of per-sample NP values (Eq. 4-5)."""
    s = softmax(clamp_max_logits(logits))
    p = np.exp(s * temperature) / temperature
    if variant == "paper":
        return p - np.exp(0.0 * temperature) / temperature
    if variant == "readme":
        return p - 1.0
    if variant == "official":
        return np.minimum(p, 2.0) - 0.6
    raise ValueError(f"unknown variant: {variant}")


def label_energy(
    logits: np.ndarray,
    temperature: float = config.TEMPERATURE,
    variant: str = "paper",
) -> np.ndarray:
    """Eq. (6): sum the per-sample energies for each label."""
    return per_sample_normalized_energy(logits, temperature, variant).sum(axis=0)


@dataclass
class EnergyStats:
    energy: np.ndarray
    mean: float
    std: float
    deviation: np.ndarray

    @property
    def zscore(self) -> np.ndarray:
        return self.deviation / self.std if self.std > 0 else np.zeros_like(self.deviation)


def energy_stats(energy: np.ndarray) -> EnergyStats:
    e = np.asarray(energy, dtype=np.float64)
    mu = float(e.mean())
    sigma = float(e.std(ddof=0))               # Eq. (7) divides by K
    return EnergyStats(energy=e, mean=mu, std=sigma, deviation=e - mu)


def detect(
    energy: np.ndarray,
    lam: float = config.LAMBDA,
    two_sided: bool = False,
) -> dict:
    """Eq. (8).  Returns the detected target label and the margin.

    One-sided test (paper): backdoored iff max_k (E_k - mu) > lam * sigma.
    Two-sided test (Appendix G.4 fix): backdoored iff max_k |E_k - mu| > lam * sigma.
    """
    st = energy_stats(energy)
    if two_sided:
        scores = np.abs(st.deviation)
        target = int(np.argmax(scores))
        margin = float(scores[target] - lam * st.std)
        direction = "low" if st.deviation[target] < 0 else "high"
    else:
        scores = st.deviation
        target = int(np.argmax(scores))
        margin = float(scores[target] - lam * st.std)
        direction = "high"
    return {
        "is_backdoored": bool(margin > 0),
        "target": target,
        "margin": margin,
        "direction": direction,
        "zscore": float(scores[target] / st.std) if st.std > 0 else 0.0,
        "mean": st.mean,
        "std": st.std,
        "lambda": lam,
        "two_sided": two_sided,
    }


def sweep_lambda(
    energy_backdoored: np.ndarray,
    energy_clean: np.ndarray,
    lambdas: np.ndarray | None = None,
    chosen: float | None = None,
) -> dict:
    """Reproduce the shape of Fig. 80: E_t - mu - lambda*sigma for both models."""
    chosen = config.LAMBDA if chosen is None else float(chosen)
    lambdas = np.linspace(0.0, 8.0, 81) if lambdas is None else np.asarray(lambdas)
    bd = energy_stats(energy_backdoored)
    cl = energy_stats(energy_clean)
    bd_target = int(np.argmax(bd.deviation))
    bd_curve = bd.deviation[bd_target] - lambdas * bd.std
    cl_curve = cl.deviation.max() - lambdas * cl.std
    # Valid lambda: backdoored margin > 0 and clean margin < 0.
    valid = (bd_curve > 0) & (cl_curve < 0)
    interval = (float(lambdas[valid].min()), float(lambdas[valid].max())) if valid.any() else None
    return {
        "lambdas": lambdas,
        "backdoored_curve": bd_curve,
        "clean_curve": cl_curve,
        "valid_interval": interval,
        "chosen_lambda": chosen,
        "chosen_valid": bool(valid[np.argmin(np.abs(lambdas - chosen))]) if valid.any() else False,
        "backdoored_target": bd_target,
    }


def variant_targets(logits: np.ndarray) -> dict[str, int]:
    """Detected target label under each normalisation variant."""
    return {
        "paper": detect(label_energy(logits, variant="paper"))["target"],
        "readme": detect(label_energy(logits, variant="readme"))["target"],
        "official": detect(label_energy(logits, variant="official"))["target"],
    }
