"""Clustering step for EBBA+ (paper Sec. 4.2 / Figs. 4-5).

The paper projects corrupted images with t-SNE and separates
"clean images of the target class" from "poisoned images of other classes".
We use the NTE vector as the feature, reduce it with t-SNE, then run one of
four off-the-shelf clusterers.  The paper does not say how the poisoned
cluster is chosen without labels, so that step is an assumption of this
harness: the cluster with the larger mean target-vs-runner-up logit margin
is called poisoned.  The only other label-free signal available in the
pipeline is the NTE peak, which our B06 diagnostics show is anti-correlated
with poisoning; B07 reports both orientations so the reader can re-adjudicate.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, KMeans, MeanShift
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from . import config


def tsne_embed(features: np.ndarray, seed: int = config.SEED) -> np.ndarray:
    n = len(features)
    if n < 8:
        return features[:, :2] if features.shape[1] >= 2 else np.pad(features, ((0, 0), (0, 2 - features.shape[1])))
    perplexity = float(min(30.0, max(2.0, (n - 1) / 3.0)))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=750,
        random_state=seed,
        # The candidate pool is small (a few hundred points), so the exact
        # solver is cheap.  Barnes-Hut's tree approximation differs across
        # sklearn/BLAS builds and moved the proxy PDR by >0.10 between the
        # local environment and the container; the exact solver does not.
        method="exact",
    )
    return tsne.fit_transform(StandardScaler().fit_transform(features))


def cluster_labels(embedding: np.ndarray, method: str) -> tuple[np.ndarray, str]:
    """Return cluster ids and the method actually used (``fallback`` if needed)."""
    method = method.lower()
    if method == "birch":
        labels = Birch(n_clusters=2, threshold=config.BIRCH_THRESHOLD).fit_predict(embedding)
    elif method in ("hc", "hierarchical"):
        labels = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(embedding)
    elif method in ("meanshift", "mean_shift"):
        try:
            labels = MeanShift(bin_seeding=True).fit_predict(embedding)
        except Exception:
            labels = np.zeros(len(embedding), dtype=int)
    elif method == "dbscan":
        labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(embedding)
    else:
        raise ValueError(f"unknown clustering method: {method}")

    valid = labels[labels >= 0]
    if len(np.unique(valid)) != 2:
        # Degenerate output: keep the comparison meaningful with a 2-means
        # fallback and record that the method needed it.
        kmeans = KMeans(n_clusters=2, random_state=config.SEED, n_init=10)
        return kmeans.fit_predict(embedding), "fallback_kmeans"
    return labels, method


def identify_poisoned_cluster(
    labels: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, int | None, dict]:
    """Turn cluster ids into a binary poisoned/clean prediction.

    The cluster whose members have the larger mean *identification score* is
    called poisoned.  The score is supplied by the caller; see
    :func:`run_pipeline`.  DBSCAN noise (-1) is treated as clean.
    """
    labels = np.asarray(labels)
    confidence = np.asarray(confidence, dtype=np.float64)
    stats: dict[str, float] = {}
    best_cluster: int | None = None
    best_score = -np.inf
    for c in np.unique(labels):
        if c < 0:
            continue
        mask = labels == c
        stats[str(int(c))] = float(confidence[mask].mean())
        if stats[str(int(c))] > best_score:
            best_score = stats[str(int(c))]
            best_cluster = int(c)
    if best_cluster is None:
        return np.zeros(len(labels), dtype=bool), None, stats
    return labels == best_cluster, best_cluster, stats


def run_pipeline(
    nte_scores: np.ndarray,
    method: str = "birch",
    use_tsne: bool = True,
    identification_scores: np.ndarray | None = None,
) -> dict:
    """Full EBBA+ clustering pipeline on a candidate pool.

    The paper (Sec. 4.2) says the poisoned and clean-target clusters are
    separated by t-SNE + Birch but does not state how the poisoned cluster is
    chosen without labels.  That choice is an assumption of this harness.
    ``identification_scores`` supplies the label-free statistic: the cluster
    with the larger mean score is called poisoned.  We use the model's
    target-vs-runner-up logit margin, which is available in the pipeline and
    is the signal a backdoor actually creates.  When it is omitted (legacy
    callers) the mean peak NTE is used, which the B06 diagnostics show is
    anti-correlated with poisoning on the proxy models.
    """
    features = np.asarray(nte_scores, dtype=np.float64)
    if use_tsne:
        embedding = tsne_embed(features)
    else:
        # Cluster in the full NTE space.  This is the stable path: t-SNE is a
        # discontinuous function of its input, so the last-bit differences
        # between torch builds moved the proxy PDR by >0.10, while the NTE
        # values themselves differ only in low bits.
        embedding = StandardScaler().fit_transform(features)
    labels, used = cluster_labels(embedding, method)
    if identification_scores is None:
        confidence = np.clip(features, 0, None).max(axis=1)
        rule = "peak_nte"
    else:
        confidence = np.asarray(identification_scores, dtype=np.float64)
        rule = "target_margin"
    pred, cluster, stats = identify_poisoned_cluster(labels, confidence)
    return {
        "pred_poisoned": pred,
        "cluster_labels": labels,
        "embedding": embedding,
        "poisoned_cluster": cluster,
        "cluster_confidence": stats,
        "method_used": used,
        "identification_rule": rule,
        "features": features,
    }
