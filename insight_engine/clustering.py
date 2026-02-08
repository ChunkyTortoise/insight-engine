"""Unsupervised clustering: K-means, DBSCAN, silhouette scores, cluster comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterResult:
    """Result of a clustering operation."""

    method: str
    n_clusters: int
    labels: list[int]
    silhouette: float  # -1 to 1, higher is better
    cluster_sizes: dict[int, int]
    inertia: float | None = None  # K-means only


@dataclass
class ClusterComparison:
    """Comparison of clustering methods/configs."""

    results: dict[str, ClusterResult]
    best_method: str
    best_silhouette: float


@dataclass
class ClusterEvaluation:
    """Cluster quality metrics."""

    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float


@dataclass
class ElbowResult:
    """Result of elbow method analysis."""

    k_values: list[int]
    inertias: list[float]
    suggested_k: int


def _count_clusters(labels: np.ndarray) -> int:
    """Count distinct clusters, excluding noise label -1."""
    unique = set(labels)
    unique.discard(-1)
    return len(unique)


def _cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    """Count points per cluster label."""
    sizes: dict[int, int] = {}
    for label in labels:
        lbl = int(label)
        sizes[lbl] = sizes.get(lbl, 0) + 1
    return dict(sorted(sizes.items()))


def _safe_silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score, returning -1.0 if only one cluster or all noise."""
    n_clusters = _count_clusters(labels)
    if n_clusters < 2:
        return -1.0
    # Need at least 2 non-noise labels with >1 sample
    non_noise_mask = labels != -1
    if non_noise_mask.sum() < 2:
        return -1.0
    try:
        return float(silhouette_score(data[non_noise_mask], labels[non_noise_mask]))
    except ValueError:
        return -1.0


class Clusterer:
    """Cluster numeric data using K-means and DBSCAN."""

    def kmeans(self, data: np.ndarray, k: int = 3, random_state: int = 42) -> ClusterResult:
        """K-means clustering with silhouette scoring."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(scaled)

        sil = _safe_silhouette(scaled, labels)

        return ClusterResult(
            method="kmeans",
            n_clusters=k,
            labels=labels.tolist(),
            silhouette=round(sil, 6),
            cluster_sizes=_cluster_sizes(labels),
            inertia=round(float(model.inertia_), 6),
        )

    def auto_kmeans(
        self,
        data: np.ndarray,
        k_range: range = range(2, 11),
        random_state: int = 42,
    ) -> ClusterResult:
        """Auto-select best K using silhouette score (elbow-like)."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        best_result: ClusterResult | None = None
        best_sil = -2.0

        for k in k_range:
            if k >= len(data):
                break
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = model.fit_predict(scaled)
            sil = _safe_silhouette(scaled, labels)

            if sil > best_sil:
                best_sil = sil
                best_result = ClusterResult(
                    method="auto_kmeans",
                    n_clusters=k,
                    labels=labels.tolist(),
                    silhouette=round(sil, 6),
                    cluster_sizes=_cluster_sizes(labels),
                    inertia=round(float(model.inertia_), 6),
                )

        if best_result is None:
            # Fallback: single k=2
            return self.kmeans(data, k=2, random_state=random_state)

        return best_result

    def dbscan(self, data: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> ClusterResult:
        """DBSCAN clustering. Noise points get label -1."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(scaled)

        n_clusters = _count_clusters(labels)
        sil = _safe_silhouette(scaled, labels)

        return ClusterResult(
            method="dbscan",
            n_clusters=n_clusters,
            labels=labels.tolist(),
            silhouette=round(sil, 6),
            cluster_sizes=_cluster_sizes(labels),
            inertia=None,
        )

    def compare(self, data: np.ndarray) -> ClusterComparison:
        """Compare auto_kmeans and dbscan, return best by silhouette."""
        results: dict[str, ClusterResult] = {}
        results["auto_kmeans"] = self.auto_kmeans(data)
        results["dbscan"] = self.dbscan(data)

        best_method = max(results, key=lambda k: results[k].silhouette)
        best_sil = results[best_method].silhouette

        return ClusterComparison(
            results=results,
            best_method=best_method,
            best_silhouette=best_sil,
        )

    def hierarchical(
        self,
        data: np.ndarray,
        n_clusters: int = 3,
        linkage: str = "ward",
    ) -> ClusterResult:
        """Hierarchical clustering using agglomerative approach.

        Supports linkage methods: ward, complete, average, single.
        """
        from sklearn.cluster import AgglomerativeClustering

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(scaled)

        sil = _safe_silhouette(scaled, labels)

        return ClusterResult(
            method=f"hierarchical_{linkage}",
            n_clusters=n_clusters,
            labels=labels.tolist(),
            silhouette=round(sil, 6),
            cluster_sizes=_cluster_sizes(labels),
            inertia=None,
        )

    def evaluate_clusters(self, data: np.ndarray, labels: list[int]) -> ClusterEvaluation:
        """Evaluate cluster quality using multiple metrics.

        Returns silhouette score, Calinski-Harabasz index, and Davies-Bouldin index.
        """
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

        labels_arr = np.array(labels)
        n_clusters = _count_clusters(labels_arr)

        # Silhouette score
        sil = _safe_silhouette(data, labels_arr)

        # Calinski-Harabasz (higher is better)
        if n_clusters >= 2 and len(data) > n_clusters:
            try:
                ch_score = float(calinski_harabasz_score(data, labels_arr))
            except ValueError:
                ch_score = 0.0
        else:
            ch_score = 0.0

        # Davies-Bouldin (lower is better)
        if n_clusters >= 2:
            try:
                db_score = float(davies_bouldin_score(data, labels_arr))
            except ValueError:
                db_score = 0.0
        else:
            db_score = 0.0

        return ClusterEvaluation(
            silhouette=round(sil, 6),
            calinski_harabasz=round(ch_score, 6),
            davies_bouldin=round(db_score, 6),
        )

    def profile_clusters(
        self,
        data: np.ndarray,
        labels: list[int],
        feature_names: list[str] | None = None,
    ) -> dict[int, dict[str, dict[str, float]]]:
        """Profile clusters by computing per-cluster statistics for each feature.

        Returns dict mapping cluster_id -> feature_name -> {mean, std, size}.
        """
        labels_arr = np.array(labels)
        unique_labels = sorted(set(labels))

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]

        profiles: dict[int, dict[str, dict[str, float]]] = {}

        for label in unique_labels:
            mask = labels_arr == label
            cluster_data = data[mask]
            cluster_size = int(mask.sum())

            feature_stats: dict[str, dict[str, float]] = {}
            for i, fname in enumerate(feature_names):
                feature_vals = cluster_data[:, i]
                feature_stats[fname] = {
                    "mean": round(float(feature_vals.mean()), 6),
                    "std": round(float(feature_vals.std()), 6),
                    "size": float(cluster_size),
                }

            profiles[int(label)] = feature_stats

        return profiles

    def elbow_method(
        self,
        data: np.ndarray,
        k_range: range = range(2, 11),
    ) -> ElbowResult:
        """Find optimal K using elbow method.

        Computes inertia for each K, suggests K with maximum second derivative
        (elbow point).
        """
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        k_values: list[int] = []
        inertias: list[float] = []

        for k in k_range:
            if k >= len(data):
                break
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            model.fit(scaled)
            k_values.append(k)
            inertias.append(float(model.inertia_))

        # Find elbow using second derivative
        suggested_k = k_values[0] if k_values else 2

        if len(inertias) >= 3:
            # Compute second derivative (discrete approximation)
            second_derivs: list[float] = []
            for i in range(1, len(inertias) - 1):
                d2 = inertias[i - 1] - 2 * inertias[i] + inertias[i + 1]
                second_derivs.append(d2)

            if second_derivs:
                # Elbow is where second derivative is maximum
                max_idx = second_derivs.index(max(second_derivs))
                suggested_k = k_values[max_idx + 1]

        return ElbowResult(
            k_values=k_values,
            inertias=[round(i, 6) for i in inertias],
            suggested_k=suggested_k,
        )
