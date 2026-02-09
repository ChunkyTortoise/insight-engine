"""Advanced Anomaly Detection: Isolation Forest, LOF, Mahalanobis, and ensemble methods."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


@dataclass
class AdvancedAnomaly:
    """A single detected anomaly from an advanced method."""

    index: int
    value: float
    score: float
    method: str
    is_outlier: bool
    context: dict = field(default_factory=dict)


class IsolationForestDetector:
    """Anomaly detection using sklearn IsolationForest.

    Handles 1D arrays (reshape to 2D), NaN filtering, and empty data.
    """

    def detect(
        self,
        data: np.ndarray | list[float],
        contamination: float | str = "auto",
    ) -> list[AdvancedAnomaly]:
        """Detect anomalies using Isolation Forest.

        Args:
            data: Input data (1D or 2D).
            contamination: Expected proportion of outliers or "auto".

        Returns:
            List of AdvancedAnomaly for each data point.
        """
        arr = np.asarray(data, dtype=float)

        if arr.ndim == 1:
            # Filter NaN for 1D
            valid_mask = ~np.isnan(arr)
            clean = arr[valid_mask]
            if len(clean) < 2:
                return []
            clean_2d = clean.reshape(-1, 1)
        else:
            # Filter rows with any NaN for 2D
            valid_mask = ~np.isnan(arr).any(axis=1)
            clean = arr[valid_mask]
            if len(clean) < 2:
                return []
            clean_2d = clean

        scaler = StandardScaler()
        scaled = scaler.fit_transform(clean_2d)

        model = IsolationForest(contamination=contamination, random_state=42)
        labels = model.fit_predict(scaled)
        scores_raw = model.decision_function(scaled)

        # Normalize scores to 0-1 (higher = more anomalous)
        s_min, s_max = float(np.min(scores_raw)), float(np.max(scores_raw))
        if s_max - s_min > 0:
            norm_scores = (s_max - scores_raw) / (s_max - s_min)
        else:
            norm_scores = np.zeros(len(scores_raw))

        # Map back to original indices
        original_indices = np.where(valid_mask)[0]
        results: list[AdvancedAnomaly] = []
        for i, idx in enumerate(original_indices):
            val = float(clean[i]) if arr.ndim == 1 else float(clean[i][0])
            results.append(
                AdvancedAnomaly(
                    index=int(idx),
                    value=round(val, 6),
                    score=round(float(norm_scores[i]), 6),
                    method="isolation_forest",
                    is_outlier=bool(labels[i] == -1),
                    context={"decision_score": round(float(scores_raw[i]), 6)},
                )
            )
        return results


class LocalOutlierFactorDetector:
    """Anomaly detection using Local Outlier Factor.

    Score normalization to 0-1 range.
    """

    def detect(
        self,
        data: np.ndarray | list[float],
        n_neighbors: int = 20,
    ) -> list[AdvancedAnomaly]:
        """Detect anomalies using LOF.

        Args:
            data: Input data (1D or 2D).
            n_neighbors: Number of neighbors for LOF.

        Returns:
            List of AdvancedAnomaly for each data point.
        """
        arr = np.asarray(data, dtype=float)

        if arr.ndim == 1:
            valid_mask = ~np.isnan(arr)
            clean = arr[valid_mask]
            if len(clean) < 2:
                return []
            clean_2d = clean.reshape(-1, 1)
        else:
            valid_mask = ~np.isnan(arr).any(axis=1)
            clean = arr[valid_mask]
            if len(clean) < 2:
                return []
            clean_2d = clean

        scaler = StandardScaler()
        scaled = scaler.fit_transform(clean_2d)

        effective_neighbors = min(n_neighbors, len(clean) - 1)
        if effective_neighbors < 1:
            effective_neighbors = 1

        model = LocalOutlierFactor(n_neighbors=effective_neighbors, novelty=False)
        labels = model.fit_predict(scaled)
        scores_raw = model.negative_outlier_factor_

        # Normalize scores to 0-1 (higher = more anomalous)
        # LOF scores are negative; more negative = more anomalous
        s_min, s_max = float(np.min(scores_raw)), float(np.max(scores_raw))
        if s_max - s_min > 0:
            norm_scores = (s_max - scores_raw) / (s_max - s_min)
        else:
            norm_scores = np.zeros(len(scores_raw))

        original_indices = np.where(valid_mask)[0]
        results: list[AdvancedAnomaly] = []
        for i, idx in enumerate(original_indices):
            val = float(clean[i]) if arr.ndim == 1 else float(clean[i][0])
            results.append(
                AdvancedAnomaly(
                    index=int(idx),
                    value=round(val, 6),
                    score=round(float(norm_scores[i]), 6),
                    method="lof",
                    is_outlier=bool(labels[i] == -1),
                    context={"lof_score": round(float(scores_raw[i]), 6)},
                )
            )
        return results


class MahalanobisDetector:
    """Anomaly detection using Mahalanobis distance.

    Falls back to diagonal covariance when the matrix is singular.
    """

    def detect(
        self,
        data: np.ndarray | list[float],
        threshold: float = 3.0,
    ) -> list[AdvancedAnomaly]:
        """Detect anomalies using Mahalanobis distance.

        Args:
            data: Input data (1D or 2D).
            threshold: Distance threshold for outlier flagging.

        Returns:
            List of AdvancedAnomaly for each data point.
        """
        arr = np.asarray(data, dtype=float)

        if arr.ndim == 1:
            valid_mask = ~np.isnan(arr)
            clean = arr[valid_mask]
            if len(clean) < 2:
                return []
            clean_2d = clean.reshape(-1, 1)
        else:
            valid_mask = ~np.isnan(arr).any(axis=1)
            clean = arr[valid_mask]
            if len(clean) < 2:
                return []
            clean_2d = clean

        mean = np.mean(clean_2d, axis=0)
        cov = np.cov(clean_2d, rowvar=False)

        # Handle 1D case where cov is a scalar
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        # Try to invert; fall back to diagonal if singular
        try:
            cov_inv = np.linalg.inv(cov)
            # Verify inversion worked (check for very large values)
            if np.any(np.abs(cov_inv) > 1e15):
                raise np.linalg.LinAlgError("Near-singular")
        except np.linalg.LinAlgError:
            diag = np.diag(np.diag(cov))
            diag[diag == 0] = 1e-10
            cov_inv = np.linalg.inv(diag) if np.all(np.diag(diag) != 0) else np.eye(cov.shape[0])

        distances = []
        for row in clean_2d:
            diff = row - mean
            d = float(np.sqrt(np.abs(diff @ cov_inv @ diff)))
            distances.append(d)

        distances = np.array(distances)

        # Normalize to 0-1
        d_max = float(np.max(distances)) if len(distances) > 0 else 1.0
        if d_max > 0:
            norm_scores = distances / d_max
        else:
            norm_scores = np.zeros(len(distances))

        original_indices = np.where(valid_mask)[0]
        results: list[AdvancedAnomaly] = []
        for i, idx in enumerate(original_indices):
            val = float(clean[i]) if arr.ndim == 1 else float(clean[i][0])
            results.append(
                AdvancedAnomaly(
                    index=int(idx),
                    value=round(val, 6),
                    score=round(float(norm_scores[i]), 6),
                    method="mahalanobis",
                    is_outlier=bool(distances[i] > threshold),
                    context={"mahalanobis_distance": round(float(distances[i]), 6)},
                )
            )
        return results


class AnomalyEnsemble:
    """Combine multiple anomaly detectors with voting strategies."""

    def __init__(self) -> None:
        self._detectors: dict[str, object] = {
            "isolation_forest": IsolationForestDetector(),
            "lof": LocalOutlierFactorDetector(),
            "mahalanobis": MahalanobisDetector(),
        }

    def detect(
        self,
        data: np.ndarray | list[float],
        methods: list[str] | None = None,
        voting: str = "majority",
    ) -> list[AdvancedAnomaly]:
        """Detect anomalies using ensemble of methods.

        Args:
            data: Input data (1D or 2D).
            methods: List of method names to use. Defaults to all three.
            voting: "majority" (>50% agree) or "any" (at least one flags).

        Returns:
            List of AdvancedAnomaly with ensemble results.
        """
        if methods is None:
            methods = ["isolation_forest", "lof", "mahalanobis"]

        per_method = self._run_methods(data, methods)

        if not per_method:
            return []

        # Collect results by index
        index_results: dict[int, list[AdvancedAnomaly]] = {}
        for method_results in per_method.values():
            for anomaly in method_results:
                if anomaly.index not in index_results:
                    index_results[anomaly.index] = []
                index_results[anomaly.index].append(anomaly)

        num_methods = len(per_method)
        results: list[AdvancedAnomaly] = []
        for idx in sorted(index_results.keys()):
            items = index_results[idx]
            outlier_votes = sum(1 for a in items if a.is_outlier)
            avg_score = sum(a.score for a in items) / len(items)

            if voting == "majority":
                is_outlier = outlier_votes > num_methods / 2
            elif voting == "any":
                is_outlier = outlier_votes > 0
            else:
                raise ValueError(f"Unknown voting strategy: {voting}")

            results.append(
                AdvancedAnomaly(
                    index=idx,
                    value=items[0].value,
                    score=round(avg_score, 6),
                    method="ensemble",
                    is_outlier=is_outlier,
                    context={
                        "voting": voting,
                        "outlier_votes": outlier_votes,
                        "total_methods": num_methods,
                        "methods_used": list(per_method.keys()),
                    },
                )
            )
        return results

    def compare_methods(
        self,
        data: np.ndarray | list[float],
    ) -> dict[str, list[AdvancedAnomaly]]:
        """Run all detectors and return per-method results.

        Args:
            data: Input data (1D or 2D).

        Returns:
            Dict mapping method name to list of anomalies.
        """
        return self._run_methods(data, ["isolation_forest", "lof", "mahalanobis"])

    def _run_methods(
        self,
        data: np.ndarray | list[float],
        methods: list[str],
    ) -> dict[str, list[AdvancedAnomaly]]:
        """Run specified detection methods."""
        results: dict[str, list[AdvancedAnomaly]] = {}
        for method_name in methods:
            detector = self._detectors.get(method_name)
            if detector is None:
                raise ValueError(f"Unknown method: {method_name}")
            result = detector.detect(data)
            if result:
                results[method_name] = result
        return results
