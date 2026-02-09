"""Dimensionality reduction: PCA, t-SNE, auto-PCA, scree plots."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


@dataclass
class ReductionResult:
    """Result of a dimensionality reduction operation."""

    method: str
    transformed: np.ndarray
    n_components: int
    explained_variance: list[float] | None = None
    total_variance_explained: float | None = None


@dataclass
class ScreePlotData:
    """Data for constructing a scree plot."""

    eigenvalues: list[float] = field(default_factory=list)
    explained_variance_ratio: list[float] = field(default_factory=list)
    cumulative_variance: list[float] = field(default_factory=list)


class DimensionalityReducer:
    """Dimensionality reduction toolkit."""

    def __init__(self) -> None:
        self._last_pca: PCA | None = None
        self._last_scaler: StandardScaler | None = None

    def pca(
        self,
        data: np.ndarray,
        n_components: int = 2,
    ) -> ReductionResult:
        """Standard PCA dimensionality reduction."""
        data = np.asarray(data, dtype=float)
        # Cap components at min(n_samples, n_features)
        max_components = min(data.shape[0], data.shape[1])
        n_components = min(n_components, max_components)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        model = PCA(n_components=n_components)
        transformed = model.fit_transform(scaled)

        self._last_pca = model
        self._last_scaler = scaler

        ev_ratio = [round(float(v), 6) for v in model.explained_variance_ratio_]
        total = round(float(sum(model.explained_variance_ratio_)), 6)

        return ReductionResult(
            method="pca",
            transformed=transformed,
            n_components=n_components,
            explained_variance=ev_ratio,
            total_variance_explained=total,
        )

    def auto_pca(
        self,
        data: np.ndarray,
        variance_threshold: float = 0.95,
    ) -> ReductionResult:
        """Auto-select number of PCA components to explain threshold% variance."""
        data = np.asarray(data, dtype=float)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        # Fit full PCA first
        max_components = min(data.shape[0], data.shape[1])
        full_pca = PCA(n_components=max_components)
        full_pca.fit(scaled)

        # Find minimum components for threshold
        cumulative = np.cumsum(full_pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
        n_components = min(n_components, max_components)

        # Refit with selected components
        model = PCA(n_components=n_components)
        transformed = model.fit_transform(scaled)

        self._last_pca = model
        self._last_scaler = scaler

        ev_ratio = [round(float(v), 6) for v in model.explained_variance_ratio_]
        total = round(float(sum(model.explained_variance_ratio_)), 6)

        return ReductionResult(
            method="auto_pca",
            transformed=transformed,
            n_components=n_components,
            explained_variance=ev_ratio,
            total_variance_explained=total,
        )

    def tsne(
        self,
        data: np.ndarray,
        n_components: int = 2,
        perplexity: float = 30.0,
        random_state: int = 42,
    ) -> ReductionResult:
        """t-SNE dimensionality reduction."""
        data = np.asarray(data, dtype=float)
        # Perplexity must be less than n_samples
        effective_perplexity = min(perplexity, max(1.0, float(data.shape[0] - 1)))

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        model = TSNE(
            n_components=n_components,
            perplexity=effective_perplexity,
            random_state=random_state,
        )
        transformed = model.fit_transform(scaled)

        return ReductionResult(
            method="tsne",
            transformed=transformed,
            n_components=n_components,
            explained_variance=None,
            total_variance_explained=None,
        )

    def compare(
        self,
        data: np.ndarray,
        n_components: int = 2,
    ) -> dict[str, ReductionResult]:
        """Run PCA and t-SNE side by side for comparison."""
        pca_result = self.pca(data, n_components=n_components)
        tsne_result = self.tsne(data, n_components=n_components)
        return {"pca": pca_result, "tsne": tsne_result}

    def scree_plot_data(self, data: np.ndarray) -> ScreePlotData:
        """Compute eigenvalues and cumulative variance for scree plot visualization."""
        data = np.asarray(data, dtype=float)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        max_components = min(data.shape[0], data.shape[1])
        model = PCA(n_components=max_components)
        model.fit(scaled)

        eigenvalues = [round(float(v), 6) for v in model.explained_variance_]
        ev_ratio = [round(float(v), 6) for v in model.explained_variance_ratio_]
        cumulative = [round(float(v), 6) for v in np.cumsum(model.explained_variance_ratio_)]

        return ScreePlotData(
            eigenvalues=eigenvalues,
            explained_variance_ratio=ev_ratio,
            cumulative_variance=cumulative,
        )

    def transform_new(self, new_data: np.ndarray) -> np.ndarray:
        """Project new data using the last fitted PCA model."""
        if self._last_pca is None or self._last_scaler is None:
            raise RuntimeError("No PCA model fitted yet. Call pca() or auto_pca() first.")

        new_data = np.asarray(new_data, dtype=float)
        scaled = self._last_scaler.transform(new_data)
        return self._last_pca.transform(scaled)
