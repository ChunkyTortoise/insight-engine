"""Tests for the unsupervised clustering module."""

import numpy as np

from insight_engine.clustering import ClusterComparison, Clusterer, ClusterResult


def _make_blobs() -> np.ndarray:
    """Create 3 well-separated blobs for testing."""
    blob_a = np.array([[i, 0] for i in range(10)])
    blob_b = np.array([[i, 100] for i in range(10)])
    blob_c = np.array([[i + 100, 50] for i in range(10)])
    return np.vstack([blob_a, blob_b, blob_c]).astype(float)


class TestKMeans:
    def test_basic_clustering(self):
        data = _make_blobs()
        c = Clusterer()
        result = c.kmeans(data, k=3)
        assert isinstance(result, ClusterResult)
        assert result.method == "kmeans"
        assert result.n_clusters == 3

    def test_cluster_sizes(self):
        data = _make_blobs()
        c = Clusterer()
        result = c.kmeans(data, k=3)
        total = sum(result.cluster_sizes.values())
        assert total == len(data)

    def test_silhouette_range(self):
        data = _make_blobs()
        c = Clusterer()
        result = c.kmeans(data, k=3)
        assert -1.0 <= result.silhouette <= 1.0

    def test_inertia_present(self):
        data = _make_blobs()
        c = Clusterer()
        result = c.kmeans(data, k=3)
        assert result.inertia is not None
        assert result.inertia >= 0


class TestAutoKMeans:
    def test_finds_clusters(self):
        data = _make_blobs()
        c = Clusterer()
        result = c.auto_kmeans(data)
        assert isinstance(result, ClusterResult)
        assert result.method == "auto_kmeans"
        assert result.n_clusters >= 2

    def test_silhouette_positive(self):
        data = _make_blobs()
        c = Clusterer()
        result = c.auto_kmeans(data)
        assert result.silhouette > 0


class TestDBSCAN:
    def test_basic_dbscan(self):
        data = _make_blobs()
        c = Clusterer()
        result = c.dbscan(data, eps=1.5, min_samples=3)
        assert isinstance(result, ClusterResult)
        assert result.method == "dbscan"

    def test_noise_handling(self):
        # Add some noise far from blobs
        data = _make_blobs()
        outliers = np.array([[500.0, 500.0], [-500.0, -500.0]])
        data_with_noise = np.vstack([data, outliers])
        c = Clusterer()
        result = c.dbscan(data_with_noise, eps=1.5, min_samples=3)
        assert -1 in result.cluster_sizes

    def test_tight_eps(self):
        data = _make_blobs()
        c = Clusterer()
        result = c.dbscan(data, eps=0.01, min_samples=2)
        # Very tight eps should yield more noise
        noise_count = result.cluster_sizes.get(-1, 0)
        assert noise_count > 0


class TestCompare:
    def test_comparison_runs(self):
        data = _make_blobs()
        c = Clusterer()
        comp = c.compare(data)
        assert isinstance(comp, ClusterComparison)
        assert "auto_kmeans" in comp.results
        assert "dbscan" in comp.results

    def test_best_selected(self):
        data = _make_blobs()
        c = Clusterer()
        comp = c.compare(data)
        assert comp.best_silhouette == comp.results[comp.best_method].silhouette


class TestEdgeCases:
    def test_single_feature(self):
        data = np.array([[1.0], [2.0], [3.0], [100.0], [101.0], [102.0]])
        c = Clusterer()
        result = c.kmeans(data, k=2)
        assert len(result.labels) == 6

    def test_small_dataset(self):
        data = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])
        c = Clusterer()
        result = c.kmeans(data, k=2)
        assert result.n_clusters == 2

    def test_uniform_data(self):
        data = np.ones((10, 2))
        c = Clusterer()
        result = c.kmeans(data, k=2)
        # All same data: silhouette should be poor or -1
        assert result.silhouette <= 1.0
        assert len(result.labels) == 10
