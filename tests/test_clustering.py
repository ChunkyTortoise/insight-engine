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


class TestHierarchical:
    def test_basic_hierarchical(self):
        import numpy as np

        from insight_engine.clustering import Clusterer, ClusterResult

        c = Clusterer()
        data = np.random.rand(20, 3)
        result = c.hierarchical(data, n_clusters=3, linkage="ward")
        assert isinstance(result, ClusterResult)
        assert result.n_clusters == 3
        assert len(result.labels) == 20

    def test_linkage_methods(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(30, 2)
        linkages = ["ward", "complete", "average", "single"]
        for link in linkages:
            result = c.hierarchical(data, n_clusters=4, linkage=link)
            assert f"hierarchical_{link}" in result.method

    def test_different_k_values(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(50, 4)
        r1 = c.hierarchical(data, n_clusters=2)
        r2 = c.hierarchical(data, n_clusters=5)
        assert r1.n_clusters == 2
        assert r2.n_clusters == 5

    def test_cluster_sizes(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(40, 3)
        result = c.hierarchical(data, n_clusters=4)
        total_size = sum(result.cluster_sizes.values())
        assert total_size == 40

    def test_silhouette_score(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(25, 2)
        result = c.hierarchical(data, n_clusters=3)
        assert -1.0 <= result.silhouette <= 1.0

    def test_inertia_none(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(20, 2)
        result = c.hierarchical(data, n_clusters=3)
        assert result.inertia is None


class TestClusterEvaluation:
    def test_basic_evaluation(self):
        import numpy as np

        from insight_engine.clustering import Clusterer, ClusterEvaluation

        c = Clusterer()
        data = np.random.rand(30, 3)
        result = c.kmeans(data, k=3)
        eval_result = c.evaluate_clusters(data, result.labels)
        assert isinstance(eval_result, ClusterEvaluation)
        assert hasattr(eval_result, "silhouette")
        assert hasattr(eval_result, "calinski_harabasz")
        assert hasattr(eval_result, "davies_bouldin")

    def test_silhouette_range(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(40, 2)
        labels = [0, 1, 0, 1] * 10
        eval_result = c.evaluate_clusters(data, labels)
        assert -1.0 <= eval_result.silhouette <= 1.0

    def test_calinski_harabasz(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(50, 4)
        result = c.kmeans(data, k=4)
        eval_result = c.evaluate_clusters(data, result.labels)
        # CH score should be positive for valid clustering
        assert eval_result.calinski_harabasz >= 0.0

    def test_davies_bouldin(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(35, 3)
        result = c.kmeans(data, k=3)
        eval_result = c.evaluate_clusters(data, result.labels)
        # DB score should be non-negative
        assert eval_result.davies_bouldin >= 0.0

    def test_single_cluster(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(20, 2)
        labels = [0] * 20
        eval_result = c.evaluate_clusters(data, labels)
        # Single cluster should have poor scores
        assert eval_result.silhouette == -1.0

    def test_two_clusters(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(30, 2)
        labels = [0] * 15 + [1] * 15
        eval_result = c.evaluate_clusters(data, labels)
        assert eval_result.calinski_harabasz > 0.0


class TestClusterProfiling:
    def test_basic_profiling(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(40, 3)
        result = c.kmeans(data, k=3)
        profile = c.profile_clusters(data, result.labels)
        assert isinstance(profile, dict)
        assert len(profile) == 3

    def test_feature_names(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(30, 2)
        result = c.kmeans(data, k=2)
        profile = c.profile_clusters(data, result.labels, feature_names=["x", "y"])
        for cluster_id, stats in profile.items():
            assert "x" in stats
            assert "y" in stats

    def test_default_feature_names(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(25, 4)
        result = c.kmeans(data, k=3)
        profile = c.profile_clusters(data, result.labels)
        for cluster_id, stats in profile.items():
            assert "feature_0" in stats

    def test_statistics_present(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(50, 3)
        result = c.kmeans(data, k=4)
        profile = c.profile_clusters(data, result.labels)
        for cluster_id, feature_stats in profile.items():
            for fname, stats in feature_stats.items():
                assert "mean" in stats
                assert "std" in stats
                assert "size" in stats

    def test_cluster_sizes_match(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(60, 2)
        result = c.kmeans(data, k=3)
        profile = c.profile_clusters(data, result.labels)
        for cluster_id, feature_stats in profile.items():
            expected_size = result.cluster_sizes[cluster_id]
            for fname, stats in feature_stats.items():
                assert stats["size"] == expected_size

    def test_noise_cluster(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(30, 2)
        result = c.dbscan(data, eps=0.1, min_samples=2)
        profile = c.profile_clusters(data, result.labels)
        # Profile should handle noise cluster (-1) if present
        assert isinstance(profile, dict)


class TestElbowMethod:
    def test_basic_elbow(self):
        import numpy as np

        from insight_engine.clustering import Clusterer, ElbowResult

        c = Clusterer()
        data = np.random.rand(50, 3)
        result = c.elbow_method(data, k_range=range(2, 8))
        assert isinstance(result, ElbowResult)
        assert len(result.k_values) > 0
        assert len(result.inertias) == len(result.k_values)

    def test_suggested_k_in_range(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(60, 4)
        result = c.elbow_method(data, k_range=range(2, 10))
        assert result.suggested_k in result.k_values

    def test_inertia_decreases(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(40, 2)
        result = c.elbow_method(data, k_range=range(2, 7))
        # Inertia should generally decrease as k increases
        assert result.inertias[0] > result.inertias[-1]

    def test_custom_k_range(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(30, 3)
        result = c.elbow_method(data, k_range=range(3, 6))
        assert min(result.k_values) >= 3
        assert max(result.k_values) <= 5

    def test_small_dataset(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(10, 2)
        result = c.elbow_method(data, k_range=range(2, 5))
        # Should handle small datasets gracefully
        assert len(result.k_values) > 0

    def test_k_values_match_inertias(self):
        import numpy as np

        from insight_engine.clustering import Clusterer

        c = Clusterer()
        data = np.random.rand(70, 4)
        result = c.elbow_method(data, k_range=range(2, 12))
        assert len(result.k_values) == len(result.inertias)
