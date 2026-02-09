"""Tests for the dimensionality reduction module."""

import numpy as np

from insight_engine.dimensionality import DimensionalityReducer, ReductionResult, ScreePlotData


def _make_high_dim_data() -> np.ndarray:
    """Create high-dimensional data with some structure."""
    rng = np.random.RandomState(42)
    # 3 latent factors, 10 features
    n = 50
    z = rng.randn(n, 3)
    W = rng.randn(3, 10)
    noise = rng.randn(n, 10) * 0.1
    return z @ W + noise


class TestPCA:
    def test_basic_pca(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        result = dr.pca(data, n_components=2)
        assert isinstance(result, ReductionResult)
        assert result.method == "pca"
        assert result.n_components == 2

    def test_output_shape(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        result = dr.pca(data, n_components=3)
        assert result.transformed.shape == (50, 3)

    def test_explained_variance_present(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        result = dr.pca(data, n_components=2)
        assert result.explained_variance is not None
        assert len(result.explained_variance) == 2
        assert all(0 <= v <= 1 for v in result.explained_variance)

    def test_total_variance(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        result = dr.pca(data, n_components=2)
        assert result.total_variance_explained is not None
        assert 0 < result.total_variance_explained <= 1.0

    def test_variance_sums_correctly(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        result = dr.pca(data, n_components=3)
        ev_sum = round(sum(result.explained_variance), 6)
        assert abs(ev_sum - result.total_variance_explained) < 0.001

    def test_more_components_more_variance(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        r2 = dr.pca(data, n_components=2)
        r5 = dr.pca(data, n_components=5)
        assert r5.total_variance_explained >= r2.total_variance_explained


class TestAutoPCA:
    def test_auto_selects_components(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        result = dr.auto_pca(data, variance_threshold=0.95)
        assert isinstance(result, ReductionResult)
        assert result.method == "auto_pca"
        assert result.total_variance_explained >= 0.95

    def test_low_threshold_fewer_components(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        r_low = dr.auto_pca(data, variance_threshold=0.5)
        r_high = dr.auto_pca(data, variance_threshold=0.99)
        assert r_low.n_components <= r_high.n_components

    def test_auto_pca_shape(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        result = dr.auto_pca(data, variance_threshold=0.9)
        assert result.transformed.shape[0] == 50
        assert result.transformed.shape[1] == result.n_components


class TestTSNE:
    def test_basic_tsne(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        result = dr.tsne(data, n_components=2)
        assert isinstance(result, ReductionResult)
        assert result.method == "tsne"

    def test_tsne_2d_output(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        result = dr.tsne(data, n_components=2)
        assert result.transformed.shape == (50, 2)

    def test_tsne_no_variance(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        result = dr.tsne(data)
        assert result.explained_variance is None
        assert result.total_variance_explained is None

    def test_tsne_reproducibility(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        r1 = dr.tsne(data, random_state=42)
        r2 = dr.tsne(data, random_state=42)
        np.testing.assert_array_almost_equal(r1.transformed, r2.transformed)


class TestCompare:
    def test_compare_returns_both(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        results = dr.compare(data, n_components=2)
        assert "pca" in results
        assert "tsne" in results

    def test_compare_shapes_match(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        results = dr.compare(data, n_components=2)
        assert results["pca"].transformed.shape == results["tsne"].transformed.shape

    def test_compare_n_components(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        results = dr.compare(data, n_components=2)
        assert results["pca"].n_components == 2
        assert results["tsne"].n_components == 2


class TestScreePlot:
    def test_scree_data_lengths(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        scree = dr.scree_plot_data(data)
        assert isinstance(scree, ScreePlotData)
        n_features = data.shape[1]
        max_comp = min(data.shape[0], n_features)
        assert len(scree.eigenvalues) == max_comp
        assert len(scree.explained_variance_ratio) == max_comp
        assert len(scree.cumulative_variance) == max_comp

    def test_cumulative_reaches_one(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        scree = dr.scree_plot_data(data)
        assert abs(scree.cumulative_variance[-1] - 1.0) < 0.01

    def test_eigenvalues_decreasing(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        scree = dr.scree_plot_data(data)
        for i in range(len(scree.eigenvalues) - 1):
            assert scree.eigenvalues[i] >= scree.eigenvalues[i + 1]

    def test_cumulative_monotonic(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        scree = dr.scree_plot_data(data)
        for i in range(len(scree.cumulative_variance) - 1):
            assert scree.cumulative_variance[i] <= scree.cumulative_variance[i + 1]


class TestTransformNew:
    def test_transform_new_shape(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        dr.pca(data, n_components=3)
        new_data = np.random.RandomState(99).randn(10, 10)
        transformed = dr.transform_new(new_data)
        assert transformed.shape == (10, 3)

    def test_transform_new_without_fit_raises(self):
        dr = DimensionalityReducer()
        try:
            dr.transform_new(np.random.randn(5, 3))
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_transform_new_after_auto_pca(self):
        data = _make_high_dim_data()
        dr = DimensionalityReducer()
        result = dr.auto_pca(data, variance_threshold=0.9)
        new_data = np.random.RandomState(99).randn(5, 10)
        transformed = dr.transform_new(new_data)
        assert transformed.shape == (5, result.n_components)


class TestEdgeCases:
    def test_single_feature(self):
        rng = np.random.RandomState(42)
        data = rng.randn(20, 1)
        dr = DimensionalityReducer()
        result = dr.pca(data, n_components=1)
        assert result.transformed.shape == (20, 1)

    def test_more_components_than_features(self):
        rng = np.random.RandomState(42)
        data = rng.randn(20, 3)
        dr = DimensionalityReducer()
        result = dr.pca(data, n_components=10)
        # Should cap at min(n_samples, n_features)
        assert result.n_components == 3

    def test_small_dataset(self):
        rng = np.random.RandomState(42)
        data = rng.randn(5, 10)
        dr = DimensionalityReducer()
        result = dr.pca(data, n_components=2)
        assert result.transformed.shape == (5, 2)
