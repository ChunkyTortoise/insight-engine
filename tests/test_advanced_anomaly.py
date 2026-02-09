"""Tests for advanced anomaly detection: Isolation Forest, LOF, Mahalanobis, Ensemble."""

from __future__ import annotations

import numpy as np
import pytest

from insight_engine.advanced_anomaly import (
    AdvancedAnomaly,
    AnomalyEnsemble,
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    MahalanobisDetector,
)


def _normal_data(n: int = 100, seed: int = 42) -> np.ndarray:
    """Generate normal data with a few outliers."""
    rng = np.random.RandomState(seed)
    data = rng.randn(n) * 2 + 10
    # Inject outliers
    data[0] = 100.0
    data[1] = -50.0
    return data


def _clean_data(n: int = 100, seed: int = 42) -> np.ndarray:
    """Generate clean normal data with no outliers."""
    rng = np.random.RandomState(seed)
    return rng.randn(n)


def _multivariate_data(n: int = 100, seed: int = 42) -> np.ndarray:
    """Generate 2D data with an outlier."""
    rng = np.random.RandomState(seed)
    data = rng.randn(n, 2)
    data[0] = [20.0, 20.0]  # outlier
    return data


# ---- IsolationForestDetector ----


class TestIsolationForestDetector:
    def setup_method(self):
        self.detector = IsolationForestDetector()

    def test_basic_detection(self):
        data = _normal_data()
        results = self.detector.detect(data)
        assert len(results) == len(data[~np.isnan(data)])
        assert all(isinstance(r, AdvancedAnomaly) for r in results)
        assert all(r.method == "isolation_forest" for r in results)

    def test_outlier_flagged(self):
        data = _normal_data()
        results = self.detector.detect(data)
        outliers = [r for r in results if r.is_outlier]
        assert len(outliers) > 0
        outlier_indices = {r.index for r in outliers}
        # The extreme values at indices 0 and 1 should be flagged
        assert 0 in outlier_indices or 1 in outlier_indices

    def test_scores_bounded(self):
        data = _normal_data()
        results = self.detector.detect(data)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_empty_data(self):
        results = self.detector.detect(np.array([]))
        assert results == []

    def test_single_element(self):
        results = self.detector.detect(np.array([5.0]))
        assert results == []

    def test_nan_handling(self):
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 100.0])
        results = self.detector.detect(data)
        indices = {r.index for r in results}
        assert 2 not in indices  # NaN indices excluded
        assert 5 not in indices

    def test_all_same_values(self):
        data = np.array([5.0] * 20)
        results = self.detector.detect(data)
        # All same = no variation, might still return results
        assert isinstance(results, list)

    def test_2d_input(self):
        data = _multivariate_data()
        results = self.detector.detect(data)
        assert len(results) > 0
        assert all(isinstance(r, AdvancedAnomaly) for r in results)

    def test_context_has_decision_score(self):
        data = _normal_data()
        results = self.detector.detect(data)
        assert all("decision_score" in r.context for r in results)

    def test_contamination_param(self):
        data = _normal_data()
        results = self.detector.detect(data, contamination=0.1)
        outlier_count = sum(1 for r in results if r.is_outlier)
        # ~10% contamination
        assert outlier_count > 0


# ---- LocalOutlierFactorDetector ----


class TestLocalOutlierFactorDetector:
    def setup_method(self):
        self.detector = LocalOutlierFactorDetector()

    def test_basic_detection(self):
        data = _normal_data()
        results = self.detector.detect(data)
        assert len(results) > 0
        assert all(r.method == "lof" for r in results)

    def test_outlier_flagged(self):
        data = _normal_data()
        results = self.detector.detect(data)
        outliers = [r for r in results if r.is_outlier]
        assert len(outliers) > 0

    def test_scores_bounded(self):
        data = _normal_data()
        results = self.detector.detect(data)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_empty_data(self):
        results = self.detector.detect(np.array([]))
        assert results == []

    def test_single_element(self):
        results = self.detector.detect(np.array([42.0]))
        assert results == []

    def test_nan_filtering(self):
        data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        results = self.detector.detect(data)
        indices = {r.index for r in results}
        assert 1 not in indices

    def test_custom_neighbors(self):
        data = _normal_data(n=50)
        results = self.detector.detect(data, n_neighbors=5)
        assert len(results) > 0

    def test_small_dataset_neighbors_clamp(self):
        """n_neighbors should be clamped to len-1 for small datasets."""
        data = np.array([1.0, 2.0, 3.0, 100.0])
        results = self.detector.detect(data, n_neighbors=20)
        assert len(results) == 4

    def test_context_has_lof_score(self):
        data = _normal_data()
        results = self.detector.detect(data)
        assert all("lof_score" in r.context for r in results)


# ---- MahalanobisDetector ----


class TestMahalanobisDetector:
    def setup_method(self):
        self.detector = MahalanobisDetector()

    def test_basic_detection(self):
        data = _normal_data()
        results = self.detector.detect(data)
        assert len(results) > 0
        assert all(r.method == "mahalanobis" for r in results)

    def test_outlier_flagged_extreme(self):
        data = _normal_data()
        results = self.detector.detect(data, threshold=3.0)
        outliers = [r for r in results if r.is_outlier]
        assert len(outliers) > 0
        # Index 0 (value=100) or 1 (value=-50) should be flagged
        outlier_indices = {r.index for r in outliers}
        assert 0 in outlier_indices or 1 in outlier_indices

    def test_scores_bounded(self):
        data = _normal_data()
        results = self.detector.detect(data)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_empty_data(self):
        results = self.detector.detect(np.array([]))
        assert results == []

    def test_single_element(self):
        results = self.detector.detect(np.array([42.0]))
        assert results == []

    def test_all_same_values(self):
        """Constant data => singular covariance => fallback to diagonal."""
        data = np.array([5.0] * 20)
        results = self.detector.detect(data)
        assert isinstance(results, list)

    def test_multivariate_input(self):
        data = _multivariate_data()
        results = self.detector.detect(data)
        assert len(results) > 0

    def test_context_has_distance(self):
        data = _normal_data()
        results = self.detector.detect(data)
        assert all("mahalanobis_distance" in r.context for r in results)

    def test_custom_threshold(self):
        data = _normal_data()
        low_t = self.detector.detect(data, threshold=1.0)
        high_t = self.detector.detect(data, threshold=10.0)
        low_outliers = sum(1 for r in low_t if r.is_outlier)
        high_outliers = sum(1 for r in high_t if r.is_outlier)
        assert low_outliers >= high_outliers


# ---- AnomalyEnsemble ----


class TestAnomalyEnsemble:
    def setup_method(self):
        self.ensemble = AnomalyEnsemble()

    def test_majority_voting(self):
        data = _normal_data()
        results = self.ensemble.detect(data, voting="majority")
        assert len(results) > 0
        assert all(r.method == "ensemble" for r in results)
        assert all(r.context["voting"] == "majority" for r in results)

    def test_any_voting(self):
        data = _normal_data()
        results = self.ensemble.detect(data, voting="any")
        outliers_any = sum(1 for r in results if r.is_outlier)
        results_maj = self.ensemble.detect(data, voting="majority")
        outliers_maj = sum(1 for r in results_maj if r.is_outlier)
        # "any" voting should flag at least as many as "majority"
        assert outliers_any >= outliers_maj

    def test_single_method(self):
        data = _normal_data()
        results = self.ensemble.detect(data, methods=["isolation_forest"])
        assert len(results) > 0
        assert all(r.context["total_methods"] == 1 for r in results)

    def test_two_methods(self):
        data = _normal_data()
        results = self.ensemble.detect(data, methods=["isolation_forest", "lof"])
        assert len(results) > 0
        assert all(r.context["total_methods"] == 2 for r in results)

    def test_all_three_methods(self):
        data = _normal_data()
        results = self.ensemble.detect(data)
        if results:
            assert all(r.context["total_methods"] == 3 for r in results)

    def test_compare_methods(self):
        data = _normal_data()
        comparison = self.ensemble.compare_methods(data)
        assert isinstance(comparison, dict)
        assert "isolation_forest" in comparison
        assert "lof" in comparison
        assert "mahalanobis" in comparison

    def test_compare_methods_all_return_results(self):
        data = _normal_data()
        comparison = self.ensemble.compare_methods(data)
        for method, results in comparison.items():
            assert len(results) > 0
            assert all(r.method == method for r in results)

    def test_unknown_method_raises(self):
        data = _normal_data()
        with pytest.raises(ValueError, match="Unknown method"):
            self.ensemble.detect(data, methods=["nonexistent"])

    def test_unknown_voting_raises(self):
        data = _normal_data()
        with pytest.raises(ValueError, match="Unknown voting"):
            self.ensemble.detect(data, voting="weighted")

    def test_empty_data(self):
        results = self.ensemble.detect(np.array([]))
        assert results == []

    def test_ensemble_scores_averaged(self):
        data = _normal_data()
        results = self.ensemble.detect(data)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_outlier_votes_tracked(self):
        data = _normal_data()
        results = self.ensemble.detect(data)
        for r in results:
            assert "outlier_votes" in r.context
            assert "total_methods" in r.context
            assert r.context["outlier_votes"] <= r.context["total_methods"]
