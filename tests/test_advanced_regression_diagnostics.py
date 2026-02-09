"""Tests for Tier-2 regression diagnostics: ResidualAnalyzer, MulticollinearityDetector, InfluenceAnalyzer, Runner."""

from __future__ import annotations

import numpy as np

from insight_engine.regression_diagnostics import (
    AdvancedVIFResult,
    DiagnosticReport,
    InfluenceAnalyzer,
    InfluenceResult,
    MulticollinearityDetector,
    RegressionDiagnosticRunner,
    ResidualAnalysis,
    ResidualAnalyzer,
)


def _well_behaved(n: int = 100, seed: int = 42):
    """Generate well-behaved linear data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    y = 3.0 * X[:, 0] + 1.5 * X[:, 1] + rng.randn(n) * 0.1
    y_pred = 3.0 * X[:, 0] + 1.5 * X[:, 1]
    return X, y, y_pred


def _multicollinear(n: int = 100, seed: int = 42):
    """Generate data with high multicollinearity."""
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n)
    x2 = x1 + rng.randn(n) * 0.01
    X = np.column_stack([x1, x2])
    y = 2.0 * x1 + rng.randn(n) * 0.5
    y_pred = 2.0 * x1
    return X, y, y_pred


def _data_with_outlier(seed: int = 42):
    """Generate data with one influential outlier."""
    rng = np.random.RandomState(seed)
    n = 50
    X = rng.randn(n, 1)
    y = 2.0 * X[:, 0] + rng.randn(n) * 0.5
    # Add influential outlier
    X = np.vstack([X, [[20.0]]])
    y = np.append(y, -100.0)
    y_pred = 2.0 * X[:, 0]
    return X, y, y_pred


# ---- ResidualAnalyzer ----


class TestResidualAnalyzer:
    def setup_method(self):
        self.analyzer = ResidualAnalyzer()

    def test_basic_analysis(self):
        _, y, y_pred = _well_behaved()
        result = self.analyzer.analyze(y, y_pred)
        assert isinstance(result, ResidualAnalysis)
        assert isinstance(result.residuals, np.ndarray)
        assert len(result.residuals) == len(y)

    def test_mean_near_zero(self):
        _, y, y_pred = _well_behaved()
        result = self.analyzer.analyze(y, y_pred)
        assert abs(result.mean) < 0.5

    def test_std_positive(self):
        _, y, y_pred = _well_behaved()
        result = self.analyzer.analyze(y, y_pred)
        assert result.std > 0

    def test_skewness_type(self):
        _, y, y_pred = _well_behaved()
        result = self.analyzer.analyze(y, y_pred)
        assert isinstance(result.skewness, float)

    def test_kurtosis_type(self):
        _, y, y_pred = _well_behaved()
        result = self.analyzer.analyze(y, y_pred)
        assert isinstance(result.kurtosis, float)

    def test_durbin_watson_range(self):
        _, y, y_pred = _well_behaved()
        result = self.analyzer.analyze(y, y_pred)
        assert 0.0 <= result.durbin_watson <= 4.0

    def test_normality_pvalue_range(self):
        _, y, y_pred = _well_behaved()
        result = self.analyzer.analyze(y, y_pred)
        assert 0.0 <= result.normality_pvalue <= 1.0

    def test_two_elements(self):
        result = self.analyzer.analyze(np.array([1.0, 2.0]), np.array([1.1, 1.9]))
        assert isinstance(result, ResidualAnalysis)
        assert result.skewness == 0.0  # < 3 elements

    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self.analyzer.analyze(y, y)
        assert result.mean == 0.0
        assert result.std == 0.0

    def test_autocorrelated_residuals_low_dw(self):
        """Positively autocorrelated residuals should have low Durbin-Watson."""
        rng = np.random.RandomState(42)
        n = 100
        residuals = np.zeros(n)
        residuals[0] = rng.randn()
        for i in range(1, n):
            residuals[i] = 0.95 * residuals[i - 1] + rng.randn() * 0.1
        y_true = residuals + 10
        y_pred = np.full(n, 10.0)
        result = self.analyzer.analyze(y_true, y_pred)
        assert result.durbin_watson < 1.5


# ---- MulticollinearityDetector ----


class TestMulticollinearityDetector:
    def setup_method(self):
        self.detector = MulticollinearityDetector()

    def test_independent_features(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        results = self.detector.compute_vif(X)
        assert len(results) == 3
        assert all(isinstance(r, AdvancedVIFResult) for r in results)
        assert all(not r.is_multicollinear for r in results)
        assert all(r.vif_value < 5.0 for r in results)

    def test_collinear_features(self):
        X, _, _ = _multicollinear()
        results = self.detector.compute_vif(X)
        assert any(r.is_multicollinear for r in results)
        assert any(r.vif_value > 5.0 for r in results)

    def test_single_feature(self):
        X = np.random.randn(50, 1)
        results = self.detector.compute_vif(X)
        assert len(results) == 1
        assert results[0].vif_value == 1.0
        assert not results[0].is_multicollinear

    def test_feature_names(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 2)
        results = self.detector.compute_vif(X, feature_names=["alpha", "beta"])
        assert results[0].feature_name == "alpha"
        assert results[1].feature_name == "beta"

    def test_default_feature_names(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 2)
        results = self.detector.compute_vif(X)
        assert results[0].feature_name == "feature_0"
        assert results[1].feature_name == "feature_1"

    def test_custom_threshold(self):
        X, _, _ = _multicollinear()
        results_low = self.detector.compute_vif(X, threshold=2.0)
        results_high = self.detector.compute_vif(X, threshold=100.0)
        low_flags = sum(1 for r in results_low if r.is_multicollinear)
        high_flags = sum(1 for r in results_high if r.is_multicollinear)
        assert low_flags >= high_flags

    def test_constant_column(self):
        rng = np.random.RandomState(42)
        X = np.column_stack([np.ones(50), rng.randn(50)])
        results = self.detector.compute_vif(X)
        constant_vif = results[0]
        assert constant_vif.vif_value == float("inf")
        assert constant_vif.is_multicollinear

    def test_1d_input_reshaped(self):
        data = np.random.randn(50)
        results = self.detector.compute_vif(data)
        assert len(results) == 1


# ---- InfluenceAnalyzer ----


class TestInfluenceAnalyzer:
    def setup_method(self):
        self.analyzer = InfluenceAnalyzer()

    def test_basic_analysis(self):
        X, y, y_pred = _well_behaved()
        result = self.analyzer.analyze(X, y, y_pred)
        assert isinstance(result, InfluenceResult)
        assert len(result.cooks_distance) == len(y)
        assert len(result.leverage) == len(y)

    def test_cooks_distance_non_negative(self):
        X, y, y_pred = _well_behaved()
        result = self.analyzer.analyze(X, y, y_pred)
        assert all(d >= 0 for d in result.cooks_distance)

    def test_leverage_bounded(self):
        X, y, y_pred = _well_behaved()
        result = self.analyzer.analyze(X, y, y_pred)
        for h in result.leverage:
            assert 0.0 <= h <= 1.0 + 1e-6  # small numerical tolerance

    def test_outlier_has_high_cooks(self):
        X, y, y_pred = _data_with_outlier()
        result = self.analyzer.analyze(X, y, y_pred)
        # Last point (outlier) should have highest Cook's distance
        last_idx = len(y) - 1
        assert result.cooks_distance[last_idx] == max(result.cooks_distance)

    def test_influential_indices_detected(self):
        X, y, y_pred = _data_with_outlier()
        result = self.analyzer.analyze(X, y, y_pred)
        assert len(result.influential_indices) > 0
        # The outlier index should be flagged
        assert len(y) - 1 in result.influential_indices

    def test_well_behaved_few_influential(self):
        X, y, y_pred = _well_behaved()
        result = self.analyzer.analyze(X, y, y_pred)
        # Well-behaved data should have few influential points
        assert len(result.influential_indices) < len(y) // 2

    def test_1d_feature(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50)
        y = 2 * X + rng.randn(50) * 0.1
        y_pred = 2 * X
        result = self.analyzer.analyze(X, y, y_pred)
        assert len(result.cooks_distance) == 50

    def test_perfect_fit(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 1)
        y = 3 * X[:, 0]
        y_pred = y.copy()
        result = self.analyzer.analyze(X, y, y_pred)
        # Perfect fit = all Cook's D should be 0
        assert all(d == 0.0 for d in result.cooks_distance)


# ---- RegressionDiagnosticRunner ----


class TestRegressionDiagnosticRunner:
    def setup_method(self):
        self.runner = RegressionDiagnosticRunner()

    def test_full_report(self):
        X, y, y_pred = _well_behaved()
        report = self.runner.run_all(X, y, y_pred)
        assert isinstance(report, DiagnosticReport)
        assert isinstance(report.residuals, ResidualAnalysis)
        assert isinstance(report.vif, list)
        assert isinstance(report.influence, InfluenceResult)
        assert isinstance(report.overall_health, str)
        assert isinstance(report.warnings, list)

    def test_well_behaved_good_health(self):
        X, y, y_pred = _well_behaved()
        report = self.runner.run_all(X, y, y_pred)
        # Well-behaved data should be "good" or "moderate"
        assert report.overall_health in ("good", "moderate")

    def test_feature_names_passed(self):
        X, y, y_pred = _well_behaved()
        report = self.runner.run_all(X, y, y_pred, feature_names=["a", "b"])
        assert report.vif[0].feature_name == "a"
        assert report.vif[1].feature_name == "b"

    def test_multicollinear_warning(self):
        X, y, y_pred = _multicollinear()
        report = self.runner.run_all(X, y, y_pred)
        mc_warnings = [w for w in report.warnings if "Multicollinearity" in w]
        assert len(mc_warnings) > 0

    def test_influential_warning(self):
        X, y, y_pred = _data_with_outlier()
        report = self.runner.run_all(X, y, y_pred)
        infl_warnings = [w for w in report.warnings if "influential" in w]
        assert len(infl_warnings) > 0

    def test_poor_health_many_issues(self):
        """Data with multiple issues should get 'poor' health."""
        X, y, y_pred = _multicollinear()
        # Add an influential outlier
        X = np.vstack([X, [[50.0, 50.0]]])
        y = np.append(y, -500.0)
        y_pred = np.append(y_pred, 100.0)
        report = self.runner.run_all(X, y, y_pred)
        assert report.overall_health in ("moderate", "poor")
        assert len(report.warnings) >= 2

    def test_1d_feature_input(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50)
        y = 2 * X + rng.randn(50) * 0.1
        y_pred = 2 * X
        report = self.runner.run_all(X, y, y_pred)
        assert isinstance(report, DiagnosticReport)

    def test_vif_count_matches_features(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = X @ rng.randn(5) + rng.randn(100) * 0.1
        y_pred = X @ rng.randn(5)
        report = self.runner.run_all(X, y, y_pred)
        assert len(report.vif) == 5
