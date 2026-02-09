"""Tests for regression diagnostics."""

from __future__ import annotations

import numpy as np

from insight_engine.regression_diagnostics import (
    DiagnosticsReport,
    RegressionDiagnostics,
    ResidualStats,
    VIFResult,
)


def _well_behaved_data(n: int = 100, seed: int = 42):
    """Generate well-behaved linear data for testing."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    y = 3.0 * X[:, 0] + 1.5 * X[:, 1] + rng.randn(n) * 0.1
    return X, y


def _multicollinear_data(n: int = 100, seed: int = 42):
    """Generate data with high multicollinearity."""
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n)
    x2 = x1 + rng.randn(n) * 0.01  # x2 ~ x1
    X = np.column_stack([x1, x2])
    y = 2.0 * x1 + rng.randn(n) * 0.5
    return X, y


class TestResidualAnalysis:
    def setup_method(self):
        self.diag = RegressionDiagnostics()

    def test_well_behaved(self):
        X, y = _well_behaved_data()
        from sklearn.linear_model import LinearRegression

        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        result = self.diag.residual_analysis(y, y_pred)
        assert isinstance(result, ResidualStats)
        assert abs(result.mean) < 0.1  # Residuals centered near 0

    def test_normality_check(self):
        X, y = _well_behaved_data()
        from sklearn.linear_model import LinearRegression

        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        result = self.diag.residual_analysis(y, y_pred)
        # Well-behaved data should have normal residuals
        assert result.is_normal is True
        assert result.p_value > 0.05

    def test_few_observations(self):
        result = self.diag.residual_analysis(np.array([1.0, 2.0]), np.array([1.1, 2.1]))
        assert isinstance(result, ResidualStats)


class TestVIF:
    def setup_method(self):
        self.diag = RegressionDiagnostics()

    def test_low_vif(self):
        X, _ = _well_behaved_data()
        results = self.diag.vif(X, feature_names=["x1", "x2"])
        assert len(results) == 2
        assert all(isinstance(r, VIFResult) for r in results)
        assert all(r.vif_value < 10 for r in results)
        assert all(not r.is_problematic for r in results)

    def test_high_vif_multicollinear(self):
        X, _ = _multicollinear_data()
        results = self.diag.vif(X)
        assert any(r.is_problematic for r in results)
        assert any(r.vif_value > 10 for r in results)

    def test_single_feature(self):
        X = np.random.randn(50, 1)
        results = self.diag.vif(X)
        assert len(results) == 1
        assert results[0].vif_value == 1.0

    def test_feature_names(self):
        X, _ = _well_behaved_data()
        results = self.diag.vif(X, feature_names=["alpha", "beta"])
        assert results[0].feature == "alpha"
        assert results[1].feature == "beta"


class TestCooksDistance:
    def setup_method(self):
        self.diag = RegressionDiagnostics()

    def test_no_influential(self):
        X, y = _well_behaved_data()
        cooks = self.diag.cooks_distance(X, y)
        assert len(cooks) == len(y)
        # Well-behaved data should have low Cook's distances
        assert max(cooks) < 1.0

    def test_influential_outlier(self):
        X, y = _well_behaved_data()
        # Add a huge outlier
        X_mod = np.vstack([X, [[100, 100]]])
        y_mod = np.append(y, -1000)
        cooks = self.diag.cooks_distance(X_mod, y_mod)
        # The last observation should have the highest Cook's distance
        assert cooks[-1] == max(cooks)

    def test_length_matches(self):
        X, y = _well_behaved_data(n=50)
        cooks = self.diag.cooks_distance(X, y)
        assert len(cooks) == 50


class TestHeteroscedasticity:
    def setup_method(self):
        self.diag = RegressionDiagnostics()

    def test_homoscedastic(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 1)
        residuals = rng.randn(100) * 0.5  # Constant variance
        p_val = self.diag.heteroscedasticity_test(residuals, X)
        assert p_val > 0.05  # Not significant = homoscedastic

    def test_heteroscedastic(self):
        rng = np.random.RandomState(42)
        n = 500
        X = np.abs(rng.randn(n, 1)) + 0.5
        # Variance strongly increases with X
        residuals = rng.randn(n) * (X[:, 0] ** 3)
        p_val = self.diag.heteroscedasticity_test(residuals, X)
        assert p_val < 0.05  # Significant = heteroscedastic


class TestDurbinWatson:
    def setup_method(self):
        self.diag = RegressionDiagnostics()

    def test_no_autocorrelation(self):
        rng = np.random.RandomState(42)
        residuals = rng.randn(100)
        dw = self.diag.durbin_watson(residuals)
        assert 1.5 < dw < 2.5  # Near 2 = no autocorrelation

    def test_positive_autocorrelation(self):
        # Create positively autocorrelated residuals
        rng = np.random.RandomState(42)
        residuals = np.zeros(100)
        residuals[0] = rng.randn()
        for i in range(1, 100):
            residuals[i] = 0.95 * residuals[i - 1] + rng.randn() * 0.1
        dw = self.diag.durbin_watson(residuals)
        assert dw < 1.5  # Low DW = positive autocorrelation

    def test_short_series(self):
        dw = self.diag.durbin_watson(np.array([1.0]))
        assert dw == 2.0  # Default for too-short series


class TestFit:
    def setup_method(self):
        self.diag = RegressionDiagnostics()

    def test_full_report(self):
        X, y = _well_behaved_data()
        report = self.diag.fit(X, y, feature_names=["x1", "x2"])
        assert isinstance(report, DiagnosticsReport)
        assert isinstance(report.residuals, ResidualStats)
        assert len(report.vif_results) == 2
        assert len(report.cooks_distances) == len(y)
        assert 0 <= report.dw_statistic <= 4
        assert 0 <= report.heteroscedasticity_pvalue <= 1

    def test_well_behaved_passes(self):
        X, y = _well_behaved_data()
        report = self.diag.fit(X, y)
        assert report.residuals.is_normal is True
        assert all(not v.is_problematic for v in report.vif_results)
        assert 1.5 < report.dw_statistic < 2.5

    def test_multicollinear_flagged(self):
        X, y = _multicollinear_data()
        report = self.diag.fit(X, y)
        assert any(v.is_problematic for v in report.vif_results)

    def test_many_features(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = X @ rng.randn(5) + rng.randn(100) * 0.1
        report = self.diag.fit(X, y)
        assert len(report.vif_results) == 5
