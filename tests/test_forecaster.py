"""Tests for the time series forecasting module."""

from insight_engine.forecaster import (
    ForecastComparison,
    Forecaster,
    ForecastResult,
)


class TestMovingAverage:
    def test_basic_prediction(self):
        f = Forecaster()
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = f.moving_average(data, window=3, horizon=3)
        assert isinstance(result, ForecastResult)
        assert result.method == "moving_average"
        assert len(result.predictions) == 3
        # Predictions should be reasonable (near the end of the series)
        for p in result.predictions:
            assert p >= 10.0

    def test_window_size_affects_result(self):
        f = Forecaster()
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        r1 = f.moving_average(data, window=2, horizon=3)
        r2 = f.moving_average(data, window=5, horizon=3)
        assert r1.predictions != r2.predictions

    def test_short_data(self):
        f = Forecaster()
        data = [5.0, 10.0]
        result = f.moving_average(data, window=5, horizon=3)
        assert len(result.predictions) == 3

    def test_constant_data(self):
        f = Forecaster()
        data = [42.0] * 10
        result = f.moving_average(data, window=3, horizon=5)
        for p in result.predictions:
            assert abs(p - 42.0) < 1e-6


class TestExponentialSmoothing:
    def test_basic_smoothing(self):
        f = Forecaster()
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = f.exponential_smoothing(data, alpha=0.3, horizon=5)
        assert isinstance(result, ForecastResult)
        assert result.method == "exponential_smoothing"
        assert len(result.predictions) == 5

    def test_alpha_extremes(self):
        f = Forecaster()
        data = [1.0, 10.0, 2.0, 9.0, 3.0, 8.0, 4.0, 7.0]
        r_low = f.exponential_smoothing(data, alpha=0.01, horizon=3)
        r_high = f.exponential_smoothing(data, alpha=0.99, horizon=3)
        # Different alphas should produce different predictions
        assert r_low.predictions != r_high.predictions

    def test_single_value(self):
        f = Forecaster()
        data = [100.0]
        result = f.exponential_smoothing(data, horizon=3)
        assert len(result.predictions) == 3
        for p in result.predictions:
            assert abs(p - 100.0) < 1e-6


class TestLinearTrend:
    def test_perfect_linear(self):
        f = Forecaster()
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = f.linear_trend(data, horizon=3)
        assert abs(result.predictions[0] - 60.0) < 1e-3
        assert abs(result.predictions[1] - 70.0) < 1e-3
        assert abs(result.predictions[2] - 80.0) < 1e-3

    def test_flat_data(self):
        f = Forecaster()
        data = [5.0] * 10
        result = f.linear_trend(data, horizon=5)
        for p in result.predictions:
            assert abs(p - 5.0) < 1e-3

    def test_negative_trend(self):
        f = Forecaster()
        data = [100.0, 90.0, 80.0, 70.0, 60.0]
        result = f.linear_trend(data, horizon=3)
        assert result.predictions[0] < 60.0
        assert result.predictions[1] < result.predictions[0]


class TestEnsemble:
    def test_ensemble_averages(self):
        f = Forecaster()
        data = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        ens = f.ensemble(data, horizon=3)
        ma = f.moving_average(data, horizon=3)
        es = f.exponential_smoothing(data, horizon=3)
        lt = f.linear_trend(data, horizon=3)
        for i in range(3):
            low = min(ma.predictions[i], es.predictions[i], lt.predictions[i])
            high = max(ma.predictions[i], es.predictions[i], lt.predictions[i])
            assert low - 1e-6 <= ens.predictions[i] <= high + 1e-6

    def test_ensemble_produces_result(self):
        f = Forecaster()
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = f.ensemble(data, horizon=5)
        assert isinstance(result, ForecastResult)
        assert result.method == "ensemble"


class TestCompareForecasts:
    def test_comparison_returns_all_methods(self):
        f = Forecaster()
        data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        comp = f.compare_forecasts(data, horizon=3)
        assert isinstance(comp, ForecastComparison)
        assert len(comp.results) == 4
        assert "moving_average" in comp.results
        assert "exponential_smoothing" in comp.results
        assert "linear_trend" in comp.results
        assert "ensemble" in comp.results

    def test_best_method_selected(self):
        f = Forecaster()
        data = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        comp = f.compare_forecasts(data, horizon=3)
        # best_mae should match the MAE of the best method
        assert comp.best_mae == comp.results[comp.best_method].mae
        # best_mae should be the minimum
        for result in comp.results.values():
            assert comp.best_mae <= result.mae + 1e-6

    def test_comparison_with_trend_data(self):
        f = Forecaster()
        # Perfectly linear data
        data = [float(i * 10) for i in range(20)]
        comp = f.compare_forecasts(data, horizon=5)
        # linear_trend should have ~0 MAE for perfectly linear data
        assert comp.results["linear_trend"].mae < 1e-3


class TestErrorMetrics:
    def test_mae_non_negative(self):
        f = Forecaster()
        data = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0]
        comp = f.compare_forecasts(data)
        for result in comp.results.values():
            assert result.mae >= 0.0

    def test_rmse_gte_mae(self):
        f = Forecaster()
        data = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0]
        comp = f.compare_forecasts(data)
        for result in comp.results.values():
            assert result.rmse >= result.mae - 1e-6

    def test_mape_handles_zeros(self):
        f = Forecaster()
        data = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0]
        comp = f.compare_forecasts(data)
        for result in comp.results.values():
            assert result.mape >= 0.0
