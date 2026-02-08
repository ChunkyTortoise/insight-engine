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


class TestSeasonalDecomposition:
    def test_basic_decomposition(self):
        f = Forecaster()
        # Simple seasonal data: [1,2,1,2,1,2,...]
        data = [1.0, 2.0] * 10
        result = f.seasonal_decompose(data, period=2)
        assert len(result.trend) == len(data)
        assert len(result.seasonal) == len(data)
        assert len(result.residual) == len(data)
        assert result.period == 2

    def test_trend_extraction(self):
        f = Forecaster()
        # Linear trend + seasonal
        data = [float(i + (i % 3)) for i in range(30)]
        result = f.seasonal_decompose(data, period=3)
        # Trend should generally increase
        assert result.trend[-1] > result.trend[0]

    def test_seasonal_pattern(self):
        f = Forecaster()
        # Pure seasonal pattern, no trend
        seasonal_pattern = [10.0, 20.0, 15.0]
        data = seasonal_pattern * 10
        result = f.seasonal_decompose(data, period=3)
        # Seasonal component should capture the pattern
        assert len(set(result.seasonal[:3])) > 1

    def test_insufficient_data(self):
        f = Forecaster()
        data = [1.0, 2.0]
        result = f.seasonal_decompose(data, period=5)
        assert len(result.trend) == 2
        assert all(t == 0.0 for t in result.trend)

    def test_residual_calculation(self):
        f = Forecaster()
        data = [float(i) for i in range(20)]
        result = f.seasonal_decompose(data, period=4)
        # Residual = data - trend - seasonal
        for i in range(len(data)):
            reconstructed = result.trend[i] + result.seasonal[i] + result.residual[i]
            assert abs(reconstructed - data[i]) < 1e-3

    def test_period_parameter(self):
        f = Forecaster()
        data = list(range(50))
        r1 = f.seasonal_decompose(data, period=3)
        r2 = f.seasonal_decompose(data, period=7)
        assert r1.period == 3
        assert r2.period == 7
        # Different periods should yield different seasonals
        assert r1.seasonal != r2.seasonal

    def test_empty_data(self):
        f = Forecaster()
        result = f.seasonal_decompose([], period=3)
        assert result.trend == []
        assert result.seasonal == []
        assert result.residual == []


class TestConfidenceIntervals:
    def test_basic_confidence_forecast(self):
        f = Forecaster()
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = f.forecast_with_confidence(data, steps=3, confidence=0.95)
        assert len(result.predictions) == 3
        assert len(result.lower_bound) == 3
        assert len(result.upper_bound) == 3
        assert result.confidence_level == 0.95

    def test_bounds_ordering(self):
        f = Forecaster()
        data = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        result = f.forecast_with_confidence(data, steps=5)
        for i in range(5):
            assert result.lower_bound[i] <= result.predictions[i]
            assert result.predictions[i] <= result.upper_bound[i]

    def test_confidence_level_affects_width(self):
        f = Forecaster()
        data = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0]
        r90 = f.forecast_with_confidence(data, steps=3, confidence=0.90)
        r99 = f.forecast_with_confidence(data, steps=3, confidence=0.99)
        # 99% should have wider intervals
        width_90 = r90.upper_bound[0] - r90.lower_bound[0]
        width_99 = r99.upper_bound[0] - r99.lower_bound[0]
        assert width_99 > width_90

    def test_empty_data(self):
        f = Forecaster()
        result = f.forecast_with_confidence([], steps=3)
        assert result.predictions == [0.0, 0.0, 0.0]
        assert result.lower_bound == [0.0, 0.0, 0.0]
        assert result.upper_bound == [0.0, 0.0, 0.0]

    def test_single_step_forecast(self):
        f = Forecaster()
        data = [10.0, 20.0, 30.0]
        result = f.forecast_with_confidence(data, steps=1)
        assert len(result.predictions) == 1

    def test_predictions_match_ensemble(self):
        f = Forecaster()
        data = [5.0, 15.0, 25.0, 35.0]
        conf_result = f.forecast_with_confidence(data, steps=3)
        ens_result = f.ensemble(data, horizon=3)
        assert conf_result.predictions == ens_result.predictions


class TestMultistepForecast:
    def test_basic_multistep(self):
        f = Forecaster()
        data = [10.0, 20.0, 30.0, 40.0]
        result = f.multistep_forecast(data, steps=3, method="ensemble")
        assert len(result) == 3
        assert all(isinstance(p, float) for p in result)

    def test_different_methods(self):
        f = Forecaster()
        data = [5.0, 10.0, 15.0, 20.0, 25.0]
        methods = ["moving_average", "exponential_smoothing", "linear_trend", "ensemble"]
        for method in methods:
            result = f.multistep_forecast(data, steps=3, method=method)
            assert len(result) == 3

    def test_invalid_method(self):
        f = Forecaster()
        data = [1.0, 2.0, 3.0]
        try:
            f.multistep_forecast(data, steps=3, method="invalid")
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "Unknown method" in str(e)

    def test_zero_steps(self):
        f = Forecaster()
        data = [1.0, 2.0, 3.0]
        result = f.multistep_forecast(data, steps=0)
        assert result == []

    def test_empty_data(self):
        f = Forecaster()
        result = f.multistep_forecast([], steps=3)
        assert result == []

    def test_iterative_extension(self):
        f = Forecaster()
        # Linear trend should produce consistent predictions
        data = [10.0, 20.0, 30.0, 40.0]
        result = f.multistep_forecast(data, steps=3, method="linear_trend")
        # Each prediction should build on previous
        assert len(result) == 3


class TestCrossValidation:
    def test_basic_cross_validation(self):
        f = Forecaster()
        data = [float(i) for i in range(30)]
        result = f.cross_validate(data, n_splits=3)
        assert len(result.scores) > 0
        assert result.mean_score >= 0.0
        assert result.std_score >= 0.0
        assert result.method == "cross_validate"

    def test_scores_length(self):
        f = Forecaster()
        data = [float(i * 2) for i in range(40)]
        result = f.cross_validate(data, n_splits=4)
        # Should have up to n_splits scores
        assert len(result.scores) <= 4
        assert len(result.scores) > 0

    def test_mean_is_average(self):
        f = Forecaster()
        data = list(range(50))
        result = f.cross_validate(data, n_splits=3)
        if result.scores:
            computed_mean = sum(result.scores) / len(result.scores)
            assert abs(computed_mean - result.mean_score) < 1e-3

    def test_insufficient_data(self):
        f = Forecaster()
        data = [1.0, 2.0]
        result = f.cross_validate(data, n_splits=3)
        assert result.scores == []
        assert result.mean_score == 0.0

    def test_empty_data(self):
        f = Forecaster()
        result = f.cross_validate([], n_splits=3)
        assert result.scores == []
        assert result.mean_score == 0.0
        assert result.std_score == 0.0

    def test_expanding_window(self):
        f = Forecaster()
        # Long series to ensure multiple splits
        data = [float(i % 10) for i in range(100)]
        result = f.cross_validate(data, n_splits=5)
        # Should produce multiple scores
        assert len(result.scores) >= 3
