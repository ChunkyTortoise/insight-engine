"""Time series forecasting: moving average, exponential smoothing, linear trend, ensemble."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ForecastResult:
    """Result of a single forecasting method."""

    method: str
    predictions: list[float]
    mae: float  # mean absolute error on training data
    rmse: float
    mape: float  # mean absolute percentage error


@dataclass
class ForecastComparison:
    """Comparison of multiple forecasting methods."""

    results: dict[str, ForecastResult]
    best_method: str
    best_mae: float


@dataclass
class DecompositionResult:
    """Result of seasonal decomposition."""

    trend: list[float]
    seasonal: list[float]
    residual: list[float]
    period: int


@dataclass
class ConfidenceForecast:
    """Forecast with confidence intervals."""

    predictions: list[float]
    lower_bound: list[float]
    upper_bound: list[float]
    confidence_level: float


@dataclass
class CrossValResult:
    """Result of time series cross-validation."""

    scores: list[float]
    mean_score: float
    std_score: float
    method: str


def _compute_mae(actual: list[float], predicted: list[float]) -> float:
    """Compute mean absolute error."""
    if not actual:
        return 0.0
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def _compute_rmse(actual: list[float], predicted: list[float]) -> float:
    """Compute root mean squared error."""
    if not actual:
        return 0.0
    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)
    return math.sqrt(mse)


def _compute_mape(actual: list[float], predicted: list[float]) -> float:
    """Compute mean absolute percentage error, skipping zero actuals."""
    pairs = [(a, p) for a, p in zip(actual, predicted) if a != 0.0]
    if not pairs:
        return 0.0
    return sum(abs((a - p) / a) for a, p in pairs) / len(pairs) * 100


class Forecaster:
    """Time series forecasting with multiple strategies.

    All methods operate on a simple list[float] time series.
    No external forecasting libraries (no statsmodels/prophet).
    """

    def moving_average(self, data: list[float], window: int = 3, horizon: int = 5) -> ForecastResult:
        """Forecast using simple moving average.

        Uses the last `window` values to predict each future step.
        Training MAE is computed using one-step-ahead prediction on historical data.
        """
        if not data:
            return ForecastResult(
                method="moving_average",
                predictions=[0.0] * horizon,
                mae=0.0,
                rmse=0.0,
                mape=0.0,
            )

        # Clamp window to data length
        w = min(window, len(data))

        # One-step-ahead predictions on training data
        actual: list[float] = []
        predicted: list[float] = []
        for i in range(w, len(data)):
            avg = sum(data[i - w : i]) / w
            predicted.append(avg)
            actual.append(data[i])

        mae = _compute_mae(actual, predicted)
        rmse = _compute_rmse(actual, predicted)
        mape = _compute_mape(actual, predicted)

        # Future predictions: iteratively extend
        extended = list(data)
        future: list[float] = []
        for _ in range(horizon):
            avg = sum(extended[-w:]) / w
            future.append(round(avg, 6))
            extended.append(avg)

        return ForecastResult(
            method="moving_average",
            predictions=future,
            mae=round(mae, 6),
            rmse=round(rmse, 6),
            mape=round(mape, 6),
        )

    def exponential_smoothing(self, data: list[float], alpha: float = 0.3, horizon: int = 5) -> ForecastResult:
        """Simple exponential smoothing (SES).

        alpha: smoothing factor (0 < alpha < 1). Higher = more weight on recent.
        """
        if not data:
            return ForecastResult(
                method="exponential_smoothing",
                predictions=[0.0] * horizon,
                mae=0.0,
                rmse=0.0,
                mape=0.0,
            )

        # Build smoothed series
        smoothed = [data[0]]
        for i in range(1, len(data)):
            s = alpha * data[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(s)

        # One-step-ahead: smoothed[i] predicts data[i+1]
        actual: list[float] = []
        predicted: list[float] = []
        for i in range(len(data) - 1):
            predicted.append(smoothed[i])
            actual.append(data[i + 1])

        mae = _compute_mae(actual, predicted)
        rmse = _compute_rmse(actual, predicted)
        mape = _compute_mape(actual, predicted)

        # Future: SES flat forecast = last smoothed value
        last_level = smoothed[-1]
        future = [round(last_level, 6)] * horizon

        return ForecastResult(
            method="exponential_smoothing",
            predictions=future,
            mae=round(mae, 6),
            rmse=round(rmse, 6),
            mape=round(mape, 6),
        )

    def linear_trend(self, data: list[float], horizon: int = 5) -> ForecastResult:
        """Forecast using linear regression trend line.

        Fits y = a + b*x using least squares, then extrapolates.
        """
        if not data:
            return ForecastResult(
                method="linear_trend",
                predictions=[0.0] * horizon,
                mae=0.0,
                rmse=0.0,
                mape=0.0,
            )

        n = len(data)
        x_vals = list(range(n))

        # Least squares: y = a + b*x
        x_mean = sum(x_vals) / n
        y_mean = sum(data) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, data))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)

        if denominator == 0:
            b = 0.0
        else:
            b = numerator / denominator
        a = y_mean - b * x_mean

        # Fitted values on training data
        fitted = [a + b * x for x in x_vals]

        mae = _compute_mae(data, fitted)
        rmse = _compute_rmse(data, fitted)
        mape = _compute_mape(data, fitted)

        # Future predictions
        future = [round(a + b * (n + i), 6) for i in range(horizon)]

        return ForecastResult(
            method="linear_trend",
            predictions=future,
            mae=round(mae, 6),
            rmse=round(rmse, 6),
            mape=round(mape, 6),
        )

    def ensemble(self, data: list[float], horizon: int = 5) -> ForecastResult:
        """Average of moving_average, exponential_smoothing, and linear_trend."""
        ma = self.moving_average(data, horizon=horizon)
        es = self.exponential_smoothing(data, horizon=horizon)
        lt = self.linear_trend(data, horizon=horizon)

        # Average predictions
        predictions = [
            round((ma.predictions[i] + es.predictions[i] + lt.predictions[i]) / 3, 6) for i in range(horizon)
        ]

        # Average error metrics
        mae = round((ma.mae + es.mae + lt.mae) / 3, 6)
        rmse = round((ma.rmse + es.rmse + lt.rmse) / 3, 6)
        mape = round((ma.mape + es.mape + lt.mape) / 3, 6)

        return ForecastResult(
            method="ensemble",
            predictions=predictions,
            mae=mae,
            rmse=rmse,
            mape=mape,
        )

    def compare_forecasts(self, data: list[float], horizon: int = 5) -> ForecastComparison:
        """Run all methods and compare. Returns best by MAE."""
        results: dict[str, ForecastResult] = {}
        results["moving_average"] = self.moving_average(data, horizon=horizon)
        results["exponential_smoothing"] = self.exponential_smoothing(data, horizon=horizon)
        results["linear_trend"] = self.linear_trend(data, horizon=horizon)
        results["ensemble"] = self.ensemble(data, horizon=horizon)

        best_method = min(results, key=lambda k: results[k].mae)
        best_mae = results[best_method].mae

        return ForecastComparison(
            results=results,
            best_method=best_method,
            best_mae=best_mae,
        )

    def seasonal_decompose(self, data: list[float], period: int) -> DecompositionResult:
        """Decompose time series into trend, seasonal, and residual components.

        Uses moving average for trend extraction, then extracts seasonal pattern
        by averaging values at each position in the period.
        """
        if not data or len(data) < period:
            # Return zeros for insufficient data
            n = len(data) if data else 0
            return DecompositionResult(
                trend=[0.0] * n,
                seasonal=[0.0] * n,
                residual=[0.0] * n,
                period=period,
            )

        n = len(data)
        trend: list[float] = []

        # Centered moving average for trend
        half_window = period // 2
        for i in range(n):
            if i < half_window or i >= n - half_window:
                # Use available data at edges
                start = max(0, i - half_window)
                end = min(n, i + half_window + 1)
                avg = sum(data[start:end]) / len(data[start:end])
                trend.append(avg)
            else:
                avg = sum(data[i - half_window : i + half_window + 1]) / period
                trend.append(avg)

        # Detrended series
        detrended = [data[i] - trend[i] for i in range(n)]

        # Extract seasonal pattern by averaging positions within period
        seasonal_avg: dict[int, list[float]] = {i: [] for i in range(period)}
        for i, val in enumerate(detrended):
            pos = i % period
            seasonal_avg[pos].append(val)

        # Average seasonal component per position
        seasonal_pattern = [
            sum(seasonal_avg[i]) / len(seasonal_avg[i]) if seasonal_avg[i] else 0.0 for i in range(period)
        ]

        # Expand seasonal pattern to full series length
        seasonal = [seasonal_pattern[i % period] for i in range(n)]

        # Residual = data - trend - seasonal
        residual = [round(data[i] - trend[i] - seasonal[i], 6) for i in range(n)]
        trend = [round(t, 6) for t in trend]
        seasonal = [round(s, 6) for s in seasonal]

        return DecompositionResult(
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            period=period,
        )

    def forecast_with_confidence(
        self,
        data: list[float],
        steps: int = 1,
        confidence: float = 0.95,
    ) -> ConfidenceForecast:
        """Forecast with confidence intervals using ensemble method.

        Confidence intervals are computed using residual standard deviation
        multiplied by the appropriate z-score for the confidence level.
        """
        if not data:
            return ConfidenceForecast(
                predictions=[0.0] * steps,
                lower_bound=[0.0] * steps,
                upper_bound=[0.0] * steps,
                confidence_level=confidence,
            )

        # Get ensemble forecast
        ensemble_result = self.ensemble(data, horizon=steps)
        predictions = ensemble_result.predictions

        # Compute residual std from training predictions
        # Use moving average one-step-ahead errors as proxy
        window = min(3, len(data))
        actual: list[float] = []
        predicted: list[float] = []
        for i in range(window, len(data)):
            avg = sum(data[i - window : i]) / window
            predicted.append(avg)
            actual.append(data[i])

        if not actual:
            # Not enough data for residuals, use default std
            residual_std = 1.0
        else:
            errors = [a - p for a, p in zip(actual, predicted)]
            residual_std = math.sqrt(sum(e**2 for e in errors) / len(errors))

        # Z-score for confidence level (approximate)
        # 0.90 -> 1.645, 0.95 -> 1.96, 0.99 -> 2.576
        if confidence >= 0.99:
            z_score = 2.576
        elif confidence >= 0.95:
            z_score = 1.96
        else:
            z_score = 1.645

        margin = z_score * residual_std

        lower_bound = [round(p - margin, 6) for p in predictions]
        upper_bound = [round(p + margin, 6) for p in predictions]

        return ConfidenceForecast(
            predictions=predictions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence,
        )

    def multistep_forecast(
        self,
        data: list[float],
        steps: int,
        method: str = "ensemble",
    ) -> list[float]:
        """Multi-step ahead forecast using iterative prediction.

        At each step, appends the prediction to the data and forecasts again.
        Supports methods: moving_average, exponential_smoothing, linear_trend, ensemble.
        """
        if not data or steps <= 0:
            return []

        method_map = {
            "moving_average": self.moving_average,
            "exponential_smoothing": self.exponential_smoothing,
            "linear_trend": self.linear_trend,
            "ensemble": self.ensemble,
        }

        if method not in method_map:
            raise ValueError(f"Unknown method '{method}'. Choose from: {list(method_map.keys())}")

        forecast_fn = method_map[method]
        extended = list(data)
        predictions: list[float] = []

        for _ in range(steps):
            result = forecast_fn(extended, horizon=1)
            next_val = result.predictions[0]
            predictions.append(next_val)
            extended.append(next_val)

        return predictions

    def cross_validate(self, data: list[float], n_splits: int = 3) -> CrossValResult:
        """Time series cross-validation using expanding window.

        Trains on first k splits, tests on k+1 split. Reports MAE scores.
        Uses ensemble method for forecasting.
        """
        if not data or n_splits < 2:
            return CrossValResult(
                scores=[],
                mean_score=0.0,
                std_score=0.0,
                method="cross_validate",
            )

        n = len(data)
        split_size = n // (n_splits + 1)

        if split_size < 3:
            # Not enough data
            return CrossValResult(
                scores=[],
                mean_score=0.0,
                std_score=0.0,
                method="cross_validate",
            )

        scores: list[float] = []
        for i in range(1, n_splits + 1):
            train_end = split_size * i
            test_start = train_end
            test_end = min(test_start + split_size, n)

            if test_start >= n:
                break

            train = data[:train_end]
            test = data[test_start:test_end]

            if not test:
                break

            # Forecast test length
            forecast_result = self.ensemble(train, horizon=len(test))
            mae = _compute_mae(test, forecast_result.predictions[: len(test)])
            scores.append(mae)

        if not scores:
            return CrossValResult(
                scores=[],
                mean_score=0.0,
                std_score=0.0,
                method="cross_validate",
            )

        mean_score = sum(scores) / len(scores)
        if len(scores) > 1:
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_score = math.sqrt(variance)
        else:
            std_score = 0.0

        return CrossValResult(
            scores=[round(s, 6) for s in scores],
            mean_score=round(mean_score, 6),
            std_score=round(std_score, 6),
            method="cross_validate",
        )
