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
