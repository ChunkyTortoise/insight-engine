"""Marketing Funnel Attribution: first-touch, last-touch, linear, time-decay models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class AttributionModel(str, Enum):
    FIRST_TOUCH = "first_touch"
    LAST_TOUCH = "last_touch"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"


@dataclass
class AttributionResult:
    model: AttributionModel
    channel_credits: dict[str, float]
    total_conversions: int
    summary: pd.DataFrame


def _validate_touchpoints(df: pd.DataFrame) -> None:
    """Validate touchpoint DataFrame has required columns."""
    required = {"user_id", "channel", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def first_touch(touchpoints: pd.DataFrame, conversions: set[str]) -> AttributionResult:
    """Credit goes entirely to the first channel the user interacted with."""
    _validate_touchpoints(touchpoints)
    sorted_tp = touchpoints.sort_values("timestamp")
    first_touches = sorted_tp.groupby("user_id").first().reset_index()
    converted = first_touches[first_touches["user_id"].isin(conversions)]
    credits = converted["channel"].value_counts().to_dict()
    total = len(converted)

    # Normalize to percentages
    credit_pct = {k: round(v / max(total, 1) * 100, 2) for k, v in credits.items()}

    summary = pd.DataFrame(
        [{"channel": k, "conversions": v, "credit_pct": credit_pct[k]} for k, v in credits.items()]
    )

    return AttributionResult(
        model=AttributionModel.FIRST_TOUCH,
        channel_credits=credit_pct,
        total_conversions=total,
        summary=summary,
    )


def last_touch(touchpoints: pd.DataFrame, conversions: set[str]) -> AttributionResult:
    """Credit goes entirely to the last channel before conversion."""
    _validate_touchpoints(touchpoints)
    sorted_tp = touchpoints.sort_values("timestamp")
    last_touches = sorted_tp.groupby("user_id").last().reset_index()
    converted = last_touches[last_touches["user_id"].isin(conversions)]
    credits = converted["channel"].value_counts().to_dict()
    total = len(converted)

    credit_pct = {k: round(v / max(total, 1) * 100, 2) for k, v in credits.items()}

    summary = pd.DataFrame(
        [{"channel": k, "conversions": v, "credit_pct": credit_pct[k]} for k, v in credits.items()]
    )

    return AttributionResult(
        model=AttributionModel.LAST_TOUCH,
        channel_credits=credit_pct,
        total_conversions=total,
        summary=summary,
    )


def linear(touchpoints: pd.DataFrame, conversions: set[str]) -> AttributionResult:
    """Credit is split equally across all channels in the user's journey."""
    _validate_touchpoints(touchpoints)
    converted_tp = touchpoints[touchpoints["user_id"].isin(conversions)]
    credits: dict[str, float] = {}

    for user_id in conversions:
        user_touches = converted_tp[converted_tp["user_id"] == user_id]
        if len(user_touches) == 0:
            continue
        share = 1.0 / len(user_touches)
        for channel in user_touches["channel"]:
            credits[channel] = credits.get(channel, 0.0) + share

    total = len(conversions)
    credit_pct = {k: round(v / max(total, 1) * 100, 2) for k, v in credits.items()}

    summary = pd.DataFrame(
        [{"channel": k, "credit": round(v, 2), "credit_pct": credit_pct[k]} for k, v in credits.items()]
    )

    return AttributionResult(
        model=AttributionModel.LINEAR,
        channel_credits=credit_pct,
        total_conversions=total,
        summary=summary,
    )


def time_decay(
    touchpoints: pd.DataFrame, conversions: set[str], half_life_days: float = 7.0
) -> AttributionResult:
    """Credit decays exponentially — recent touchpoints get more credit."""
    _validate_touchpoints(touchpoints)
    converted_tp = touchpoints[touchpoints["user_id"].isin(conversions)].copy()
    converted_tp["timestamp"] = pd.to_datetime(converted_tp["timestamp"])

    credits: dict[str, float] = {}

    for user_id in conversions:
        user_touches = converted_tp[converted_tp["user_id"] == user_id].sort_values("timestamp")
        if len(user_touches) == 0:
            continue

        max_time = user_touches["timestamp"].max()
        days_before = (max_time - user_touches["timestamp"]).dt.total_seconds() / 86400

        # Exponential decay weights
        decay_rate = np.log(2) / half_life_days
        weights = np.exp(-decay_rate * days_before.values)
        total_weight = weights.sum()

        if total_weight > 0:
            normalized = weights / total_weight
            for channel, weight in zip(user_touches["channel"], normalized):
                credits[channel] = credits.get(channel, 0.0) + weight

    total = len(conversions)
    credit_pct = {k: round(v / max(total, 1) * 100, 2) for k, v in credits.items()}

    summary = pd.DataFrame(
        [{"channel": k, "credit": round(v, 2), "credit_pct": credit_pct[k]} for k, v in credits.items()]
    )

    return AttributionResult(
        model=AttributionModel.TIME_DECAY,
        channel_credits=credit_pct,
        total_conversions=total,
        summary=summary,
    )


def run_all_models(
    touchpoints: pd.DataFrame, conversions: set[str]
) -> dict[AttributionModel, AttributionResult]:
    """Run all four attribution models and return results."""
    return {
        AttributionModel.FIRST_TOUCH: first_touch(touchpoints, conversions),
        AttributionModel.LAST_TOUCH: last_touch(touchpoints, conversions),
        AttributionModel.LINEAR: linear(touchpoints, conversions),
        AttributionModel.TIME_DECAY: time_decay(touchpoints, conversions),
    }
