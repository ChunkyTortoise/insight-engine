"""Tests for the marketing attribution module."""

import pandas as pd
import pytest

from insight_engine.attribution import (
    AttributionModel,
    first_touch,
    last_touch,
    linear,
    run_all_models,
    time_decay,
)


@pytest.fixture
def touchpoints():
    return pd.DataFrame(
        {
            "user_id": ["U1", "U1", "U1", "U2", "U2", "U3", "U3", "U3", "U3"],
            "channel": ["Google", "Email", "Facebook", "Google", "Google", "LinkedIn", "Email", "Google", "Facebook"],
            "timestamp": [
                "2024-01-01 10:00",
                "2024-01-05 10:00",
                "2024-01-10 10:00",
                "2024-01-01 10:00",
                "2024-01-03 10:00",
                "2024-01-01 10:00",
                "2024-01-07 10:00",
                "2024-01-14 10:00",
                "2024-01-20 10:00",
            ],
        }
    )


@pytest.fixture
def conversions():
    return {"U1", "U3"}


class TestFirstTouch:
    def test_basic(self, touchpoints, conversions):
        result = first_touch(touchpoints, conversions)
        assert result.model == AttributionModel.FIRST_TOUCH
        assert result.total_conversions == 2
        assert "Google" in result.channel_credits or "LinkedIn" in result.channel_credits

    def test_empty_conversions(self, touchpoints):
        result = first_touch(touchpoints, set())
        assert result.total_conversions == 0


class TestLastTouch:
    def test_basic(self, touchpoints, conversions):
        result = last_touch(touchpoints, conversions)
        assert result.model == AttributionModel.LAST_TOUCH
        assert result.total_conversions == 2
        # U1's last touch is Facebook, U3's is Facebook
        assert "Facebook" in result.channel_credits


class TestLinear:
    def test_basic(self, touchpoints, conversions):
        result = linear(touchpoints, conversions)
        assert result.model == AttributionModel.LINEAR
        assert result.total_conversions == 2
        # Credits should be distributed across channels
        assert len(result.channel_credits) > 0

    def test_single_touch(self):
        tp = pd.DataFrame(
            {
                "user_id": ["U1"],
                "channel": ["Google"],
                "timestamp": ["2024-01-01 10:00"],
            }
        )
        result = linear(tp, {"U1"})
        assert result.channel_credits["Google"] == 100.0


class TestTimeDecay:
    def test_basic(self, touchpoints, conversions):
        result = time_decay(touchpoints, conversions)
        assert result.model == AttributionModel.TIME_DECAY
        assert result.total_conversions == 2

    def test_recent_touch_gets_more_credit(self):
        tp = pd.DataFrame(
            {
                "user_id": ["U1", "U1"],
                "channel": ["Old Channel", "New Channel"],
                "timestamp": ["2024-01-01 10:00", "2024-01-30 10:00"],
            }
        )
        result = time_decay(tp, {"U1"}, half_life_days=7.0)
        # New Channel should get more credit due to recency
        assert result.channel_credits.get("New Channel", 0) > result.channel_credits.get("Old Channel", 0)


class TestRunAllModels:
    def test_returns_all_models(self, touchpoints, conversions):
        results = run_all_models(touchpoints, conversions)
        assert len(results) == 4
        assert all(m in results for m in AttributionModel)


class TestValidation:
    def test_missing_columns(self):
        bad_df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            first_touch(bad_df, {"U1"})
