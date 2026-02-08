"""Tests for the feature engineering module."""

import pandas as pd
import pytest

from insight_engine.feature_lab import FeatureLab, FeatureResult


class TestScaling:
    def test_standard_scaling(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        lab = FeatureLab()
        result = lab.scale(df, ["a"], method="standard")
        assert isinstance(result, FeatureResult)
        scaled = result.data["a_scaled"]
        assert abs(scaled.mean()) < 1e-6
        assert abs(scaled.std(ddof=0) - 1.0) < 0.1

    def test_minmax_scaling(self):
        df = pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0, 50.0]})
        lab = FeatureLab()
        result = lab.scale(df, ["a"], method="minmax")
        scaled = result.data["a_scaled"]
        assert scaled.min() >= -1e-6
        assert scaled.max() <= 1.0 + 1e-6

    def test_robust_scaling(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 1000.0]})
        lab = FeatureLab()
        result = lab.scale(df, ["a"], method="robust")
        assert "a_scaled" in result.data.columns

    def test_invalid_method_raises(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        lab = FeatureLab()
        with pytest.raises(ValueError, match="Unknown scaling method"):
            lab.scale(df, ["a"], method="bogus")


class TestEncoding:
    def test_onehot_creates_dummies(self):
        df = pd.DataFrame({"color": ["red", "blue", "green", "red", "blue"]})
        lab = FeatureLab()
        result = lab.encode_onehot(df, ["color"])
        assert len(result.new_columns) == 3

    def test_ordinal_preserves_order(self):
        df = pd.DataFrame({"size": ["S", "M", "L", "XL", "M"]})
        lab = FeatureLab()
        result = lab.encode_ordinal(df, "size", order=["S", "M", "L", "XL"])
        vals = result.data["size_ordinal"].tolist()
        assert vals == [0, 1, 2, 3, 1]

    def test_onehot_drops_original(self):
        df = pd.DataFrame({"color": ["red", "blue", "green"]})
        lab = FeatureLab()
        result = lab.encode_onehot(df, ["color"])
        assert "color" not in result.data.columns


class TestPolynomial:
    def test_degree_2(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        lab = FeatureLab()
        result = lab.polynomial_features(df, ["x"], degree=2)
        assert "x_pow2" in result.data.columns
        assert result.data["x_pow2"].tolist() == [1.0, 4.0, 9.0]

    def test_correct_column_count(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        lab = FeatureLab()
        result = lab.polynomial_features(df, ["a", "b"], degree=2)
        # 2 squared cols + 1 cross term = 3 new columns
        assert len(result.new_columns) == 3


class TestInteraction:
    def test_pairwise_products(self):
        df = pd.DataFrame({"a": [2.0, 3.0], "b": [4.0, 5.0], "c": [6.0, 7.0]})
        lab = FeatureLab()
        result = lab.interaction_terms(df, ["a", "b", "c"])
        assert result.data["a_x_b"].tolist() == [8.0, 15.0]
        assert result.data["a_x_c"].tolist() == [12.0, 21.0]
        assert result.data["b_x_c"].tolist() == [24.0, 35.0]

    def test_two_columns(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        lab = FeatureLab()
        result = lab.interaction_terms(df, ["x", "y"])
        assert len(result.new_columns) == 1


class TestAutoEngineer:
    def test_mixed_dataframe(self):
        df = pd.DataFrame(
            {
                "price": [10.0, 20.0, 30.0],
                "quantity": [1, 2, 3],
                "category": ["A", "B", "A"],
            }
        )
        lab = FeatureLab()
        result = lab.auto_engineer(df)
        assert isinstance(result, FeatureResult)
        assert result.method == "auto"
        assert len(result.new_columns) > 0

    def test_adds_columns(self):
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
                "cat": ["x", "y", "z"],
            }
        )
        lab = FeatureLab()
        result = lab.auto_engineer(df)
        assert len(result.data.columns) > len(df.columns)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        lab = FeatureLab()
        result = lab.auto_engineer(df)
        assert len(result.new_columns) == 0
