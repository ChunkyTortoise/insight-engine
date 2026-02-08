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


class TestFeatureSelection:
    def test_variance_selection(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab, SelectionResult

        lab = FeatureLab()
        df = pd.DataFrame({"a": [1, 1, 1, 1], "b": [1, 2, 3, 4], "c": [5, 5, 5, 5], "target": [0, 1, 0, 1]})
        result = lab.select_features(df, target="target", method="variance", threshold=0.5)
        assert isinstance(result, SelectionResult)
        # Columns a and c have low variance, should be dropped
        assert "b" in result.selected_columns

    def test_mutual_info_selection(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": [5, 4, 3, 2, 1], "x3": [1, 1, 1, 1, 1], "y": [0, 0, 1, 1, 1]})
        result = lab.select_features(df, target="y", method="mutual_info", threshold=0.01)
        assert len(result.selected_columns) > 0
        assert "y" not in result.selected_columns

    def test_correlation_selection(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4], "c": [4, 3, 2, 1], "target": [0, 1, 0, 1]})
        result = lab.select_features(df, target="target", method="correlation", threshold=0.9)
        # a and b are perfectly correlated
        assert len(result.dropped_columns) > 0

    def test_invalid_method(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        try:
            lab.select_features(df, target="target", method="invalid")
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "Unknown method" in str(e)

    def test_no_numeric_columns(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"cat1": ["a", "b", "c"], "cat2": ["x", "y", "z"]})
        result = lab.select_features(df, target="cat1", method="variance")
        assert result.selected_columns == []

    def test_scores_returned(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8], "target": [0, 1, 0, 1]})
        result = lab.select_features(df, target="target", method="variance", threshold=0.1)
        assert isinstance(result.scores, dict)
        assert len(result.scores) > 0

    def test_dropped_columns(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"low_var": [1, 1, 1, 1], "high_var": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
        result = lab.select_features(df, target="target", method="variance", threshold=0.5)
        assert "low_var" in result.dropped_columns


class TestTimeFeatures:
    def test_basic_time_features(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=10)})
        result = lab.extract_time_features(df, column="date")
        assert "date_day_of_week" in result.new_columns
        assert "date_month" in result.new_columns
        assert "date_quarter" in result.new_columns
        assert "date_year" in result.new_columns
        assert "date_is_weekend" in result.new_columns

    def test_hour_extraction(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"timestamp": pd.date_range("2023-01-01 08:00", periods=5, freq="h")})
        result = lab.extract_time_features(df, column="timestamp")
        assert "timestamp_hour" in result.new_columns

    def test_weekend_detection(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        # Saturday and Sunday
        df = pd.DataFrame({"date": pd.to_datetime(["2023-01-07", "2023-01-08"])})
        result = lab.extract_time_features(df, column="date")
        weekend_col = "date_is_weekend"
        assert all(result.data[weekend_col] == 1)

    def test_missing_column(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = lab.extract_time_features(df, column="nonexistent")
        assert result.new_columns == []

    def test_month_extraction(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"date": pd.date_range("2023-03-01", periods=3)})
        result = lab.extract_time_features(df, column="date")
        assert all(result.data["date_month"] == 3)

    def test_quarter_extraction(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15"])})
        result = lab.extract_time_features(df, column="date")
        quarters = result.data["date_quarter"].tolist()
        assert quarters == [1, 2, 3, 4]


class TestBinning:
    def test_equal_width_binning(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        result = lab.bin_numeric(df, column="value", n_bins=5, method="equal_width")
        assert "value_binned" in result.new_columns
        assert result.data["value_binned"].nunique() <= 5

    def test_equal_frequency_binning(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"value": list(range(20))})
        result = lab.bin_numeric(df, column="value", n_bins=4, method="equal_frequency")
        assert "value_binned" in result.new_columns

    def test_invalid_binning_method(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        try:
            lab.bin_numeric(df, column="value", method="invalid")
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "Unknown method" in str(e)

    def test_missing_column_binning(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = lab.bin_numeric(df, column="nonexistent")
        assert result.new_columns == []

    def test_custom_bins(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"score": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
        result = lab.bin_numeric(df, column="score", n_bins=10)
        assert "score_binned" in result.new_columns

    def test_binned_values(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        result = lab.bin_numeric(df, column="x", n_bins=2, method="equal_width")
        # All values should be binned
        assert result.data["x_binned"].notna().all()


class TestTargetEncoding:
    def test_basic_target_encoding(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"category": ["A", "A", "B", "B"], "target": [1, 1, 0, 0]})
        result, mapping = lab.target_encode(df, column="category", target="target")
        assert "category_encoded" in result.new_columns
        assert isinstance(mapping, dict)

    def test_encoding_mapping(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"cat": ["X", "X", "Y", "Y"], "outcome": [10, 20, 30, 40]})
        result, mapping = lab.target_encode(df, column="cat", target="outcome")
        assert mapping["X"] == 15.0
        assert mapping["Y"] == 35.0

    def test_missing_values_filled(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"cat": ["A", "B", "C"], "target": [1, 2, 3]})
        result, mapping = lab.target_encode(df, column="cat", target="target")
        # All encoded values should be present
        assert result.data["cat_encoded"].notna().all()

    def test_missing_column(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"other": [1, 2, 3]})
        result, mapping = lab.target_encode(df, column="nonexistent", target="other")
        assert result.new_columns == []
        assert mapping == {}

    def test_missing_target(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"cat": ["A", "B"]})
        result, mapping = lab.target_encode(df, column="cat", target="nonexistent")
        assert result.new_columns == []

    def test_multiple_categories(self):
        import pandas as pd

        from insight_engine.feature_lab import FeatureLab

        lab = FeatureLab()
        df = pd.DataFrame({"group": ["G1", "G1", "G2", "G2", "G3", "G3"], "value": [5, 15, 25, 35, 45, 55]})
        result, mapping = lab.target_encode(df, column="group", target="value")
        assert len(mapping) == 3
        assert "group_encoded" in result.data.columns
