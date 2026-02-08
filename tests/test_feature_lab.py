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


# ============================================================
# NEW TESTS: Encoding, scaling, interactions, feature selection,
# datetime features, edge cases
# ============================================================


class TestOneHotEncodingAdvanced:
    """Advanced tests for one-hot encoding."""

    def test_binary_column(self):
        lab = FeatureLab()
        df = pd.DataFrame({"flag": ["yes", "no", "yes", "no"]})
        result = lab.encode_onehot(df, ["flag"])
        assert len(result.new_columns) == 2
        assert "flag" not in result.data.columns

    def test_single_category(self):
        lab = FeatureLab()
        df = pd.DataFrame({"cat": ["A", "A", "A"]})
        result = lab.encode_onehot(df, ["cat"])
        assert len(result.new_columns) == 1

    def test_multiple_columns(self):
        lab = FeatureLab()
        df = pd.DataFrame({"color": ["R", "G", "B"], "size": ["S", "M", "L"]})
        result = lab.encode_onehot(df, ["color", "size"])
        # Both original columns should be dropped
        assert "color" not in result.data.columns
        assert "size" not in result.data.columns
        assert len(result.new_columns) == 6  # 3 colors + 3 sizes

    def test_onehot_preserves_numeric_columns(self):
        lab = FeatureLab()
        df = pd.DataFrame({"cat": ["A", "B", "C"], "val": [1.0, 2.0, 3.0]})
        result = lab.encode_onehot(df, ["cat"])
        assert "val" in result.data.columns
        assert result.data["val"].tolist() == [1.0, 2.0, 3.0]


class TestLabelEncodingAdvanced:
    """Advanced tests for ordinal encoding."""

    def test_ordinal_with_missing_values(self):
        lab = FeatureLab()
        df = pd.DataFrame({"grade": ["A", "B", "C", "D"]})
        result = lab.encode_ordinal(df, "grade", order=["D", "C", "B", "A"])
        vals = result.data["grade_ordinal"].tolist()
        assert vals == [3, 2, 1, 0]

    def test_ordinal_unmapped_value_becomes_nan(self):
        lab = FeatureLab()
        df = pd.DataFrame({"level": ["low", "medium", "unknown"]})
        result = lab.encode_ordinal(df, "level", order=["low", "medium", "high"])
        # "unknown" is not in the order, should be NaN
        assert pd.isna(result.data["level_ordinal"].iloc[2])


class TestScalingAdvanced:
    """Advanced tests for standard, minmax, and robust scaling."""

    def test_standard_scaling_zero_mean(self):
        lab = FeatureLab()
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = lab.scale(df, ["x"], method="standard")
        assert abs(result.data["x_scaled"].mean()) < 1e-6

    def test_minmax_scaling_range_01(self):
        lab = FeatureLab()
        df = pd.DataFrame({"x": [0.0, 25.0, 50.0, 75.0, 100.0]})
        result = lab.scale(df, ["x"], method="minmax")
        assert result.data["x_scaled"].min() >= -1e-6
        assert result.data["x_scaled"].max() <= 1.0 + 1e-6

    def test_robust_scaling_handles_outliers(self):
        lab = FeatureLab()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 1000.0]})
        result = lab.scale(df, ["x"], method="robust")
        # The median-based scaler should not be as affected by the outlier
        scaled = result.data["x_scaled"]
        assert abs(scaled.iloc[2]) < abs(scaled.iloc[4])

    def test_scale_multiple_columns(self):
        lab = FeatureLab()
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [100.0, 200.0, 300.0]})
        result = lab.scale(df, ["a", "b"], method="standard")
        assert "a_scaled" in result.data.columns
        assert "b_scaled" in result.data.columns
        assert len(result.new_columns) == 2

    def test_scale_preserves_original_columns(self):
        lab = FeatureLab()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = lab.scale(df, ["x"], method="standard")
        assert "x" in result.data.columns
        assert result.data["x"].tolist() == [1.0, 2.0, 3.0]


class TestInteractionAdvanced:
    """Advanced tests for interaction terms."""

    def test_interaction_with_zeros(self):
        lab = FeatureLab()
        df = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [3.0, 4.0, 5.0]})
        result = lab.interaction_terms(df, ["a", "b"])
        assert result.data["a_x_b"].tolist() == [0.0, 4.0, 10.0]

    def test_interaction_four_columns(self):
        lab = FeatureLab()
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]})
        result = lab.interaction_terms(df, ["a", "b", "c", "d"])
        # C(4,2) = 6 pairs
        assert len(result.new_columns) == 6

    def test_interaction_single_column_no_output(self):
        lab = FeatureLab()
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = lab.interaction_terms(df, ["a"])
        # No pairs possible with 1 column
        assert len(result.new_columns) == 0


class TestFeatureSelectionAdvanced:
    """Advanced tests for feature selection methods."""

    def test_variance_keeps_high_variance_features(self):
        lab = FeatureLab()
        df = pd.DataFrame(
            {
                "constant": [5, 5, 5, 5, 5],
                "varying": [1, 10, 100, 1000, 10000],
                "target": [0, 1, 0, 1, 0],
            }
        )
        result = lab.select_features(df, target="target", method="variance", threshold=1.0)
        assert "varying" in result.selected_columns
        assert "constant" in result.dropped_columns

    def test_correlation_drops_redundant_features(self):
        lab = FeatureLab()
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "a_copy": [1, 2, 3, 4, 5],  # perfectly correlated with a
                "unrelated": [5, 3, 1, 4, 2],
                "target": [0, 1, 0, 1, 0],
            }
        )
        result = lab.select_features(df, target="target", method="correlation", threshold=0.9)
        # At least one of the correlated pair should be dropped
        assert len(result.dropped_columns) >= 1

    def test_selection_scores_are_numeric(self):
        lab = FeatureLab()
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 3, 2, 1], "target": [0, 1, 0, 1]})
        result = lab.select_features(df, target="target", method="variance", threshold=0.01)
        for col, score in result.scores.items():
            assert isinstance(score, float)


class TestDatetimeFeaturesAdvanced:
    """Advanced tests for datetime feature extraction."""

    def test_weekday_values(self):
        lab = FeatureLab()
        # Monday 2023-01-02 through Friday 2023-01-06
        df = pd.DataFrame({"date": pd.date_range("2023-01-02", periods=5)})
        result = lab.extract_time_features(df, column="date")
        dow = result.data["date_day_of_week"].tolist()
        assert dow == [0, 1, 2, 3, 4]  # Mon-Fri

    def test_year_extraction(self):
        lab = FeatureLab()
        df = pd.DataFrame({"date": pd.to_datetime(["2020-06-15", "2021-06-15", "2022-06-15"])})
        result = lab.extract_time_features(df, column="date")
        years = result.data["date_year"].tolist()
        assert years == [2020, 2021, 2022]

    def test_no_hour_for_date_only(self):
        lab = FeatureLab()
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5)})
        result = lab.extract_time_features(df, column="date")
        # No hour info when all times are midnight
        assert "date_hour" not in result.new_columns

    def test_string_dates_converted(self):
        lab = FeatureLab()
        df = pd.DataFrame({"date": ["2023-03-15", "2023-06-15", "2023-09-15"]})
        result = lab.extract_time_features(df, column="date")
        quarters = result.data["date_quarter"].tolist()
        assert quarters == [1, 2, 3]


class TestEdgeCasesFeatureLab:
    """Edge cases for feature engineering."""

    def test_polynomial_degree_3(self):
        lab = FeatureLab()
        df = pd.DataFrame({"x": [2.0, 3.0]})
        result = lab.polynomial_features(df, ["x"], degree=3)
        assert "x_pow2" in result.data.columns
        assert "x_pow3" in result.data.columns
        assert result.data["x_pow3"].tolist() == [8.0, 27.0]

    def test_auto_engineer_numeric_only(self):
        lab = FeatureLab()
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = lab.auto_engineer(df)
        # Should have scaled columns and interaction terms
        assert "a_scaled" in result.data.columns
        assert "b_scaled" in result.data.columns
        assert "a_x_b" in result.data.columns

    def test_auto_engineer_categorical_only(self):
        lab = FeatureLab()
        df = pd.DataFrame({"color": ["R", "G", "B"], "size": ["S", "M", "L"]})
        result = lab.auto_engineer(df)
        # Should have one-hot encoded columns
        assert len(result.new_columns) > 0

    def test_bin_numeric_three_bins(self):
        lab = FeatureLab()
        df = pd.DataFrame({"x": list(range(30))})
        result = lab.bin_numeric(df, column="x", n_bins=3, method="equal_width")
        unique_bins = result.data["x_binned"].nunique()
        assert unique_bins <= 3

    def test_scale_method_in_result(self):
        lab = FeatureLab()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = lab.scale(df, ["x"], method="minmax")
        assert result.method == "scale_minmax"

    def test_description_in_result(self):
        lab = FeatureLab()
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = lab.interaction_terms(df, ["x", "y"])
        assert "interaction" in result.description.lower() or "1" in result.description
