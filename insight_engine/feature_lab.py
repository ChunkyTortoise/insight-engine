"""Feature engineering: scaling, encoding, polynomial features, interaction terms."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


@dataclass
class FeatureResult:
    """Result of a feature engineering operation."""

    data: pd.DataFrame
    new_columns: list[str]
    method: str
    description: str


@dataclass
class SelectionResult:
    """Result of feature selection."""

    selected_columns: list[str]
    dropped_columns: list[str]
    scores: dict[str, float]


_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


class FeatureLab:
    """Feature engineering toolkit for tabular data."""

    def scale(self, df: pd.DataFrame, columns: list[str], method: str = "standard") -> FeatureResult:
        """Scale numeric columns. Methods: standard, minmax, robust."""
        if method not in _SCALERS:
            raise ValueError(f"Unknown scaling method '{method}'. Choose from: {list(_SCALERS.keys())}")

        scaler = _SCALERS[method]()
        result_df = df.copy()
        new_cols: list[str] = []

        for col in columns:
            new_name = f"{col}_scaled"
            values = df[[col]].values.astype(float)
            result_df[new_name] = scaler.fit_transform(values).flatten()
            new_cols.append(new_name)

        return FeatureResult(
            data=result_df,
            new_columns=new_cols,
            method=f"scale_{method}",
            description=f"Scaled {len(columns)} column(s) using {method} scaler",
        )

    def encode_onehot(self, df: pd.DataFrame, columns: list[str]) -> FeatureResult:
        """One-hot encode categorical columns."""
        result_df = df.copy()
        new_cols: list[str] = []

        for col in columns:
            dummies = pd.get_dummies(result_df[col], prefix=col, dtype=int)
            dummy_names = dummies.columns.tolist()
            result_df = pd.concat([result_df, dummies], axis=1)
            result_df = result_df.drop(columns=[col])
            new_cols.extend(dummy_names)

        return FeatureResult(
            data=result_df,
            new_columns=new_cols,
            method="onehot",
            description=f"One-hot encoded {len(columns)} column(s)",
        )

    def encode_ordinal(self, df: pd.DataFrame, column: str, order: list[str]) -> FeatureResult:
        """Ordinal encode a column using specified order."""
        result_df = df.copy()
        mapping = {val: idx for idx, val in enumerate(order)}
        new_name = f"{column}_ordinal"
        result_df[new_name] = result_df[column].map(mapping)

        return FeatureResult(
            data=result_df,
            new_columns=[new_name],
            method="ordinal",
            description=f"Ordinal encoded '{column}' with {len(order)} levels",
        )

    def polynomial_features(self, df: pd.DataFrame, columns: list[str], degree: int = 2) -> FeatureResult:
        """Generate polynomial features (degree 2 by default)."""
        result_df = df.copy()
        new_cols: list[str] = []

        for col in columns:
            for d in range(2, degree + 1):
                new_name = f"{col}_pow{d}"
                result_df[new_name] = df[col].values ** d
                new_cols.append(new_name)

        # Cross terms for degree >= 2 with multiple columns
        if degree >= 2 and len(columns) >= 2:
            for col_a, col_b in combinations(columns, 2):
                new_name = f"{col_a}_x_{col_b}"
                result_df[new_name] = df[col_a].values * df[col_b].values
                new_cols.append(new_name)

        return FeatureResult(
            data=result_df,
            new_columns=new_cols,
            method="polynomial",
            description=f"Generated degree-{degree} polynomial features for {len(columns)} column(s)",
        )

    def interaction_terms(self, df: pd.DataFrame, columns: list[str]) -> FeatureResult:
        """Generate pairwise interaction terms (col_a * col_b)."""
        result_df = df.copy()
        new_cols: list[str] = []

        for col_a, col_b in combinations(columns, 2):
            new_name = f"{col_a}_x_{col_b}"
            result_df[new_name] = df[col_a].values * df[col_b].values
            new_cols.append(new_name)

        return FeatureResult(
            data=result_df,
            new_columns=new_cols,
            method="interaction",
            description=f"Generated {len(new_cols)} interaction term(s)",
        )

    def auto_engineer(self, df: pd.DataFrame) -> FeatureResult:
        """Auto-engineer features: scale numerics, encode categoricals, add interactions."""
        if df.empty:
            return FeatureResult(
                data=df.copy(),
                new_columns=[],
                method="auto",
                description="No features to engineer (empty DataFrame)",
            )

        result_df = df.copy()
        all_new_cols: list[str] = []

        # Scale numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            scale_result = self.scale(result_df, numeric_cols, method="standard")
            result_df = scale_result.data
            all_new_cols.extend(scale_result.new_columns)

        # One-hot encode categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            encode_result = self.encode_onehot(result_df, cat_cols)
            result_df = encode_result.data
            all_new_cols.extend(encode_result.new_columns)

        # Interaction terms for top numeric pairs (up to 3 columns)
        if len(numeric_cols) >= 2:
            top_cols = numeric_cols[:3]
            interact_result = self.interaction_terms(result_df, top_cols)
            result_df = interact_result.data
            all_new_cols.extend(interact_result.new_columns)

        return FeatureResult(
            data=result_df,
            new_columns=all_new_cols,
            method="auto",
            description=f"Auto-engineered {len(all_new_cols)} new feature(s)",
        )

    def select_features(
        self,
        df: pd.DataFrame,
        target: str,
        method: str = "variance",
        threshold: float = 0.01,
    ) -> SelectionResult:
        """Select features using variance, mutual information, or correlation.

        Methods:
        - variance: Remove features with variance below threshold
        - mutual_info: Keep features with mutual info above threshold
        - correlation: Remove features correlated >threshold with each other
        """
        from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression

        if method == "variance":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in numeric_cols:
                numeric_cols.remove(target)

            if not numeric_cols:
                return SelectionResult(
                    selected_columns=[],
                    dropped_columns=[],
                    scores={},
                )

            selector = VarianceThreshold(threshold=threshold)
            data = df[numeric_cols].values
            selector.fit(data)

            selected = [col for col, selected in zip(numeric_cols, selector.get_support()) if selected]
            dropped = [col for col in numeric_cols if col not in selected]
            scores = {col: float(df[col].var()) for col in numeric_cols}

            return SelectionResult(
                selected_columns=selected,
                dropped_columns=dropped,
                scores=scores,
            )

        elif method == "mutual_info":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in numeric_cols:
                numeric_cols.remove(target)

            if not numeric_cols or target not in df.columns:
                return SelectionResult(
                    selected_columns=[],
                    dropped_columns=[],
                    scores={},
                )

            X = df[numeric_cols].values
            y = df[target].values

            # Determine if classification or regression
            if len(np.unique(y)) < 10:
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=42)

            scores = {col: float(score) for col, score in zip(numeric_cols, mi_scores)}
            selected = [col for col, score in scores.items() if score >= threshold]
            dropped = [col for col in numeric_cols if col not in selected]

            return SelectionResult(
                selected_columns=selected,
                dropped_columns=dropped,
                scores=scores,
            )

        elif method == "correlation":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in numeric_cols:
                numeric_cols.remove(target)

            if not numeric_cols:
                return SelectionResult(
                    selected_columns=[],
                    dropped_columns=[],
                    scores={},
                )

            corr_matrix = df[numeric_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            # Find features to drop
            to_drop = set()
            for column in upper_tri.columns:
                if any(upper_tri[column] > threshold):
                    to_drop.add(column)

            selected = [col for col in numeric_cols if col not in to_drop]
            dropped = list(to_drop)
            scores = {
                col: float(df[col].corr(df[numeric_cols[0]])) if len(numeric_cols) > 0 else 0.0 for col in numeric_cols
            }

            return SelectionResult(
                selected_columns=selected,
                dropped_columns=dropped,
                scores=scores,
            )

        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: variance, mutual_info, correlation")

    def extract_time_features(self, df: pd.DataFrame, column: str) -> FeatureResult:
        """Extract time-based features from a datetime column.

        Extracts: day_of_week, month, quarter, year, is_weekend, hour (if time present).
        """
        if column not in df.columns:
            return FeatureResult(
                data=df.copy(),
                new_columns=[],
                method="time_features",
                description=f"Column '{column}' not found",
            )

        result_df = df.copy()
        result_df[column] = pd.to_datetime(result_df[column])
        new_cols: list[str] = []

        # Day of week (0=Monday, 6=Sunday)
        col_name = f"{column}_day_of_week"
        result_df[col_name] = result_df[column].dt.dayofweek
        new_cols.append(col_name)

        # Month
        col_name = f"{column}_month"
        result_df[col_name] = result_df[column].dt.month
        new_cols.append(col_name)

        # Quarter
        col_name = f"{column}_quarter"
        result_df[col_name] = result_df[column].dt.quarter
        new_cols.append(col_name)

        # Year
        col_name = f"{column}_year"
        result_df[col_name] = result_df[column].dt.year
        new_cols.append(col_name)

        # Is weekend
        col_name = f"{column}_is_weekend"
        result_df[col_name] = (result_df[column].dt.dayofweek >= 5).astype(int)
        new_cols.append(col_name)

        # Hour (if time is present)
        if result_df[column].dt.hour.sum() > 0:
            col_name = f"{column}_hour"
            result_df[col_name] = result_df[column].dt.hour
            new_cols.append(col_name)

        return FeatureResult(
            data=result_df,
            new_columns=new_cols,
            method="time_features",
            description=f"Extracted {len(new_cols)} time feature(s) from '{column}'",
        )

    def bin_numeric(
        self,
        df: pd.DataFrame,
        column: str,
        n_bins: int = 5,
        method: str = "equal_width",
    ) -> FeatureResult:
        """Bin a numeric column into categories.

        Methods:
        - equal_width: Equal-width bins using pd.cut
        - equal_frequency: Equal-frequency bins using pd.qcut
        """
        if column not in df.columns:
            return FeatureResult(
                data=df.copy(),
                new_columns=[],
                method="binning",
                description=f"Column '{column}' not found",
            )

        result_df = df.copy()
        new_col_name = f"{column}_binned"

        if method == "equal_width":
            result_df[new_col_name] = pd.cut(df[column], bins=n_bins, labels=False)
        elif method == "equal_frequency":
            result_df[new_col_name] = pd.qcut(df[column], q=n_bins, labels=False, duplicates="drop")
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: equal_width, equal_frequency")

        return FeatureResult(
            data=result_df,
            new_columns=[new_col_name],
            method=f"binning_{method}",
            description=f"Binned '{column}' into {n_bins} bins using {method}",
        )

    def target_encode(
        self,
        df: pd.DataFrame,
        column: str,
        target: str,
    ) -> tuple[FeatureResult, dict[str, float]]:
        """Target encode a categorical column using mean of target.

        Returns (FeatureResult, mapping_dict).
        """
        if column not in df.columns or target not in df.columns:
            return (
                FeatureResult(
                    data=df.copy(),
                    new_columns=[],
                    method="target_encoding",
                    description="Column or target not found",
                ),
                {},
            )

        result_df = df.copy()
        new_col_name = f"{column}_encoded"

        # Compute mean target value per category
        mapping = df.groupby(column)[target].mean().to_dict()

        # Apply encoding
        result_df[new_col_name] = df[column].map(mapping)

        # Fill missing with global mean
        global_mean = df[target].mean()
        result_df[new_col_name].fillna(global_mean, inplace=True)

        return (
            FeatureResult(
                data=result_df,
                new_columns=[new_col_name],
                method="target_encoding",
                description=f"Target encoded '{column}' using '{target}'",
            ),
            mapping,
        )
