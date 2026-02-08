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
