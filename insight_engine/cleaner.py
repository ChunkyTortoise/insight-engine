"""Data Cleaning Pipeline: dedup, standardization, imputation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CleaningReport:
    original_rows: int
    cleaned_rows: int
    duplicates_removed: int
    nulls_imputed: int
    columns_standardized: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)


def remove_duplicates(
    df: pd.DataFrame, subset: list[str] | None = None, fuzzy: bool = False, threshold: float = 0.85
) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows. Optionally use fuzzy matching on string columns."""
    if fuzzy and subset:
        # Simple fuzzy dedup using normalized string comparison
        to_drop = set()
        str_cols = [c for c in subset if df[c].dtype == "object"]
        if str_cols:
            normalized = df[str_cols].fillna("").apply(
                lambda col: col.str.lower().str.strip()
                .str.replace(r"[^a-z0-9\s]", "", regex=True)
                .str.replace(r"\s+", " ", regex=True)
            )
            for i in range(len(normalized)):
                if i in to_drop:
                    continue
                for j in range(i + 1, len(normalized)):
                    if j in to_drop:
                        continue
                    matches = sum(
                        1 for c in str_cols if normalized.iloc[i][c] == normalized.iloc[j][c]
                    )
                    if matches / len(str_cols) >= threshold:
                        to_drop.add(j)

            cleaned = df.drop(index=list(to_drop)).reset_index(drop=True)
            return cleaned, len(to_drop)

    original_len = len(df)
    cleaned = df.drop_duplicates(subset=subset).reset_index(drop=True)
    return cleaned, original_len - len(cleaned)


def standardize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Standardize column names to snake_case and strip whitespace from string values."""
    result = df.copy()
    standardized = []

    # Column name standardization
    new_cols = {}
    for col in result.columns:
        new_name = col.strip().lower().replace(" ", "_").replace("-", "_")
        new_name = "".join(c for c in new_name if c.isalnum() or c == "_")
        if new_name != col:
            standardized.append(f"Renamed '{col}' → '{new_name}'")
        new_cols[col] = new_name
    result = result.rename(columns=new_cols)

    # Strip whitespace from string columns
    for col in result.select_dtypes(include=["object"]).columns:
        result[col] = result[col].str.strip()
        standardized.append(f"Stripped whitespace from '{col}'")

    return result, standardized


def impute_missing(
    df: pd.DataFrame, strategy: str = "smart"
) -> tuple[pd.DataFrame, int]:
    """Impute missing values.

    Strategies:
        smart: median for numeric, mode for categorical
        mean: mean for numeric, mode for categorical
        drop: drop rows with any missing values
    """
    result = df.copy()
    total_imputed = 0

    if strategy == "drop":
        before = len(result)
        result = result.dropna().reset_index(drop=True)
        return result, before - len(result)

    for col in result.columns:
        null_count = result[col].isna().sum()
        if null_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(result[col]):
            if strategy == "mean":
                fill_value = result[col].mean()
            else:  # smart or median
                fill_value = result[col].median()
            result[col] = result[col].fillna(fill_value)
        else:
            mode_vals = result[col].mode()
            if len(mode_vals) > 0:
                result[col] = result[col].fillna(mode_vals.iloc[0])

        total_imputed += null_count

    return result, int(total_imputed)


def clean_dataframe(
    df: pd.DataFrame,
    dedup: bool = True,
    dedup_subset: list[str] | None = None,
    fuzzy_dedup: bool = False,
    standardize: bool = True,
    impute: bool = True,
    impute_strategy: str = "smart",
) -> tuple[pd.DataFrame, CleaningReport]:
    """Run full cleaning pipeline."""
    result = df.copy()
    report = CleaningReport(original_rows=len(df), cleaned_rows=0, duplicates_removed=0, nulls_imputed=0)

    if standardize:
        result, std_ops = standardize_columns(result)
        report.columns_standardized = std_ops
        report.operations.append(f"Standardized {len(std_ops)} columns")

    if dedup:
        result, dups = remove_duplicates(result, subset=dedup_subset, fuzzy=fuzzy_dedup)
        report.duplicates_removed = dups
        report.operations.append(f"Removed {dups} duplicate rows")

    if impute:
        result, nulls = impute_missing(result, strategy=impute_strategy)
        report.nulls_imputed = nulls
        report.operations.append(f"Imputed {nulls} missing values")

    report.cleaned_rows = len(result)
    return result, report
