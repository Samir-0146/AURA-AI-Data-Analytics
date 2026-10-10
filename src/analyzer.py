"""Statistical analysis routines for AURA. All functions are pure and never modify their input."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import IQR_MULTIPLIER, TOP_CATEGORIES, ZSCORE_THRESHOLD
from src.utils import (
    dataframe_memory,
    get_column_groups,
    is_empty_dataframe,
    resolve_column,
)


class AnalysisError(ValueError):
    """Raised when an analysis cannot be performed on the given data."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _require_data(df: pd.DataFrame) -> None:
    if is_empty_dataframe(df):
        raise AnalysisError("The dataset is empty.")


def _resolve_columns(df: pd.DataFrame, columns: list[str] | None, kind: str) -> list[str]:
    """Resolve requested columns of a kind ('numeric' or 'categorical'); None selects all."""
    available = get_column_groups(df)[kind]
    if columns is None:
        return available
    resolved: list[str] = []
    for name in columns:
        column = resolve_column(df, name)
        if column is None:
            raise AnalysisError(f"Column '{name}' was not found in the dataset.")
        if column not in available:
            raise AnalysisError(f"Column '{column}' is not a {kind} column.")
        if column not in resolved:
            resolved.append(column)
    return resolved


def _safe_nunique(series: pd.Series, dropna: bool = True) -> int:
    try:
        return int(series.nunique(dropna=dropna))
    except TypeError:
        return int(series.astype(str).nunique(dropna=dropna))


def duplicate_mask(df: pd.DataFrame, subset: list[str] | None = None, keep: str = "first") -> pd.Series:
    """Boolean mask of duplicated rows, tolerant of unhashable cell values."""
    try:
        return df.duplicated(subset=subset, keep=keep)
    except TypeError:
        return df.astype(str).duplicated(subset=subset, keep=keep)


def _strength(r: float) -> str:
    a = abs(r)
    if a >= 0.8:
        return "Very strong"
    if a >= 0.6:
        return "Strong"
    if a >= 0.4:
        return "Moderate"
    if a >= 0.2:
        return "Weak"
    return "Negligible"


def _skew_label(skew: float) -> str:
    a = abs(skew)
    if a < 0.5:
        return "Approximately symmetric"
    direction = "right" if skew > 0 else "left"
    return f"Moderately {direction}-skewed" if a < 1 else f"Highly {direction}-skewed"


def _is_numeric_non_bool(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


# ---------------------------------------------------------------------------
# Overview and quality
# ---------------------------------------------------------------------------
def quality_label(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 50:
        return "Fair"
    return "Poor"


def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Headline metrics about a dataset. The quality score weights completeness 60% and uniqueness 40%."""
    _require_data(df)
    rows, cols = df.shape
    total_cells = rows * cols
    missing_total = int(df.isna().sum().sum())
    duplicates = int(duplicate_mask(df).sum())
    groups = get_column_groups(df)
    missing_pct = missing_total / total_cells * 100 if total_cells else 0.0
    duplicate_pct = duplicates / rows * 100 if rows else 0.0
    score = round(0.6 * (100 - missing_pct) + 0.4 * (100 - duplicate_pct), 1)
    return {
        "rows": int(rows),
        "columns": int(cols),
        "total_cells": int(total_cells),
        "missing_values": missing_total,
        "missing_pct": round(missing_pct, 2),
        "duplicate_rows": duplicates,
        "duplicate_pct": round(duplicate_pct, 2),
        "complete_rows": int((~df.isna().any(axis=1)).sum()),
        "numeric_columns": len(groups["numeric"]),
        "categorical_columns": len(groups["categorical"]),
        "datetime_columns": len(groups["datetime"]),
        "memory_usage": dataframe_memory(df),
        "quality_score": score,
        "quality_label": quality_label(score),
        "column_names": [str(c) for c in df.columns],
    }


def data_types_table(df: pd.DataFrame) -> pd.DataFrame:
    _require_data(df)
    groups = get_column_groups(df)
    rows = []
    for column in df.columns:
        if column in groups["numeric"]:
            group = "Numeric"
        elif column in groups["datetime"]:
            group = "Datetime"
        else:
            group = "Categorical"
        rows.append(
            {
                "Column": str(column),
                "Data Type": str(df[column].dtype),
                "Group": group,
                "Non-Null": int(df[column].notna().sum()),
                "Unique": _safe_nunique(df[column]),
            }
        )
    return pd.DataFrame(rows)


def column_information(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Detailed profile of one column."""
    _require_data(df)
    name = resolve_column(df, column)
    if name is None:
        raise AnalysisError(f"Column '{column}' was not found in the dataset.")
    series = df[name]
    rows = len(df)
    unique = _safe_nunique(series)
    info: dict[str, Any] = {
        "name": str(name),
        "dtype": str(series.dtype),
        "rows": int(rows),
        "non_null": int(series.notna().sum()),
        "missing": int(series.isna().sum()),
        "missing_pct": round(float(series.isna().mean() * 100), 2),
        "unique": unique,
        "unique_pct": round(unique / rows * 100, 2) if rows else 0.0,
        "sample_values": [str(v) for v in series.dropna().head(5).tolist()],
    }
    if _is_numeric_non_bool(series):
        values = series.dropna().astype(float)
        info["kind"] = "numeric"
        if not values.empty:
            info.update(
                {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()) if len(values) > 1 else None,
                }
            )
    elif pd.api.types.is_datetime64_any_dtype(series):
        info["kind"] = "datetime"
        valid = series.dropna()
        if not valid.empty:
            info["earliest"] = str(valid.min())
            info["latest"] = str(valid.max())
    else:
        info["kind"] = "categorical"
        counts = series.astype(str).where(series.notna()).value_counts().head(5)
        info["top_values"] = {str(k): int(v) for k, v in counts.items()}
    return info


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------
def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    _require_data(df)
    counts = df.isna().sum()
    table = pd.DataFrame(
        {
            "Column": [str(c) for c in counts.index],
            "Data Type": [str(df[c].dtype) for c in counts.index],
            "Missing": counts.values.astype(int),
            "Missing %": np.round(counts.values / len(df) * 100, 2),
        }
    )
    return table.sort_values(["Missing", "Column"], ascending=[False, True]).reset_index(drop=True)


def missing_summary(df: pd.DataFrame) -> dict[str, Any]:
    _require_data(df)
    per_column = df.isna().sum()
    return {
        "total_missing": int(per_column.sum()),
        "columns_with_missing": int((per_column > 0).sum()),
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "complete_rows": int((~df.isna().any(axis=1)).sum()),
        "total_rows": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
def statistics_table(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Mean, median, mode, std, variance, min, max, quartiles, skewness and kurtosis per numeric column."""
    _require_data(df)
    cols = _resolve_columns(df, columns, "numeric")
    if not cols:
        raise AnalysisError("The dataset has no numeric columns.")
    rows = []
    for column in cols:
        values = df[column].dropna().astype(float)
        if values.empty:
            rows.append({"Column": column, "Count": 0})
            continue
        modes = values.mode()
        q1, q3 = values.quantile([0.25, 0.75]).tolist()
        rows.append(
            {
                "Column": column,
                "Count": int(values.count()),
                "Mean": float(values.mean()),
                "Median": float(values.median()),
                "Mode": float(modes.iloc[0]) if not modes.empty else np.nan,
                "Std Dev": float(values.std()) if len(values) > 1 else np.nan,
                "Variance": float(values.var()) if len(values) > 1 else np.nan,
                "Min": float(values.min()),
                "Q1 (25%)": float(q1),
                "Q3 (75%)": float(q3),
                "Max": float(values.max()),
                "IQR": float(q3 - q1),
                "Range": float(values.max() - values.min()),
                "Skewness": float(values.skew()) if len(values) > 2 else np.nan,
                "Kurtosis": float(values.kurt()) if len(values) > 3 else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------
def outlier_mask(
    series: pd.Series,
    method: str = "iqr",
    multiplier: float = IQR_MULTIPLIER,
    threshold: float = ZSCORE_THRESHOLD,
) -> tuple[pd.Series, tuple[float, float]]:
    """Return (boolean mask aligned with the series, (lower bound, upper bound))."""
    if method not in {"iqr", "zscore"}:
        raise AnalysisError("Outlier method must be 'iqr' or 'zscore'.")
    if multiplier <= 0 or threshold <= 0:
        raise AnalysisError("The IQR multiplier and Z-score threshold must be positive.")
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    valid = numeric.dropna()
    empty_mask = pd.Series(False, index=series.index)
    if len(valid) < 4:
        return empty_mask, (np.nan, np.nan)
    if method == "iqr":
        q1, q3 = valid.quantile([0.25, 0.75]).tolist()
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        mask = (numeric < lower) | (numeric > upper)
    else:
        mean, std = float(valid.mean()), float(valid.std(ddof=0))
        if std == 0:
            return empty_mask, (mean, mean)
        lower, upper = mean - threshold * std, mean + threshold * std
        mask = ((numeric - mean) / std).abs() > threshold
    return mask.fillna(False).astype(bool), (float(lower), float(upper))


def outlier_summary(
    df: pd.DataFrame,
    method: str = "iqr",
    columns: list[str] | None = None,
    multiplier: float = IQR_MULTIPLIER,
    threshold: float = ZSCORE_THRESHOLD,
) -> pd.DataFrame:
    """Outlier counts per numeric column using IQR, Z-score or both ('both')."""
    _require_data(df)
    methods = ["iqr", "zscore"] if method == "both" else [method]
    for item in methods:
        if item not in {"iqr", "zscore"}:
            raise AnalysisError("Outlier method must be 'iqr', 'zscore' or 'both'.")
    cols = _resolve_columns(df, columns, "numeric")
    if not cols:
        raise AnalysisError("The dataset has no numeric columns.")
    rows = []
    for column in cols:
        valid_count = int(df[column].notna().sum())
        for item in methods:
            mask, (lower, upper) = outlier_mask(df[column], item, multiplier, threshold)
            count = int(mask.sum())
            flagged = df.loc[mask, column]
            rows.append(
                {
                    "Column": column,
                    "Method": "IQR" if item == "iqr" else "Z-score",
                    "Lower Bound": round(lower, 4) if np.isfinite(lower) else np.nan,
                    "Upper Bound": round(upper, 4) if np.isfinite(upper) else np.nan,
                    "Outliers": count,
                    "Outlier %": round(count / valid_count * 100, 2) if valid_count else 0.0,
                    "Min Outlier": float(flagged.min()) if count else np.nan,
                    "Max Outlier": float(flagged.max()) if count else np.nan,
                }
            )
    return pd.DataFrame(rows)


def outlier_rows(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    multiplier: float = IQR_MULTIPLIER,
    threshold: float = ZSCORE_THRESHOLD,
) -> pd.DataFrame:
    """The rows of the dataset flagged as outliers in one numeric column."""
    _require_data(df)
    name = _resolve_columns(df, [column], "numeric")[0]
    mask, _ = outlier_mask(df[name], method, multiplier, threshold)
    return df.loc[mask]


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
def correlation_matrix(
    df: pd.DataFrame, method: str = "pearson", columns: list[str] | None = None
) -> pd.DataFrame:
    _require_data(df)
    if method not in {"pearson", "spearman", "kendall"}:
        raise AnalysisError("Correlation method must be 'pearson', 'spearman' or 'kendall'.")
    cols = _resolve_columns(df, columns, "numeric")
    if len(cols) < 2:
        raise AnalysisError("At least two numeric columns are required for correlation analysis.")
    return df[cols].astype("float64").corr(method=method)


def top_correlations(
    df: pd.DataFrame,
    method: str = "pearson",
    min_abs: float = 0.0,
    limit: int = 15,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Unique column pairs ranked by absolute correlation."""
    corr = correlation_matrix(df, method, columns)
    names = corr.columns.tolist()
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = corr.iloc[i, j]
            if pd.isna(r) or abs(r) < min_abs:
                continue
            pairs.append(
                {
                    "Column A": names[i],
                    "Column B": names[j],
                    "Correlation": round(float(r), 4),
                    "Strength": _strength(float(r)),
                    "Direction": "Positive" if r > 0 else "Negative",
                }
            )
    table = pd.DataFrame(pairs, columns=["Column A", "Column B", "Correlation", "Strength", "Direction"])
    if table.empty:
        return table
    table = table.assign(_abs=table["Correlation"].abs()).sort_values("_abs", ascending=False)
    return table.drop(columns="_abs").head(limit).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Distributions and categories
# ---------------------------------------------------------------------------
def distribution_analysis(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Shape (skewness / kurtosis) and a normality test for each numeric column."""
    _require_data(df)
    cols = _resolve_columns(df, columns, "numeric")
    if not cols:
        raise AnalysisError("The dataset has no numeric columns.")
    rows = []
    for column in cols:
        values = df[column].dropna().astype(float)
        row: dict[str, Any] = {
            "Column": column,
            "Count": int(len(values)),
            "Skewness": np.nan,
            "Kurtosis": np.nan,
            "Shape": "Not enough variation",
            "Normality Test": "n/a",
            "p-value": np.nan,
        }
        if len(values) >= 3 and values.nunique() > 1:
            skew = float(values.skew())
            kurt = float(values.kurt()) if len(values) > 3 else np.nan
            sample = values.sample(5000, random_state=42) if len(values) > 5000 else values
            try:
                p_value = float(stats.shapiro(sample).pvalue)
            except (ValueError, TypeError):
                p_value = np.nan
            row.update(
                {
                    "Skewness": round(skew, 4),
                    "Kurtosis": round(kurt, 4) if not np.isnan(kurt) else np.nan,
                    "Shape": _skew_label(skew),
                    "p-value": round(p_value, 6) if not np.isnan(p_value) else np.nan,
                    "Normality Test": (
                        "n/a"
                        if np.isnan(p_value)
                        else ("Normal-like (Shapiro-Wilk p > 0.05)" if p_value > 0.05 else "Non-normal (Shapiro-Wilk p <= 0.05)")
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def categorical_analysis(
    df: pd.DataFrame, columns: list[str] | None = None, top: int = TOP_CATEGORIES
) -> dict[str, Any]:
    """Frequency tables and a cardinality summary for categorical columns."""
    _require_data(df)
    cols = _resolve_columns(df, columns, "categorical")
    if not cols:
        raise AnalysisError("The dataset has no categorical columns.")
    summary_rows = []
    frequencies: dict[str, pd.DataFrame] = {}
    for column in cols:
        series = df[column]
        text = series.astype(str).where(series.notna())
        counts = text.value_counts()
        non_null = int(counts.sum())
        table = pd.DataFrame(
            {
                "Value": counts.index.astype(str),
                "Count": counts.values.astype(int),
                "Percent": np.round(counts.values / non_null * 100, 2) if non_null else [],
            }
        ).head(max(1, int(top)))
        frequencies[column] = table.reset_index(drop=True)
        summary_rows.append(
            {
                "Column": column,
                "Unique": int(counts.shape[0]),
                "Most Frequent": str(counts.index[0]) if non_null else "—",
                "Frequency": int(counts.iloc[0]) if non_null else 0,
                "Frequency %": round(float(counts.iloc[0] / non_null * 100), 2) if non_null else 0.0,
                "Missing": int(series.isna().sum()),
            }
        )
    return {"summary": pd.DataFrame(summary_rows), "frequencies": frequencies}


# ---------------------------------------------------------------------------
# Findings (computed from real data)
# ---------------------------------------------------------------------------
def generate_findings(df: pd.DataFrame, max_items: int = 14) -> list[str]:
    """Plain-language findings derived from the actual dataset."""
    _require_data(df)
    overview = dataset_overview(df)
    findings: list[str] = [
        f"The dataset has {overview['rows']:,} rows and {overview['columns']:,} columns "
        f"({overview['numeric_columns']} numeric, {overview['categorical_columns']} categorical, "
        f"{overview['datetime_columns']} datetime)."
    ]

    missing = missing_values_table(df)
    with_missing = missing[missing["Missing"] > 0]
    if with_missing.empty:
        findings.append("No missing values were detected.")
    else:
        worst = with_missing.iloc[0]
        findings.append(
            f"{len(with_missing)} column(s) contain missing values; the highest is '{worst['Column']}' "
            f"with {worst['Missing %']:.1f}% missing ({int(overview['missing_values']):,} missing cells in total)."
        )
        heavy = with_missing[with_missing["Missing %"] >= 30]["Column"].tolist()
        if heavy:
            findings.append(
                "Columns with 30% or more missing data: " + ", ".join(f"'{c}'" for c in heavy[:5])
                + ". Consider dropping or imputing them carefully."
            )

    if overview["duplicate_rows"] > 0:
        findings.append(
            f"{overview['duplicate_rows']:,} duplicate row(s) found ({overview['duplicate_pct']:.1f}% of all rows)."
        )
    else:
        findings.append("No duplicate rows were found.")

    constant = [str(c) for c in df.columns if _safe_nunique(df[c], dropna=False) <= 1]
    if constant:
        findings.append(
            "Constant columns carrying no information: " + ", ".join(f"'{c}'" for c in constant[:5]) + "."
        )

    try:
        outliers = outlier_summary(df, "iqr")
        notable = outliers[outliers["Outlier %"] >= 5].sort_values("Outlier %", ascending=False)
        if notable.empty:
            findings.append("No numeric column has more than 5% IQR outliers.")
        else:
            parts = [f"'{r['Column']}' ({r['Outlier %']:.1f}%)" for _, r in notable.head(3).iterrows()]
            findings.append("Columns with notable IQR outlier shares: " + ", ".join(parts) + ".")
    except AnalysisError:
        pass

    try:
        strong = top_correlations(df, min_abs=0.7, limit=3)
        for _, r in strong.iterrows():
            findings.append(
                f"{r['Strength']} {r['Direction'].lower()} correlation between '{r['Column A']}' and "
                f"'{r['Column B']}' (r = {r['Correlation']:.2f})."
            )
    except AnalysisError:
        pass

    try:
        dist = distribution_analysis(df)
        skewed = dist[dist["Skewness"].abs() > 1].sort_values("Skewness", key=lambda s: s.abs(), ascending=False)
        if not skewed.empty:
            parts = [f"'{r['Column']}' ({r['Skewness']:.2f})" for _, r in skewed.head(3).iterrows()]
            findings.append("Highly skewed numeric columns: " + ", ".join(parts) + ".")
    except AnalysisError:
        pass

    if overview["rows"] > 20:
        id_like = [
            str(c)
            for c in df.columns
            if not _is_numeric_non_bool(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])
            and _safe_nunique(df[c]) / overview["rows"] > 0.95
        ]
        if id_like:
            findings.append(
                "Text columns that look like identifiers (almost every value is unique): "
                + ", ".join(f"'{c}'" for c in id_like[:5]) + "."
            )

    findings.append(f"Overall data quality score: {overview['quality_score']}/100 ({overview['quality_label']}).")
    return findings[:max_items]