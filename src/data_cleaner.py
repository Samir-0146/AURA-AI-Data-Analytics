"""Non-destructive data cleaning operations. Every function returns a new DataFrame inside a
CleaningResult and never mutates its input, so the original dataset stays unchanged and any
operation can be previewed before it is applied."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import reduce
from typing import Any

import numpy as np
import pandas as pd

from src.analyzer import duplicate_mask
from src.utils import is_empty_dataframe, resolve_column


class CleaningError(ValueError):
    """Raised when a cleaning operation receives invalid input."""


@dataclass
class CleaningResult:
    df: pd.DataFrame
    operation: str
    summary: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    cells_changed: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return (
            self.rows_before != self.rows_after
            or self.cols_before != self.cols_after
            or self.cells_changed > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "summary": self.summary,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "columns_before": self.cols_before,
            "columns_after": self.cols_after,
            "cells_changed": self.cells_changed,
            "changed": self.changed,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_data(df: pd.DataFrame) -> None:
    if is_empty_dataframe(df):
        raise CleaningError("The dataset is empty.")


def _make_result(
    before: pd.DataFrame,
    after: pd.DataFrame,
    operation: str,
    summary: str,
    cells_changed: int = 0,
    details: dict[str, Any] | None = None,
) -> CleaningResult:
    return CleaningResult(
        df=after,
        operation=operation,
        summary=summary,
        rows_before=int(before.shape[0]),
        rows_after=int(after.shape[0]),
        cols_before=int(before.shape[1]),
        cols_after=int(after.shape[1]),
        cells_changed=int(cells_changed),
        details=details or {},
    )


def _resolve_many(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    if columns is None:
        return list(df.columns)
    resolved: list[str] = []
    for name in columns:
        column = resolve_column(df, name)
        if column is None:
            raise CleaningError(f"Column '{name}' was not found in the dataset.")
        if column not in resolved:
            resolved.append(column)
    return resolved


def _is_numeric_non_bool(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _unique_labels(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for label in labels:
        candidate, counter = label, 1
        while candidate in seen:
            counter += 1
            candidate = f"{label}_{counter}"
        seen.add(candidate)
        result.append(candidate)
    return result


_BOOL_MAP = {
    "true": True, "false": False, "yes": True, "no": False, "y": True, "n": False,
    "t": True, "f": False, "1": True, "0": False, "1.0": True, "0.0": False,
}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    key = str(value).strip().lower()
    if key not in _BOOL_MAP:
        raise CleaningError(f"'{value}' cannot be interpreted as true/false.")
    return _BOOL_MAP[key]


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------
def remove_duplicates(
    df: pd.DataFrame, subset: list[str] | None = None, keep: str = "first"
) -> CleaningResult:
    _require_data(df)
    if keep not in {"first", "last"}:
        raise CleaningError("'keep' must be 'first' or 'last'.")
    columns = _resolve_many(df, subset) if subset else None
    mask = duplicate_mask(df, columns, keep)
    removed = int(mask.sum())
    cleaned = df.loc[~mask].reset_index(drop=True)
    scope = f"on columns {', '.join(columns)}" if columns else "across all columns"
    summary = (
        f"Removed {removed:,} duplicate row(s) {scope} (kept the {keep} occurrence)."
        if removed
        else f"No duplicate rows found {scope}."
    )
    return _make_result(df, cleaned, "remove_duplicates", summary, 0, {"removed_rows": removed})


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------
def _coerce_custom(series: pd.Series, raw: Any) -> Any:
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        raise CleaningError("A custom fill value is required.")
    dtype = series.dtype
    try:
        if pd.api.types.is_bool_dtype(dtype):
            return _parse_bool(raw)
        if pd.api.types.is_integer_dtype(dtype):
            number = float(raw)
            if not number.is_integer():
                raise CleaningError(f"'{raw}' is not a whole number, but column '{series.name}' is an integer column.")
            return int(number)
        if pd.api.types.is_float_dtype(dtype):
            return float(raw)
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return pd.to_datetime(raw)
    except (TypeError, ValueError) as exc:
        raise CleaningError(f"'{raw}' is not a valid value for column '{series.name}' ({dtype}).") from exc
    return str(raw)


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str,
    columns: list[str] | None = None,
    custom_value: Any = None,
) -> CleaningResult:
    """Fill (mean / median / mode / custom) or drop rows (drop_rows) with missing values."""
    _require_data(df)
    strategies = {"mean", "median", "mode", "custom", "drop_rows"}
    if strategy not in strategies:
        raise CleaningError(f"Strategy must be one of: {', '.join(sorted(strategies))}.")

    if columns is None:
        targets = [c for c in df.columns if df[c].isna().any()]
    else:
        targets = _resolve_many(df, columns)

    out = df.copy()
    filled: dict[str, int] = {}
    skipped: dict[str, str] = {}

    if strategy == "drop_rows":
        if not targets:
            return _make_result(df, out, "handle_missing_values", "No missing values to remove.")
        mask = out[targets].isna().any(axis=1)
        removed = int(mask.sum())
        out = out.loc[~mask].reset_index(drop=True)
        summary = f"Dropped {removed:,} row(s) with missing values in {len(targets)} column(s)."
        return _make_result(df, out, "handle_missing_values", summary, 0, {"removed_rows": removed})

    for column in targets:
        series = out[column]
        missing_count = int(series.isna().sum())
        if missing_count == 0:
            continue
        if strategy in {"mean", "median"}:
            if not _is_numeric_non_bool(series):
                skipped[column] = f"{strategy} needs a numeric column"
                continue
            if series.notna().sum() == 0:
                skipped[column] = "column has no values to compute from"
                continue
            value: Any = series.mean() if strategy == "mean" else series.median()
            if pd.api.types.is_integer_dtype(series.dtype):
                value = int(round(float(value)))
        elif strategy == "mode":
            modes = series.mode(dropna=True)
            if modes.empty:
                skipped[column] = "column has no values to compute from"
                continue
            value = modes.iloc[0]
        else:
            value = _coerce_custom(series, custom_value)

        if isinstance(series.dtype, pd.CategoricalDtype) and value not in series.cat.categories:
            series = series.cat.add_categories([value])
        out[column] = series.fillna(value)
        filled[column] = missing_count

    total = sum(filled.values())
    if total:
        summary = f"Filled {total:,} missing value(s) in {len(filled)} column(s) using {strategy}."
    else:
        summary = "No missing values were filled."
    if skipped:
        summary += " Skipped: " + "; ".join(f"{c} ({why})" for c, why in skipped.items()) + "."
    return _make_result(
        df, out, "handle_missing_values", summary, total, {"filled": filled, "skipped": skipped, "strategy": strategy}
    )


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
def convert_dtype(
    df: pd.DataFrame,
    column: str,
    target: str,
    errors: str = "raise",
    datetime_format: str | None = None,
) -> CleaningResult:
    """Convert a column to integer, float, string, category, datetime or boolean."""
    _require_data(df)
    targets = {"integer", "float", "string", "category", "datetime", "boolean"}
    if target not in targets:
        raise CleaningError(f"Target type must be one of: {', '.join(sorted(targets))}.")
    if errors not in {"raise", "coerce"}:
        raise CleaningError("'errors' must be 'raise' or 'coerce'.")
    name = resolve_column(df, column)
    if name is None:
        raise CleaningError(f"Column '{column}' was not found in the dataset.")

    series = df[name]
    failed = pd.Series(False, index=series.index)

    if target == "float":
        converted = pd.to_numeric(series, errors="coerce").astype("float64")
        failed = converted.isna() & series.notna()
    elif target == "integer":
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        fractional = numeric.notna() & ((numeric % 1) != 0)
        failed = (numeric.isna() & series.notna()) | fractional
        converted = numeric.where(~fractional).astype("Int64")
    elif target == "string":
        converted = series.astype("string")
    elif target == "category":
        converted = series.astype("category")
    elif target == "datetime":
        converted = pd.to_datetime(series, errors="coerce", format=datetime_format or "mixed")
        failed = converted.isna() & series.notna()
    else:
        values: list[Any] = []
        flags: list[bool] = []
        for item in series.tolist():
            if pd.isna(item):
                values.append(pd.NA)
                flags.append(False)
                continue
            try:
                values.append(_parse_bool(item))
                flags.append(False)
            except CleaningError:
                values.append(pd.NA)
                flags.append(True)
        converted = pd.Series(values, index=series.index, dtype="boolean")
        failed = pd.Series(flags, index=series.index)

    failed_count = int(failed.sum())
    examples = [str(v) for v in series[failed].head(3).tolist()]
    if failed_count and errors == "raise":
        raise CleaningError(
            f"{failed_count:,} value(s) in '{name}' cannot be converted to {target} "
            f"(examples: {', '.join(examples)}). Use 'coerce' to turn them into missing values."
        )

    out = df.copy()
    out[name] = converted
    summary = f"Converted '{name}' from {series.dtype} to {converted.dtype}."
    if failed_count:
        summary += f" {failed_count:,} value(s) could not be converted and became missing."
    return _make_result(
        df, out, "convert_dtype", summary, int(series.notna().sum()),
        {"column": name, "from": str(series.dtype), "to": str(converted.dtype), "failed": failed_count, "examples": examples},
    )


# ---------------------------------------------------------------------------
# Column names and whitespace
# ---------------------------------------------------------------------------
def rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> CleaningResult:
    _require_data(df)
    if not mapping:
        raise CleaningError("Provide at least one column to rename.")
    resolved: dict[str, str] = {}
    for old, new in mapping.items():
        column = resolve_column(df, old)
        if column is None:
            raise CleaningError(f"Column '{old}' was not found in the dataset.")
        label = str(new).strip()
        if not label:
            raise CleaningError(f"The new name for '{column}' cannot be empty.")
        resolved[column] = label
    new_names = [resolved.get(c, c) for c in df.columns]
    if len(set(new_names)) != len(new_names):
        raise CleaningError("Renaming would create duplicate column names.")
    out = df.copy()
    out.columns = new_names
    changed = {k: v for k, v in resolved.items() if k != v}
    summary = (
        f"Renamed {len(changed)} column(s): " + ", ".join(f"'{k}' → '{v}'" for k, v in changed.items()) + "."
        if changed
        else "The new names match the existing names; nothing changed."
    )
    return _make_result(df, out, "rename_columns", summary, 0, {"renamed": changed})


def clean_whitespace(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    clean_headers: bool = True,
    blank_to_nan: bool = True,
) -> CleaningResult:
    """Trim and collapse whitespace in text cells (and optionally headers); blank strings become missing."""
    _require_data(df)
    out = df.copy()
    if columns is None:
        text_cols = [
            c for c in out.columns
            if (pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]))
            and not isinstance(out[c].dtype, pd.CategoricalDtype)
        ]
    else:
        text_cols = _resolve_many(out, columns)

    cells_changed = 0
    blanks = 0
    for column in text_cols:
        series = out[column]
        is_text = series.map(lambda v: isinstance(v, str))
        if not is_text.any():
            continue
        cleaned = series[is_text].map(lambda v: re.sub(r"\s+", " ", v).strip())
        cells_changed += int((cleaned != series[is_text]).sum())
        updated = series.copy()
        updated[is_text] = cleaned
        if blank_to_nan:
            blank_mask = updated.map(lambda v: isinstance(v, str) and v == "")
            blanks += int(blank_mask.sum())
            updated = updated.mask(blank_mask, np.nan)
        out[column] = updated

    renamed = 0
    if clean_headers:
        old_names = [str(c) for c in out.columns]
        new_names = _unique_labels([re.sub(r"\s+", " ", n).strip() or "column" for n in old_names])
        renamed = sum(1 for a, b in zip(old_names, new_names) if a != b)
        out.columns = new_names

    cells_changed += blanks
    parts = []
    if cells_changed:
        parts.append(f"cleaned {cells_changed - blanks:,} text cell(s)")
    if blanks:
        parts.append(f"converted {blanks:,} blank string(s) to missing")
    if renamed:
        parts.append(f"tidied {renamed} column name(s)")
    summary = ("Whitespace cleaning: " + ", ".join(parts) + ".") if parts else "No whitespace issues found."
    return _make_result(
        df, out, "clean_whitespace", summary, cells_changed + renamed,
        {"cells_cleaned": cells_changed - blanks, "blank_to_missing": blanks, "headers_changed": renamed},
    )


# ---------------------------------------------------------------------------
# Composite cleaning
# ---------------------------------------------------------------------------
def clean_dataset(
    df: pd.DataFrame,
    fix_whitespace: bool = True,
    remove_dups: bool = True,
    missing_strategy: str = "none",
    drop_empty_columns: bool = False,
) -> CleaningResult:
    """Run a standard cleaning pipeline.

    missing_strategy: 'none', 'auto' (median for numeric, mode for other columns) or 'drop_rows'.
    """
    _require_data(df)
    if missing_strategy not in {"none", "auto", "drop_rows"}:
        raise CleaningError("missing_strategy must be 'none', 'auto' or 'drop_rows'.")

    current = df.copy()
    steps: list[dict[str, str]] = []
    cells = 0

    def record(result: CleaningResult) -> None:
        nonlocal current, cells
        current = result.df
        cells += result.cells_changed
        steps.append({"operation": result.operation, "summary": result.summary})

    if fix_whitespace:
        record(clean_whitespace(current))
    if drop_empty_columns:
        empty_cols = [c for c in current.columns if current[c].isna().all()]
        trimmed = current.drop(columns=empty_cols)
        if trimmed.shape[1] == 0:
            raise CleaningError("Every column is empty; nothing would remain after dropping empty columns.")
        record(
            _make_result(
                current, trimmed, "drop_empty_columns",
                f"Dropped {len(empty_cols)} fully empty column(s)." if empty_cols else "No fully empty columns found.",
                0, {"dropped": empty_cols},
            )
        )
    if remove_dups:
        record(remove_duplicates(current))
    if missing_strategy == "drop_rows":
        record(handle_missing_values(current, "drop_rows"))
    elif missing_strategy == "auto":
        numeric = [c for c in current.columns if _is_numeric_non_bool(current[c]) and current[c].isna().any()]
        other = [c for c in current.columns if c not in numeric and current[c].isna().any()]
        if numeric:
            record(handle_missing_values(current, "median", numeric))
        if other:
            record(handle_missing_values(current, "mode", other))
        if not numeric and not other:
            steps.append({"operation": "handle_missing_values", "summary": "No missing values to fill."})

    summary = (
        f"Cleaning pipeline finished: {df.shape[0]:,} → {current.shape[0]:,} rows, "
        f"{df.shape[1]:,} → {current.shape[1]:,} columns, {cells:,} cell(s) modified."
    )
    return _make_result(df, current, "clean_dataset", summary, cells, {"steps": steps})


# ---------------------------------------------------------------------------
# Filtering (declarative conditions only; no code evaluation)
# ---------------------------------------------------------------------------
_OPERATOR_ALIASES = {
    "=": "==", "eq": "==", "equals": "==", "ne": "!=", "not_equals": "!=",
    "gt": ">", "gte": ">=", "lt": "<", "lte": "<=",
    "is_null": "isnull", "isna": "isnull", "not_null": "notnull", "notna": "notnull",
}
FILTER_OPERATORS = (
    "==", "!=", ">", ">=", "<", "<=", "between", "in",
    "contains", "not_contains", "startswith", "endswith", "isnull", "notnull",
)


def _split_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _convert_scalar(raw: Any, kind: str, column: str) -> Any:
    try:
        if kind == "numeric":
            return float(raw)
        if kind == "datetime":
            return pd.to_datetime(raw)
        return _parse_bool(raw)
    except (TypeError, ValueError, CleaningError) as exc:
        raise CleaningError(f"'{raw}' is not a valid {kind} value for column '{column}'.") from exc


def _condition_mask(df: pd.DataFrame, condition: dict[str, Any]) -> pd.Series:
    if not isinstance(condition, dict):
        raise CleaningError("Each filter condition must be an object with column, operator and value.")
    column = resolve_column(df, condition.get("column"))
    if column is None:
        raise CleaningError(f"Column '{condition.get('column')}' was not found in the dataset.")
    op = str(condition.get("operator", "")).strip().lower()
    op = _OPERATOR_ALIASES.get(op, op)
    if op not in FILTER_OPERATORS:
        raise CleaningError(f"Unsupported operator '{op}'. Supported: {', '.join(FILTER_OPERATORS)}.")
    value = condition.get("value")
    series = df[column]

    if op == "isnull":
        return series.isna()
    if op == "notnull":
        return series.notna()

    is_num = _is_numeric_non_bool(series)
    is_dt = pd.api.types.is_datetime64_any_dtype(series)
    is_bool = pd.api.types.is_bool_dtype(series)
    kind = "numeric" if is_num else "datetime" if is_dt else "boolean" if is_bool else "text"

    if op in {"contains", "not_contains", "startswith", "endswith"}:
        if value is None or str(value) == "":
            raise CleaningError(f"A value is required for the '{op}' operator.")
        text = series.astype(str)
        pattern = str(value)
        if op in {"contains", "not_contains"}:
            hit = text.str.contains(pattern, case=False, regex=False)
        elif op == "startswith":
            hit = text.str.lower().str.startswith(pattern.lower())
        else:
            hit = text.str.lower().str.endswith(pattern.lower())
        hit = hit & series.notna()
        return (~hit & series.notna()) if op == "not_contains" else hit

    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise CleaningError(f"A value is required for the '{op}' operator.")

    if op in {">", ">=", "<", "<=", "between"} and kind not in {"numeric", "datetime"}:
        raise CleaningError(f"Operator '{op}' needs a numeric or datetime column; '{column}' is {kind}.")

    if kind == "text":
        lowered = series.astype(str).str.strip().str.lower()
        if op == "in":
            wanted = {str(v).strip().lower() for v in _split_list(value)}
            return lowered.isin(wanted) & series.notna()
        match = (lowered == str(value).strip().lower()) & series.notna()
        return match if op == "==" else ~match

    if op == "in":
        return series.isin([_convert_scalar(v, kind, column) for v in _split_list(value)])
    if op == "between":
        bounds = _split_list(value)
        if len(bounds) != 2:
            raise CleaningError("The 'between' operator needs exactly two values (low and high).")
        low, high = (_convert_scalar(b, kind, column) for b in bounds)
        return series.between(low, high)

    target = _convert_scalar(value, kind, column)
    comparisons = {
        "==": lambda s: s == target,
        "!=": lambda s: s != target,
        ">": lambda s: s > target,
        ">=": lambda s: s >= target,
        "<": lambda s: s < target,
        "<=": lambda s: s <= target,
    }
    return comparisons[op](series)


def describe_conditions(conditions: list[dict[str, Any]], logic: str = "and") -> str:
    parts = []
    for c in conditions:
        op = str(c.get("operator", "")).strip()
        value = c.get("value")
        parts.append(f"{c.get('column')} {op}" + (f" {value}" if value not in (None, "") else ""))
    return f" {logic.upper()} ".join(parts)


def apply_filters(
    df: pd.DataFrame, conditions: list[dict[str, Any]], logic: str = "and"
) -> CleaningResult:
    """Filter rows with declarative conditions: [{'column':..., 'operator':..., 'value':...}]."""
    _require_data(df)
    if not conditions:
        raise CleaningError("Provide at least one filter condition.")
    logic = str(logic).lower()
    if logic not in {"and", "or"}:
        raise CleaningError("Filter logic must be 'and' or 'or'.")
    masks = [_condition_mask(df, c).fillna(False).astype(bool) for c in conditions]
    combined = reduce(lambda a, b: (a & b) if logic == "and" else (a | b), masks)
    filtered = df.loc[combined].reset_index(drop=True)
    description = describe_conditions(conditions, logic)
    summary = f"Filter [{description}] kept {filtered.shape[0]:,} of {df.shape[0]:,} row(s)."
    return _make_result(
        df, filtered, "filter_dataset", summary, 0,
        {"conditions": conditions, "logic": logic, "matched_rows": int(filtered.shape[0]), "empty": filtered.empty},
    )