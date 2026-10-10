"""Shared helpers for AURA. This module imports nothing from other AURA modules."""
from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def now_iso() -> str:
    """Current local time as an ISO string without microseconds."""
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def human_bytes(num_bytes: float) -> str:
    """Format a byte count, e.g. 1536 -> '1.5 KB'."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def format_number(value: Any, decimals: int = 2) -> str:
    """Format numbers for display; integers get thousands separators."""
    if value is None:
        return "—"
    try:
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return "—"
    if number == int(number) and abs(number) < 1e15:
        return f"{int(number):,}"
    return f"{number:,.{decimals}f}"


def format_percent(value: float | None, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{decimals}f}%"


def truncate_text(text: str, limit: int = 300) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def safe_filename(name: str, default: str = "dataset") -> str:
    """Strip unsafe characters from a file name stem."""
    stem = Path(str(name)).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or default


def to_jsonable(obj: Any, max_rows: int = 200) -> Any:
    """Recursively convert numpy / pandas objects into JSON-safe Python values."""
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        number = float(obj)
        return None if (math.isnan(number) or math.isinf(number)) else number
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return None if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, (pd.Timedelta, timedelta)):
        return None if pd.isna(obj) else str(obj)
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.head(max_rows).to_dict(orient="records"), max_rows)
    if isinstance(obj, pd.Series):
        return {str(k): to_jsonable(v, max_rows) for k, v in obj.head(max_rows).items()}
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v, max_rows) for v in obj.tolist()[:max_rows]]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, max_rows) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v, max_rows) for v in list(obj)[:max_rows]]
    if isinstance(obj, Path):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def get_column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Split columns into numeric, datetime and categorical groups."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric and c not in datetime_cols]
    return {"numeric": numeric, "datetime": datetime_cols, "categorical": categorical}


def dataframe_memory(df: pd.DataFrame) -> str:
    return human_bytes(int(df.memory_usage(deep=True).sum()))


def resolve_column(df: pd.DataFrame, name: str) -> str | None:
    """Match a user-supplied column name to a real column (exact, case-insensitive, stripped)."""
    if name is None:
        return None
    target = str(name)
    if target in df.columns:
        return target
    lowered = target.strip().lower()
    for column in df.columns:
        if str(column).strip().lower() == lowered:
            return column
    return None


def is_empty_dataframe(df: pd.DataFrame | None) -> bool:
    return df is None or df.shape[0] == 0 or df.shape[1] == 0