"""Plotly chart builders for AURA, styled for the dark premium theme."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.settings import CHART_PALETTE, COLORS, HISTOGRAM_BINS
from src.utils import get_column_groups, is_empty_dataframe, resolve_column

CHART_TYPES = ["Bar", "Line", "Scatter", "Histogram", "Pie", "Box", "Area", "Heatmap"]
AGGREGATIONS = ["sum", "mean", "median", "min", "max", "count"]
MAX_PLOT_POINTS = 20_000
MAX_CATEGORIES = 30
MAX_BOX_GROUPS = 20
PIE_SLICES = 10

DIVERGING_SCALE = [[0.0, "#F87171"], [0.5, "#1A1924"], [1.0, "#A78BFA"]]
SEQUENTIAL_SCALE = [[0.0, "#1E1B4B"], [0.5, "#7C3AED"], [1.0, "#F4F1F8"]]


class ChartError(ValueError):
    """Raised when a chart cannot be built from the given selections."""


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
def apply_theme(fig: go.Figure, title: str | None = None, height: int = 460) -> go.Figure:
    layout: dict = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "DM Sans, Space Grotesk, sans-serif", "color": COLORS["text"], "size": 13},
        "colorway": CHART_PALETTE,
        "margin": {"l": 24, "r": 24, "t": 64 if title else 24, "b": 24},
        "height": height,
        "legend": {"bgcolor": "rgba(0,0,0,0)"},
        "hoverlabel": {"bgcolor": COLORS["card"], "font_color": COLORS["text"], "bordercolor": COLORS["accent"]},
    }
    if title:
        layout["title"] = {
            "text": title,
            "x": 0.01,
            "font": {"family": "Space Grotesk, DM Sans, sans-serif", "size": 18, "color": COLORS["text"]},
        }
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"], linecolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"], linecolor=COLORS["border"])
    return fig


def _empty_figure(message: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        font={"size": 15, "color": COLORS["muted"]},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_theme(fig, None, height)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _col(df: pd.DataFrame, name: str | None, label: str) -> str:
    if not name:
        raise ChartError(f"Select a column for {label}.")
    column = resolve_column(df, name)
    if column is None:
        raise ChartError(f"Column '{name}' was not found in the dataset.")
    return column


def _optional_col(df: pd.DataFrame, name: str | None) -> str | None:
    if not name or str(name).lower() in {"none", "(none)"}:
        return None
    column = resolve_column(df, name)
    if column is None:
        raise ChartError(f"Column '{name}' was not found in the dataset.")
    return column


def _require_numeric(df: pd.DataFrame, column: str, label: str) -> None:
    series = df[column]
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        raise ChartError(f"{label} must be a numeric column; '{column}' is {series.dtype}.")


def _limit_points(data: pd.DataFrame, limit: int = MAX_PLOT_POINTS) -> pd.DataFrame:
    return data if len(data) <= limit else data.sample(limit, random_state=42).sort_index()


def _unique(items: list[str | None]) -> list[str]:
    return list(dict.fromkeys(i for i in items if i))


def _aggregate(df: pd.DataFrame, keys: list[str], y: str | None, agg: str) -> tuple[pd.DataFrame, str]:
    if agg not in AGGREGATIONS:
        raise ChartError(f"Aggregation must be one of: {', '.join(AGGREGATIONS)}.")
    if y is None:
        grouped = df.groupby(keys, dropna=True, observed=True).size().reset_index(name="Count")
        return grouped, "Count"
    if y in keys:
        raise ChartError("The value column must differ from the grouping columns.")
    if agg != "count":
        _require_numeric(df, y, "The Y axis")
    value_name = f"{agg.title()} of {y}"
    grouped = df.groupby(keys, dropna=True, observed=True)[y].agg(agg).reset_index()
    return grouped.rename(columns={y: value_name}), value_name


def _top_categories(grouped: pd.DataFrame, x: str, value: str, limit: int = MAX_CATEGORIES) -> pd.DataFrame:
    if grouped[x].nunique() <= limit:
        return grouped
    keep = grouped.groupby(x)[value].sum().nlargest(limit).index
    return grouped[grouped[x].isin(keep)]


# ---------------------------------------------------------------------------
# Individual chart builders
# ---------------------------------------------------------------------------
def _bar(df: pd.DataFrame, x: str, y: str | None, color: str | None, agg: str) -> tuple[go.Figure, str]:
    if agg == "none":
        raise ChartError("Bar charts need an aggregation (sum, mean, median, min, max or count).")
    grouped, value = _aggregate(df, _unique([x, color]), y, agg)
    if grouped.empty:
        raise ChartError("No data to plot after removing missing values.")
    grouped = _top_categories(grouped, x, value)
    if not color and not pd.api.types.is_numeric_dtype(grouped[x]) and not pd.api.types.is_datetime64_any_dtype(grouped[x]):
        grouped = grouped.sort_values(value, ascending=False)
    fig = px.bar(grouped, x=x, y=value, color=color, barmode="group", color_discrete_sequence=CHART_PALETTE)
    return fig, f"{value} by {x}"


def _line_or_area(
    df: pd.DataFrame, x: str, y: str, color: str | None, agg: str, area: bool
) -> tuple[go.Figure, str]:
    _require_numeric(df, y, "The Y axis")
    data = df[_unique([x, y, color])].dropna(subset=[x, y])
    if data.empty:
        raise ChartError("No data to plot after removing missing values.")
    y_plot = y
    if agg != "none":
        data, y_plot = _aggregate(data, _unique([x, color]), y, agg)
    data = _limit_points(data.sort_values(x))
    if area:
        fig = px.area(data, x=x, y=y_plot, color=color, color_discrete_sequence=CHART_PALETTE)
    else:
        fig = px.line(data, x=x, y=y_plot, color=color, markers=len(data) <= 60, color_discrete_sequence=CHART_PALETTE)
    return fig, f"{y_plot} over {x}"


def _scatter(df: pd.DataFrame, x: str, y: str, color: str | None) -> tuple[go.Figure, str]:
    _require_numeric(df, x, "The X axis")
    _require_numeric(df, y, "The Y axis")
    data = df[_unique([x, y, color])].dropna(subset=[x, y])
    if data.empty:
        raise ChartError("No data to plot after removing missing values.")
    data = _limit_points(data)
    fig = px.scatter(data, x=x, y=y, color=color, opacity=0.75, color_discrete_sequence=CHART_PALETTE)
    return fig, f"{y} vs {x}"


def _histogram(df: pd.DataFrame, x: str, color: str | None, bins: int) -> tuple[go.Figure, str]:
    _require_numeric(df, x, "The X axis")
    if not 2 <= int(bins) <= 200:
        raise ChartError("Number of bins must be between 2 and 200.")
    data = df[_unique([x, color])].dropna(subset=[x])
    if data.empty:
        raise ChartError("No data to plot after removing missing values.")
    fig = px.histogram(
        data, x=x, color=color, nbins=int(bins),
        barmode="overlay" if color else "relative", opacity=0.8 if color else 1.0,
        color_discrete_sequence=CHART_PALETTE,
    )
    fig.update_traces(marker_line_color=COLORS["background"], marker_line_width=1)
    return fig, f"Distribution of {x}"


def _pie(df: pd.DataFrame, x: str, y: str | None, agg: str) -> tuple[go.Figure, str]:
    if agg == "none":
        raise ChartError("Pie charts need an aggregation (sum, mean, median, min, max or count).")
    grouped, value = _aggregate(df, [x], y, agg)
    if grouped.empty:
        raise ChartError("No data to plot after removing missing values.")
    if (grouped[value] < 0).any():
        raise ChartError("Pie charts cannot display negative values.")
    grouped[x] = grouped[x].astype(str)
    grouped = grouped.sort_values(value, ascending=False)
    if len(grouped) > PIE_SLICES:
        other = grouped.iloc[PIE_SLICES:][value].sum()
        grouped = pd.concat(
            [grouped.head(PIE_SLICES), pd.DataFrame({x: ["Other"], value: [other]})], ignore_index=True
        )
    fig = px.pie(grouped, names=x, values=value, hole=0.45, color_discrete_sequence=CHART_PALETTE)
    fig.update_traces(
        textinfo="percent", textposition="inside",
        marker={"line": {"color": COLORS["background"], "width": 2}},
    )
    return fig, f"{value} by {x}"


def _box(df: pd.DataFrame, y: str, x: str | None) -> tuple[go.Figure, str]:
    _require_numeric(df, y, "The Y axis")
    data = df[_unique([y, x])].dropna(subset=[y])
    if data.empty:
        raise ChartError("No data to plot after removing missing values.")
    if x:
        data = data.dropna(subset=[x])
        top = data[x].astype(str).value_counts().head(MAX_BOX_GROUPS).index
        data = data[data[x].astype(str).isin(top)].copy()
        data[x] = data[x].astype(str)
    fig = px.box(data, y=y, x=x, color=x, points="outliers", color_discrete_sequence=CHART_PALETTE)
    fig.update_layout(showlegend=False)
    return fig, f"Box plot of {y}" + (f" by {x}" if x else "")


def correlation_heatmap(corr: pd.DataFrame, title: str = "Correlation Heatmap", height: int = 520) -> go.Figure:
    if corr is None or corr.shape[0] < 2:
        return _empty_figure("At least two numeric columns are required.", height)
    fig = px.imshow(
        corr, zmin=-1, zmax=1, color_continuous_scale=DIVERGING_SCALE, aspect="auto",
        text_auto=".2f" if corr.shape[0] <= 15 else False,
    )
    fig.update_xaxes(side="bottom", tickangle=-35)
    fig.update_coloraxes(colorbar={"thickness": 12, "outlinewidth": 0})
    return apply_theme(fig, title, height)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def create_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    agg: str = "sum",
    bins: int = HISTOGRAM_BINS,
    columns: list[str] | None = None,
    title: str | None = None,
    corr_method: str = "pearson",
    height: int = 460,
) -> go.Figure:
    """Build a themed Plotly figure. Raises ChartError for invalid selections."""
    if is_empty_dataframe(df):
        raise ChartError("The dataset is empty.")
    kind = str(chart_type).strip().lower()
    agg = str(agg or "sum").lower()
    if agg != "none" and agg not in AGGREGATIONS:
        raise ChartError(f"Aggregation must be 'none' or one of: {', '.join(AGGREGATIONS)}.")
    color_col = _optional_col(df, color)

    if kind == "heatmap":
        from src.analyzer import AnalysisError, correlation_matrix  # local import keeps module graph acyclic

        try:
            corr = correlation_matrix(df, corr_method, columns)
        except AnalysisError as exc:
            raise ChartError(str(exc)) from exc
        return correlation_heatmap(corr, title or "Correlation Heatmap", max(height, 480))

    if kind == "bar":
        x_col = _col(df, x, "the X axis")
        y_col = _optional_col(df, y)
        fig, auto_title = _bar(df, x_col, y_col, color_col, agg)
    elif kind in {"line", "area"}:
        fig, auto_title = _line_or_area(df, _col(df, x, "the X axis"), _col(df, y, "the Y axis"), color_col, agg, kind == "area")
    elif kind == "scatter":
        fig, auto_title = _scatter(df, _col(df, x, "the X axis"), _col(df, y, "the Y axis"), color_col)
    elif kind == "histogram":
        fig, auto_title = _histogram(df, _col(df, x, "the X axis"), color_col, bins)
    elif kind == "pie":
        fig, auto_title = _pie(df, _col(df, x, "the categories"), _optional_col(df, y), agg)
    elif kind == "box":
        fig, auto_title = _box(df, _col(df, y, "the Y axis"), _optional_col(df, x))
    else:
        raise ChartError(f"Unknown chart type '{chart_type}'. Choose from: {', '.join(CHART_TYPES)}.")

    return apply_theme(fig, title or auto_title, height)


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------
def missing_values_chart(missing_table: pd.DataFrame, top: int = 20) -> go.Figure:
    data = missing_table[missing_table["Missing"] > 0].head(top)
    if data.empty:
        return _empty_figure("No missing values in this dataset 🎉".replace(" 🎉", ""))
    data = data.sort_values("Missing %", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=data["Missing %"], y=data["Column"], orientation="h",
            marker={"color": COLORS["accent"]},
            customdata=data["Missing"],
            hovertemplate="%{y}<br>%{x:.2f}% missing (%{customdata:,} cells)<extra></extra>",
        )
    )
    fig.update_xaxes(title="Missing %")
    return apply_theme(fig, "Missing Values by Column", max(320, 60 + 28 * len(data)))


def dtype_chart(df: pd.DataFrame) -> go.Figure:
    groups = get_column_groups(df)
    labels = ["Numeric", "Categorical", "Datetime"]
    values = [len(groups["numeric"]), len(groups["categorical"]), len(groups["datetime"])]
    pairs = [(l, v) for l, v in zip(labels, values) if v > 0]
    if not pairs:
        return _empty_figure("No columns to display.")
    fig = go.Figure(
        go.Pie(
            labels=[p[0] for p in pairs], values=[p[1] for p in pairs], hole=0.6,
            marker={"colors": CHART_PALETTE[: len(pairs)], "line": {"color": COLORS["background"], "width": 2}},
            textinfo="label+value",
        )
    )
    fig.update_layout(showlegend=False)
    return apply_theme(fig, "Column Types", 320)


def distribution_chart(df: pd.DataFrame, column: str, bins: int = HISTOGRAM_BINS) -> go.Figure:
    name = _col(df, column, "the distribution chart")
    _require_numeric(df, name, "The selected column")
    data = df[[name]].dropna()
    if data.empty:
        raise ChartError(f"Column '{name}' has no values to plot.")
    fig = px.histogram(
        data, x=name, nbins=int(bins), marginal="box", color_discrete_sequence=[COLORS["accent"]]
    )
    fig.update_traces(marker_line_color=COLORS["background"], marker_line_width=1, selector={"type": "histogram"})
    return apply_theme(fig, f"Distribution of {name}", 420)


def category_bar(frequency: pd.DataFrame, column: str) -> go.Figure:
    if frequency is None or frequency.empty:
        return _empty_figure("No values to display.")
    data = frequency.sort_values("Count", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=data["Count"], y=data["Value"], orientation="h", marker={"color": COLORS["accent_light"]},
            customdata=data["Percent"], hovertemplate="%{y}<br>%{x:,} rows (%{customdata:.1f}%)<extra></extra>",
        )
    )
    return apply_theme(fig, f"Top values in {column}", max(300, 70 + 30 * len(data)))