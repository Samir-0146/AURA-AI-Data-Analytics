"""A safe, tool-using data analysis agent with an offline deterministic fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI

from config.settings import (
    LLM_TIMEOUT_SECONDS,
    MAX_AGENT_STEPS,
    OPENAI_MODEL,
    get_api_key,
)

from src import analyzer
from src.data_cleaner import clean_dataset, remove_duplicates
from src.report_generator import generate_pdf_report
from src.utils import to_jsonable
from src.visualizer import create_chart


# =============================================================
# AGENT RESULT
# =============================================================

@dataclass
class AgentResult:
    response: str
    plan: list[str]
    trace: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "rule-based"
    cleaned_df: pd.DataFrame | None = None
    report_bytes: bytes | None = None
    chart: go.Figure | None = None


def _compact(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        default=str,
    )[:12000]


# =============================================================
# DATA ANALYSIS AGENT
# =============================================================

class DataAnalysisAgent:

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_name: str = "Dataset",
    ) -> None:

        self.df = df.copy()
        self.dataset_name = dataset_name

        self.cleaned_df = None
        self.report_bytes = None
        self.generated_chart = None

        self.tools: dict[str, Callable[..., Any]] = {

            "dataset_information":
                lambda: analyzer.dataset_overview(self.df),

            "missing_value_analysis": lambda: {
                "summary": analyzer.missing_summary(self.df),
                "table": analyzer.missing_values_table(self.df),
            },

            "statistical_analysis":
                lambda: analyzer.statistics_table(self.df),

            "outlier_detection":
                lambda: analyzer.outlier_summary(
                    self.df,
                    method="iqr",
                ),

            "correlation_analysis":
                lambda: analyzer.top_correlations(
                    self.df,
                    min_abs=0.3,
                ),

            "duplicate_analysis": lambda: {
                "duplicate_rows":
                    analyzer.dataset_overview(self.df)["duplicate_rows"]
            },

            "data_cleaning": self._clean,

            "duplicate_removal": self._dedupe,

            "visualization": self._visualize,

            "report_generation": self._report,
        }

    # =========================================================
    # DATA CLEANING
    # =========================================================

    def _clean(
        self,
        missing_strategy: str = "none",
    ) -> dict[str, Any]:

        if missing_strategy not in {
            "none",
            "auto",
            "drop_rows",
        }:
            missing_strategy = "none"

        result = clean_dataset(
            self.df,
            missing_strategy=missing_strategy,
        )

        self.cleaned_df = result.df

        return {
            "summary": result.summary,
            "rows_after": len(result.df),
            "details": result.details,
        }

    # =========================================================
    # DUPLICATE REMOVAL
    # =========================================================

    def _dedupe(self) -> dict[str, Any]:

        result = remove_duplicates(self.df)

        self.cleaned_df = result.df

        return {
            "summary": result.summary,
            "rows_after": len(result.df),
        }

    # =========================================================
    # VISUALIZATION
    # =========================================================

    def _visualize(
        self,
        chart_type: str = "histogram",
        x: str | None = None,
        y: str | None = None,
    ) -> dict[str, Any]:

        numeric = self.df.select_dtypes(
            include="number"
        ).columns.tolist()

        columns = self.df.columns.tolist()

        if not columns:
            raise ValueError("Dataset has no columns.")

        x = x if x in columns else (
            numeric[0] if numeric else columns[0]
        )

        y = y if y in columns else (
            numeric[1] if len(numeric) > 1 else None
        )

        chart_type = chart_type.lower()

        if chart_type == "scatter" and y is None:
            chart_type = "histogram"

        figure = create_chart(
            self.df,
            chart_type,
            x=x,
            y=y,
            agg=(
                "count"
                if chart_type in {"bar", "pie"}
                else "none"
            ),
        )

        self.generated_chart = figure

        return {
            "chart_type": chart_type,
            "x": x,
            "y": y,
            "traces": len(figure.data),
            "status": "Chart created in the Visualization Studio.",
        }

    # =========================================================
    # REPORT GENERATION
    # =========================================================

    def _report(self) -> dict[str, Any]:

        try:
            self.report_bytes = generate_pdf_report(
                self.df,
                self.dataset_name,
            )

            return {
                "status": "Report generated successfully.",
                "filename": "AURA_Analysis_Report.pdf",
            }

        except TypeError:

            try:
                self.report_bytes = generate_pdf_report(self.df)

                return {
                    "status": "Report generated successfully.",
                    "filename": "AURA_Analysis_Report.pdf",
                }

            except Exception as exc:
                return {"error": str(exc)}

        except Exception as exc:
            return {"error": str(exc)}

    # =========================================================
    # TOOL EXECUTION
    # =========================================================

    def _run_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        trace: list[dict[str, Any]],
    ) -> Any:

        if name not in self.tools:
            raise ValueError(
                f"Tool '{name}' is not registered."
            )

        allowed_arguments = {
            "data_cleaning": {"missing_strategy"},
            "visualization": {"chart_type", "x", "y"},
        }

        try:

            allowed = allowed_arguments.get(name, set())

            kwargs = {
                key: value
                for key, value in arguments.items()
                if key in allowed
            }

            result = self.tools[name](**kwargs)

            trace.append({
                "tool": name,
                "status": "complete",
                "result": _compact(result)[:1000],
            })

            return result

        except Exception as exc:

            trace.append({
                "tool": name,
                "status": "error",
                "result": str(exc),
            })

            return {"error": str(exc)}

    # =========================================================
    # FALLBACK PLAN
    # =========================================================

    def _fallback_plan(
        self,
        request: str,
    ) -> list[str]:

        q = request.lower()
        matches: list[str] = []

        if "remove" in q and "duplicate" in q:
            matches.append("duplicate_removal")

        mapping = [
            ("missing", "missing_value_analysis"),
            ("null", "missing_value_analysis"),
            ("stat", "statistical_analysis"),
            ("outlier", "outlier_detection"),
            ("correl", "correlation_analysis"),
            ("duplicate", "duplicate_analysis"),
            ("clean", "data_cleaning"),
            ("visual", "visualization"),
            ("chart", "visualization"),
            ("report", "report_generation"),
        ]

        for keyword, tool in mapping:
            if keyword in q and tool not in matches:
                matches.append(tool)

        if not matches:
            return [
                "dataset_information",
                "missing_value_analysis",
                "statistical_analysis",
                "correlation_analysis",
            ]

        return matches

    # =========================================================
    # OFFLINE FALLBACK - INTELLIGENT RESPONSE
    # =========================================================

    def _fallback(
        self,
        request: str,
    ) -> AgentResult:

        plan = self._fallback_plan(request)

        trace: list[dict[str, Any]] = []
        results: dict[str, Any] = {}

        # Execute tools
        for tool in plan:

            arguments = {}

            if tool == "data_cleaning":
                arguments = {
                    "missing_strategy": "auto"
                }

            results[tool] = self._run_tool(
                tool,
                arguments,
                trace,
            )

        lines = [
            "## AURA Analysis Report",
            "",
            "I analyzed the active dataset using AURA's local analysis tools.",
            "",
        ]

        errors = []

        def has_error(result):
            return (
                isinstance(result, dict)
                and "error" in result
            )

        # -----------------------------------------------------
        # DATASET OVERVIEW
        # -----------------------------------------------------

        info = results.get("dataset_information")

        if isinstance(info, dict) and not has_error(info):

            rows = info.get("rows", "N/A")
            columns = info.get("columns", "N/A")
            quality = info.get("quality_score", "N/A")

            lines.extend([
                "### Dataset Overview",
                f"- Total records: {rows:,}" if isinstance(rows, (int, float)) else f"- Total records: {rows}",
                f"- Total columns: {columns}",
                f"- Data quality score: {quality}/100",
                "",
            ])

        elif has_error(info):
            errors.append(
                f"Dataset overview: {info['error']}"
            )

        # -----------------------------------------------------
        # MISSING VALUE ANALYSIS
        # -----------------------------------------------------

        missing = results.get("missing_value_analysis")

        if missing is not None:

            if has_error(missing):

                errors.append(
                    f"Missing value analysis: {missing['error']}"
                )

            else:

                summary = missing.get("summary", {})
                table = missing.get("table", [])

                if isinstance(table, pd.DataFrame):
                    table = table.to_dict("records")

                total = summary.get("total_missing", 0)
                columns_missing = summary.get(
                    "columns_with_missing", 0
                )
                rows_missing = summary.get(
                    "rows_with_missing", "N/A"
                )

                lines.extend([
                    "### Missing Value Analysis",
                    f"- Total missing cells: {total:,}" if isinstance(total, (int, float)) else f"- Total missing cells: {total}",
                    f"- Columns containing missing values: {columns_missing}",
                    f"- Rows containing missing values: {rows_missing}",
                ])

                if isinstance(table, list):

                    missing_columns = [
                        row for row in table
                        if isinstance(row, dict)
                        and (row.get("Missing", 0) or 0) > 0
                    ]

                    missing_columns.sort(
                        key=lambda row: row.get("Missing", 0) or 0,
                        reverse=True,
                    )

                    if missing_columns:

                        lines.append(
                            "- Column-wise missing values:"
                        )

                        for row in missing_columns[:10]:

                            count = row.get("Missing", 0)
                            percent = row.get("Missing %", "N/A")

                            lines.append(
                                f"  - {row.get('Column', 'Unknown')}: "
                                f"{count:,} missing ({percent}%)"
                                if isinstance(count, (int, float))
                                else f"  - {row.get('Column', 'Unknown')}: {count} missing ({percent}%)"
                            )

                    else:
                        lines.append(
                            "- No columns with missing values were listed."
                        )

                lines.append("")

        # -----------------------------------------------------
        # OUTLIER DETECTION
        # -----------------------------------------------------

        outliers = results.get("outlier_detection")

        if outliers is not None:

            if has_error(outliers):

                errors.append(
                    f"Outlier detection: {outliers['error']}"
                )

            else:

                if isinstance(outliers, pd.DataFrame):
                    outliers = outliers.to_dict("records")

                lines.extend([
                    "### Outlier Detection",
                    "",
                ])

                if isinstance(outliers, list):

                    detected = [
                        row for row in outliers
                        if isinstance(row, dict)
                        and (row.get("Outliers", 0) or 0) > 0
                    ]

                    detected.sort(
                        key=lambda row: row.get("Outliers", 0) or 0,
                        reverse=True,
                    )

                    if detected:

                        for row in detected[:10]:

                            count = row.get("Outliers", 0)
                            percent = row.get("Outlier %", "N/A")

                            lines.append(
                                f"- **{row.get('Column', 'Unknown')}**: "
                                f"{count:,} outliers ({percent}%)."
                                if isinstance(count, (int, float))
                                else f"- **{row.get('Column', 'Unknown')}**: {count} outliers ({percent}%)."
                            )

                    else:
                        lines.append(
                            "- No outliers were reported."
                        )

                else:
                    lines.append(
                        "- Outlier results are available in the tool trace."
                    )

                lines.append("")

        # -----------------------------------------------------
        # CORRELATION ANALYSIS
        # -----------------------------------------------------

        correlations = results.get("correlation_analysis")

        if correlations is not None:

            if has_error(correlations):

                errors.append(
                    f"Correlation analysis: {correlations['error']}"
                )

            else:

                if isinstance(correlations, pd.DataFrame):
                    correlations = correlations.to_dict("records")

                lines.extend([
                    "### Correlation Analysis",
                    "",
                ])

                if isinstance(correlations, list) and correlations:

                    for row in correlations[:10]:

                        if not isinstance(row, dict):
                            continue

                        lines.append(
                            f"- **{row.get('Column A', 'Unknown')}** "
                            f"and **{row.get('Column B', 'Unknown')}**: "
                            f"{row.get('Correlation', 'N/A')} "
                            f"({row.get('Strength', 'N/A')}, "
                            f"{row.get('Direction', 'N/A')})."
                        )

                else:
                    lines.append(
                        "- No correlation pairs were returned."
                    )

                lines.append("")

        # -----------------------------------------------------
        # STATISTICAL ANALYSIS
        # -----------------------------------------------------

        if "statistical_analysis" in results:

            result = results["statistical_analysis"]

            if has_error(result):
                errors.append(
                    f"Statistical analysis: {result['error']}"
                )

            else:
                lines.extend([
                    "### Statistical Analysis",
                    "- Statistical summary generated successfully.",
                    "- View the workspace panel for detailed statistics.",
                    "",
                ])

        # -----------------------------------------------------
        # DUPLICATE ANALYSIS
        # -----------------------------------------------------

        if "duplicate_analysis" in results:

            result = results["duplicate_analysis"]

            if has_error(result):

                errors.append(
                    f"Duplicate analysis: {result['error']}"
                )

            else:

                count = result.get(
                    "duplicate_rows", "N/A"
                )

                lines.extend([
                    "### Duplicate Analysis",
                    f"- Duplicate rows: {count:,}" if isinstance(count, (int, float)) else f"- Duplicate rows: {count}",
                    "",
                ])

        # -----------------------------------------------------
        # DATA CLEANING
        # -----------------------------------------------------

        if "data_cleaning" in results:

            result = results["data_cleaning"]

            if has_error(result):

                errors.append(
                    f"Data cleaning: {result['error']}"
                )

            else:

                lines.extend([
                    "### Data Cleaning",
                    f"- {result.get('summary', 'Cleaning completed.')}",
                    f"- Rows after cleaning: {result.get('rows_after', 'N/A')}",
                    "",
                ])

        # -----------------------------------------------------
        # DUPLICATE REMOVAL
        # -----------------------------------------------------

        if "duplicate_removal" in results:

            result = results["duplicate_removal"]

            if has_error(result):

                errors.append(
                    f"Duplicate removal: {result['error']}"
                )

            else:

                lines.extend([
                    "### Duplicate Removal",
                    f"- {result.get('summary', 'Duplicate removal completed.')}",
                    f"- Rows after removal: {result.get('rows_after', 'N/A')}",
                    "",
                ])

        # -----------------------------------------------------
        # VISUALIZATION
        # -----------------------------------------------------

        if "visualization" in results:

            result = results["visualization"]

            if has_error(result):

                errors.append(
                    f"Visualization: {result['error']}"
                )

            else:

                lines.extend([
                    "### Visualization",
                    f"- Chart type: {result.get('chart_type', 'N/A')}",
                    f"- X-axis: {result.get('x', 'N/A')}",
                    f"- Y-axis: {result.get('y', 'N/A')}",
                    "- Chart created in the Visualization Studio.",
                    "",
                ])

        # -----------------------------------------------------
        # REPORT GENERATION
        # -----------------------------------------------------

        if "report_generation" in results:

            result = results["report_generation"]

            if has_error(result):

                errors.append(
                    f"Report generation: {result['error']}"
                )

            else:

                lines.extend([
                    "### Report Generation",
                    "- PDF report generated successfully.",
                    "",
                ])

        # -----------------------------------------------------
        # RECOMMENDATIONS
        # -----------------------------------------------------

        recommendations = []

        missing_result = results.get(
            "missing_value_analysis", {}
        )

        if (
            isinstance(missing_result, dict)
            and not has_error(missing_result)
        ):

            total_missing = missing_result.get(
                "summary", {}
            ).get("total_missing", 0)

            if isinstance(total_missing, (int, float)) and total_missing > 0:

                recommendations.append(
                    "Review columns with missing values and choose an appropriate imputation or removal strategy before modeling."
                )

        outlier_result = results.get(
            "outlier_detection", []
        )

        if isinstance(outlier_result, pd.DataFrame):
            outlier_result = outlier_result.to_dict("records")

        if isinstance(outlier_result, list):

            high_outliers = [
                row for row in outlier_result
                if isinstance(row, dict)
                and isinstance(row.get("Outlier %"), (int, float))
                and row["Outlier %"] >= 5
            ]

            if high_outliers:

                names = ", ".join(
                    str(row.get("Column", "Unknown"))
                    for row in high_outliers[:5]
                )

                recommendations.append(
                    f"Investigate relatively high outlier percentages in: {names}. Verify whether these are valid business cases before removing or transforming them."
                )

        if recommendations:

            lines.extend([
                "### Recommendations",
                "",
            ])

            for recommendation in recommendations:
                lines.append(f"- {recommendation}")

            lines.append("")

        # -----------------------------------------------------
        # ERRORS
        # -----------------------------------------------------

        if errors:

            lines.extend([
                "### Issues",
                "",
            ])

            for error in errors:
                lines.append(f"- {error}")

            lines.append("")

        lines.extend([
            "---",
            "*Analysis generated from the active dataset using AURA's local tools.*",
        ])

        return AgentResult(
            response="\n".join(lines),
            plan=plan,
            trace=trace,
            mode="rule-based",
            cleaned_df=self.cleaned_df,
            report_bytes=self.report_bytes,
            chart=self.generated_chart,
        )

    # =========================================================
    # MAIN AGENT - OPENAI
    # =========================================================

    def run(
        self,
        request: str,
    ) -> AgentResult:

        if not request.strip():
            raise ValueError(
                "Enter an analysis request."
            )

        if not get_api_key():
            return self._fallback(request)

        try:

            tool_properties = {

                "data_cleaning": {
                    "missing_strategy": {
                        "type": "string",
                        "enum": [
                            "none",
                            "auto",
                            "drop_rows",
                        ],
                    }
                },

                "visualization": {
                    "chart_type": {
                        "type": "string"
                    },
                    "x": {
                        "type": "string"
                    },
                    "y": {
                        "type": "string"
                    },
                },
            }

            schema = [

                {
                    "type": "function",
                    "name": name,
                    "description": (
                        f"Run {name.replace('_', ' ')} "
                        "on the active dataset."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": tool_properties.get(
                            name,
                            {},
                        ),
                        "additionalProperties": False,
                    },
                }

                for name in self.tools
            ]

            overview = _compact(
                analyzer.dataset_overview(self.df)
            )

            client = OpenAI(
                timeout=LLM_TIMEOUT_SECONDS
            )

            inputs: list[Any] = [

                {
                    "role": "user",
                    "content": (
                        f"Dataset metadata: {overview}\n"
                        f"Request: {request}"
                    ),
                }

            ]

            trace: list[dict[str, Any]] = []
            plan: list[str] = []

            for _ in range(MAX_AGENT_STEPS):

                response = client.responses.create(
                    model=OPENAI_MODEL,
                    input=inputs,
                    tools=schema,
                    store=False,
                    instructions=(
                        "You are AURA's data analyst. "
                        "Use only provided tools for factual claims. "
                        "Plan briefly, call the needed tools, and explain "
                        "the actual findings using their returned values. "
                        "Mention important columns, counts, percentages, "
                        "correlation values, and practical recommendations. "
                        "Do not give generic tool-completed messages. "
                        "Never invent findings or execute code."
                    ),
                )

                calls = [
                    item
                    for item in response.output
                    if item.type == "function_call"
                ]

                if not calls:

                    return AgentResult(
                        response=(
                            response.output_text
                            or "Analysis complete."
                        ),
                        plan=plan or ["evaluate results"],
                        trace=trace,
                        mode="openai",
                        cleaned_df=self.cleaned_df,
                        report_bytes=self.report_bytes,
                        chart=self.generated_chart,
                    )

                inputs.extend(response.output)

                for call in calls:

                    args = json.loads(
                        call.arguments or "{}"
                    )

                    plan.append(call.name)

                    result = self._run_tool(
                        call.name,
                        args,
                        trace,
                    )

                    inputs.append({

                        "type": "function_call_output",

                        "call_id": call.call_id,

                        "output": _compact(result),

                    })

            return AgentResult(
                response=(
                    "The agent reached its safe tool-call limit. "
                    "The completed tool results are shown in the workspace."
                ),
                plan=plan,
                trace=trace,
                mode="openai",
                cleaned_df=self.cleaned_df,
                report_bytes=self.report_bytes,
                chart=self.generated_chart,
            )

        except Exception:
            return self._fallback(request)