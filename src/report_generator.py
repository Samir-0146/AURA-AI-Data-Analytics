"""Professional, data-derived PDF reports for AURA."""
from __future__ import annotations

from io import BytesIO
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.analyzer import dataset_overview, generate_findings, missing_values_table, statistics_table, top_correlations


def _table_from_frame(frame: pd.DataFrame, max_rows: int = 12) -> Table:
    data = [list(frame.columns)] + frame.head(max_rows).fillna("—").astype(str).values.tolist()
    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6D4EDB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDD7F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F5FC")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _missing_chart(missing: pd.DataFrame) -> Drawing | None:
    """Create a compact native ReportLab chart; no browser or image renderer required."""
    data = missing[missing["Missing"] > 0].head(8)
    if data.empty:
        return None
    drawing = Drawing(460, max(130, 35 + 22 * len(data)))
    chart = HorizontalBarChart()
    chart.x, chart.y, chart.width, chart.height = 150, 12, 285, max(95, 18 * len(data))
    chart.data = [data["Missing %"].astype(float).tolist()]
    chart.categoryAxis.categoryNames = data["Column"].astype(str).tolist()
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.valueMin = 0
    chart.bars[0].fillColor = colors.HexColor("#8B5CF6")
    drawing.add(String(0, chart.height + 30, "Missing values by column (%)", fontName="Helvetica-Bold", fontSize=10, fillColor=colors.HexColor("#322460")))
    drawing.add(chart)
    return drawing


def _numeric_mean_chart(stats: pd.DataFrame) -> Drawing | None:
    """Show a compact profile chart for numeric columns when statistics exist."""
    if stats.empty or "Mean" not in stats:
        return None
    data = stats.dropna(subset=["Mean"]).head(8)
    if data.empty:
        return None
    drawing = Drawing(460, 210)
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 45, 32, 380, 135
    chart.data = [data["Mean"].astype(float).tolist()]
    chart.categoryAxis.categoryNames = data["Column"].astype(str).tolist()
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.angle = 25
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = colors.HexColor("#6D4EDB")
    drawing.add(String(0, 190, "Numeric column means", fontName="Helvetica-Bold", fontSize=10, fillColor=colors.HexColor("#322460")))
    drawing.add(chart)
    return drawing


def generate_pdf_report(df: pd.DataFrame, dataset_name: str = "Dataset") -> bytes:
    """Build a downloadable PDF containing only computed dataset results."""
    overview = dataset_overview(df)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("AuraTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24,
                           leading=29, textColor=colors.HexColor("#322460"), alignment=TA_LEFT)
    heading = ParagraphStyle("AuraHeading", parent=styles["Heading2"], textColor=colors.HexColor("#6D4EDB"),
                             spaceBefore=12, spaceAfter=7)
    body = ParagraphStyle("AuraBody", parent=styles["BodyText"], leading=16, textColor=colors.HexColor("#332F3F"))
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    story = [Paragraph("AURA", title), Paragraph("Autonomous Data Intelligence Report", heading),
             Paragraph(f"Dataset: <b>{dataset_name}</b>", body), Spacer(1, 10)]
    metrics = pd.DataFrame([{"Rows": overview["rows"], "Columns": overview["columns"], "Numeric": overview["numeric_columns"],
                             "Missing cells": overview["missing_values"], "Duplicate rows": overview["duplicate_rows"],
                             "Quality score": f'{overview["quality_score"]}/100 ({overview["quality_label"]})'}])
    story += [Paragraph("Executive summary", heading), _table_from_frame(metrics, 1), Paragraph("Key findings", heading)]
    for finding in generate_findings(df, max_items=10):
        story.append(Paragraph("• " + finding, body))
        story.append(Spacer(1, 3))
    missing = missing_values_table(df)
    if (missing["Missing"] > 0).any():
        story += [Paragraph("Data quality", heading), _table_from_frame(missing[missing["Missing"] > 0]), Spacer(1, 8)]
        chart = _missing_chart(missing)
        if chart is not None:
            story += [chart, Spacer(1, 8)]
    try:
        stats = statistics_table(df).round(4)
        story += [PageBreak(), Paragraph("Numeric statistics", heading), _table_from_frame(stats)]
        chart = _numeric_mean_chart(stats)
        if chart is not None:
            story += [Spacer(1, 8), chart]
    except Exception:
        pass
    try:
        correlations = top_correlations(df, min_abs=0.3).head(10)
        if not correlations.empty:
            story += [Paragraph("Notable correlations", heading), _table_from_frame(correlations)]
    except Exception:
        pass
    document.build(story)
    return buffer.getvalue()
