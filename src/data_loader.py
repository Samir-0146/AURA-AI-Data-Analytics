
"""Validated loading and exporting of CSV / Excel datasets.
Has no Streamlit dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import pandas as pd

from config.settings import (
    ALLOWED_EXTENSIONS,
    MAX_COLUMNS,
    MAX_ROWS,
    MAX_UPLOAD_MB,
)


# ---------------------------------------------------------
# EXCEPTIONS AND RESULT MODEL
# ---------------------------------------------------------

class DataLoadError(ValueError):
    """Raised when an uploaded file cannot be turned into a usable dataset."""


@dataclass
class LoadResult:
    df: pd.DataFrame
    name: str
    size_bytes: int
    sheet_names: list[str] = field(default_factory=list)
    sheet: str | None = None


# ---------------------------------------------------------
# FILE VALIDATION
# ---------------------------------------------------------

def _extension(file_name: str) -> str:
    return Path(file_name or "").suffix.lower()


def validate_upload(file_name: str, content: bytes) -> None:
    """Validate the name, extension and size of an upload before parsing."""

    if not file_name or not file_name.strip():
        raise DataLoadError("The uploaded file has no name.")

    extension = _extension(file_name)

    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        raise DataLoadError(
            f"Unsupported file type '{extension or 'none'}'. "
            f"Allowed types: {allowed}."
        )

    if not content:
        raise DataLoadError("The uploaded file is empty.")

    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise DataLoadError(
            f"The file exceeds the {MAX_UPLOAD_MB} MB upload limit."
        )


# ---------------------------------------------------------
# EXCEL SHEET DETECTION
# ---------------------------------------------------------

def get_excel_sheets(file_name: str, content: bytes) -> list[str]:
    """Return sheet names of an Excel workbook."""

    if _extension(file_name) not in {".xlsx", ".xlsm"}:
        return []

    try:
        with pd.ExcelFile(
            BytesIO(content),
            engine="openpyxl",
        ) as workbook:
            return [str(name) for name in workbook.sheet_names]

    except Exception as exc:
        raise DataLoadError(
            f"Could not read the Excel workbook: {exc}"
        ) from exc


# ---------------------------------------------------------
# CSV / TSV READER
# ---------------------------------------------------------

def _read_csv(
    content: bytes,
    separator: str | None,
) -> pd.DataFrame:
    """Read CSV or TSV with supported text encodings."""

    last_error: Exception | None = None

    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):

        try:
            return pd.read_csv(
                BytesIO(content),
                sep=separator,
                engine="python" if separator is None else "c",
                encoding=encoding,
            )

        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    raise DataLoadError(
        "Could not decode the file with a supported text encoding: "
        f"{last_error}"
    )


# ---------------------------------------------------------
# COLUMN NORMALISATION
# ---------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert headers to strings and make duplicate names unique.

    The DataFrame is freshly loaded, so its column labels can be
    updated directly without creating another full DataFrame copy.
    """

    seen: dict[str, int] = {}
    new_columns: list[str] = []

    for position, column in enumerate(df.columns):

        label = str(column).strip() or f"column_{position + 1}"

        if label in seen:
            seen[label] += 1
            label = f"{label}_{seen[label]}"

        else:
            seen[label] = 0

        new_columns.append(label)

    df.columns = new_columns

    return df


# ---------------------------------------------------------
# MAIN DATA LOADER
# ---------------------------------------------------------

def load_dataframe(
    file_name: str,
    content: bytes,
    sheet_name: str | None = None,
) -> LoadResult:
    """
    Parse a CSV / TSV / Excel file into a validated DataFrame.

    Excel workbooks are opened once. Sheet names and selected
    sheet data are obtained from the same workbook instance.
    """

    validate_upload(file_name, content)

    extension = _extension(file_name)

    sheets: list[str] = []
    selected_sheet: str | None = None

    try:

        # CSV
        if extension == ".csv":

            df = _read_csv(
                content,
                separator=None,
            )

        # TSV
        elif extension == ".tsv":

            df = _read_csv(
                content,
                separator="\t",
            )

        # Excel
        else:

            with pd.ExcelFile(
                BytesIO(content),
                engine="openpyxl",
            ) as workbook:

                sheets = [
                    str(name)
                    for name in workbook.sheet_names
                ]

                if not sheets:
                    raise DataLoadError(
                        "The Excel workbook contains no sheets."
                    )

                selected_sheet = (
                    sheet_name
                    if sheet_name in sheets
                    else sheets[0]
                )

                df = pd.read_excel(
                    workbook,
                    sheet_name=selected_sheet,
                )

    except DataLoadError:
        raise

    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(
            "The file contains no data."
        ) from exc

    except pd.errors.ParserError as exc:
        raise DataLoadError(
            f"The file could not be parsed as a table: {exc}"
        ) from exc

    except Exception as exc:
        raise DataLoadError(
            f"Could not read the file: {exc}"
        ) from exc

    # -----------------------------------------------------
    # DATASET VALIDATION
    # -----------------------------------------------------

    if df.shape[1] == 0 or df.shape[0] == 0:
        raise DataLoadError(
            "The dataset is empty: it has no rows or no columns."
        )

    if df.shape[0] > MAX_ROWS:
        raise DataLoadError(
            f"The dataset has {df.shape[0]:,} rows; "
            f"the limit is {MAX_ROWS:,}."
        )

    if df.shape[1] > MAX_COLUMNS:
        raise DataLoadError(
            f"The dataset has {df.shape[1]:,} columns; "
            f"the limit is {MAX_COLUMNS:,}."
        )

    # Normalise headers without copying the entire DataFrame.
    df = _normalise_columns(df)

    return LoadResult(
        df=df,
        name=Path(file_name).name,
        size_bytes=len(content),
        sheet_names=sheets,
        sheet=selected_sheet,
    )


# ---------------------------------------------------------
# CSV EXPORT
# ---------------------------------------------------------

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to UTF-8 CSV bytes."""

    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------
# EXCEL EXPORT
# ---------------------------------------------------------

def dataframe_to_excel_bytes(
    df: pd.DataFrame,
    sheet_name: str = "AURA Data",
) -> bytes:
    """
    Serialise a DataFrame to XLSX bytes.

    Copy the DataFrame only when timezone-aware datetime
    columns need conversion.
    """

    timezone_columns = df.select_dtypes(
        include=["datetimetz"]
    ).columns

    if len(timezone_columns) > 0:

        export_df = df.copy()

        for column in timezone_columns:
            export_df[column] = (
                export_df[column].dt.tz_localize(None)
            )

    else:
        export_df = df

    buffer = BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:

        export_df.to_excel(
            writer,
            index=False,
            sheet_name=sheet_name[:31],
        )

    return buffer.getvalue()
