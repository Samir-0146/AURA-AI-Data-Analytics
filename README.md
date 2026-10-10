# AURA++++ — Autonomous Data Intelligence

AURA is a premium Streamlit data-intelligence workspace with a safe, tool-driven AI analyst. It ingests tabular data, preserves the original upload, performs real analysis and cleaning on a working copy, builds interactive Plotly charts, and generates professional PDF reports.

## Workspaces

- **Executive Dashboard** — quality score, missing-data signals, distribution chart, and computed findings.
- **Data Explorer** — preview, filtering, sorting, data types, descriptive statistics, categories, outliers, and correlations.
- **AI Agent Workspace** — OpenAI Responses API function-calling orchestration or deterministic local analysis fallback.
- **Data Cleaning Studio** — duplicate removal, standard cleaning, missing-data strategies, type conversion, renaming, and exports.
- **Visualization Studio** — bar, line, scatter, histogram, pie, box, area, and correlation heatmap charts.
- **Reports Center** — downloadable PDF containing overview, real findings, tables, correlation results, and missing-data chart.

## Setup and run

1. Ensure Python 3.12 is installed and available to this project’s virtual environment.
2. Activate the existing environment: `./venv/Scripts/Activate.ps1`
3. Install or confirm dependencies: `python -m pip install -r requirements.txt`
4. Optionally copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
5. Start AURA with the project interpreter:

```powershell
& 'B:\COLLEGE\PDS\venv\Scripts\python.exe' -m streamlit run app.py
```

CSV, TSV, XLSX, and XLSM uploads are supported. All cleaning affects only the working copy; use **Restore original dataset** at any time.

## AI safety

The model may select only registered analysis functions. AURA validates and filters tool arguments and never runs arbitrary model-provided Python or shell code. Without an API key, the fallback executes genuine local pandas/scipy/report-generation operations rather than inventing results.
