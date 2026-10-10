

"""AURA — autonomous data analysis platform."""

from __future__ import annotations



import hashlib



import pandas as pd

import streamlit as st



from config.settings import (

    APP_VERSION,

    OPENAI_MODEL,

    STYLE_PATH,

    ensure_directories,

    get_api_key,

)

from src.ai_agent import DataAnalysisAgent

from src.analyzer import (

    categorical_analysis,

    correlation_matrix,

    data_types_table,

    dataset_overview,

    generate_findings,

    missing_values_table,

    outlier_summary,

    statistics_table,

    top_correlations,

)

from src.data_cleaner import (

    CleaningError,

    clean_dataset,

    convert_dtype,

    remove_duplicates,

    rename_columns,

)

from src.data_loader import (

    DataLoadError,

    dataframe_to_csv_bytes,

    dataframe_to_excel_bytes,

    load_dataframe,

)

from src.report_generator import generate_pdf_report

from src.utils import get_column_groups, safe_filename

from src.visualizer import (

    AGGREGATIONS,

    CHART_TYPES,

    ChartError,

    correlation_heatmap,

    create_chart,

    distribution_chart,

    missing_values_chart,

)





# ---------------------------------------------------------

# PAGE CONFIGURATION

# ---------------------------------------------------------



st.set_page_config(

    page_title="AURA | Autonomous Analytics",

    page_icon="✦",

    layout="wide",

    initial_sidebar_state="expanded",

)



ensure_directories()



if STYLE_PATH.exists():

    st.markdown(

        f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>",

        unsafe_allow_html=True,

    )





# ---------------------------------------------------------

# SESSION STATE

# ---------------------------------------------------------



DEFAULT_STATE = {

    "original_df": None,

    "working_df": None,

    "dataset_name": None,

    "upload_token": None,

    "chat": [],

    "last_chart": None,

    "pdf_report": None,

    "dataset_revision": 0,

    "analysis_cache": {},

}



for key, value in DEFAULT_STATE.items():

    st.session_state.setdefault(key, value)





# ---------------------------------------------------------

# COMMON UI FUNCTIONS

# ---------------------------------------------------------



def header(title, subtitle):

    st.markdown(

        f'<div class="aura-page-header">'

        f'<h1>{title}</h1><p>{subtitle}</p></div>',

        unsafe_allow_html=True,

    )





def metric(label, value, sub="", tone=""):

    st.markdown(

        f'<div class="aura-metric">'

        f'<div class="aura-metric-label">{label}</div>'

        f'<div class="aura-metric-value {tone}">{value}</div>'

        f'<div class="aura-metric-sub">{sub}</div>'

        f'</div>',

        unsafe_allow_html=True,

    )





def df():

    return st.session_state.working_df





def must_df():

    if df() is None:

        st.markdown(

            '<div class="aura-empty">'

            '<div class="aura-empty-mark">✦</div>'

            '<h3>Connect a dataset to begin</h3>'

            '<p>Use the uploader in the sidebar to activate every AURA workspace.</p>'

            '</div>',

            unsafe_allow_html=True,

        )

        return None



    return df()





# ---------------------------------------------------------

# ANALYSIS CACHE

# ---------------------------------------------------------



def cached(name, factory):

    """

    Cache analysis results for the current dataset revision.

    Cache is cleared whenever the working dataset changes.

    """

    key = f"{st.session_state.dataset_revision}:{name}"



    if key not in st.session_state.analysis_cache:

        st.session_state.analysis_cache[key] = factory()



    return st.session_state.analysis_cache[key]





def overview(data):

    return cached("overview", lambda: dataset_overview(data))





def missing_table(data):

    return cached("missing", lambda: missing_values_table(data))





def reset_analysis():
    st.session_state.analysis_cache = {}
    st.session_state.pop("export_cache", None)


def lazy_export(data, export_key, filename, mime, formatter):
    """Prepare large CSV/Excel exports only after an explicit user click."""
    cache = st.session_state.setdefault("export_cache", {})
    revision_key = f"{st.session_state.dataset_revision}:{export_key}"

    if st.button(
        f"Prepare {filename}",
        key=f"prepare_{export_key}",
        use_container_width=True,
    ):
        with st.spinner(f"Preparing {filename}..."):
            cache[revision_key] = formatter(data)

    if revision_key in cache:
        st.download_button(
            f"Download {filename}",
            cache[revision_key],
            filename,
            mime,
            use_container_width=True,
            key=f"download_{export_key}",
        )





# ---------------------------------------------------------

# DATASET UPLOAD

# ---------------------------------------------------------



def ingest(upload):

    """

    Load a dataset only when a new upload is detected.



    Streamlit provides a file_id for uploaded files in supported

    versions. When available, use it to avoid hashing the entire

    uploaded file on every application rerun.

    """



    file_id = getattr(upload, "file_id", None)



    if file_id:

        token = f"{file_id}:{upload.name}:{upload.size}"

        content = None

    else:

        content = upload.getvalue()

        token = hashlib.sha256(content).hexdigest()



    if token == st.session_state.upload_token:

        return



    try:

        if content is None:

            content = upload.getvalue()



        result = load_dataframe(upload.name, content)



        st.session_state.original_df = result.df

        st.session_state.working_df = result.df.copy(deep=True)



        st.session_state.dataset_name = result.name

        st.session_state.upload_token = token



        st.session_state.chat = []

        st.session_state.pdf_report = None

        st.session_state.last_chart = None



        st.session_state.dataset_revision += 1

        reset_analysis()



        st.success(

            f"Connected {result.name}: "

            f"{len(result.df):,} rows × "

            f"{len(result.df.columns):,} columns."

        )



    except DataLoadError as exc:

        st.error(str(exc))





# ---------------------------------------------------------

# COMMIT CLEANING RESULT

# ---------------------------------------------------------



def commit(result):

    st.session_state.working_df = result.df



    st.session_state.dataset_revision += 1

    st.session_state.pdf_report = None

    st.session_state.last_chart = None



    reset_analysis()



    st.success(result.summary)





# ---------------------------------------------------------

# SIDEBAR

# ---------------------------------------------------------



with st.sidebar:



    st.markdown(

        '<div class="aura-brand">'

        '<div class="aura-brand-name">✦ AURA</div>'

        '</div>'

        '<div class="aura-brand-tag">'

        'AUTONOMOUS DATA INTELLIGENCE'

        '</div>',

        unsafe_allow_html=True,

    )



    st.markdown(

        '<div class="aura-sidebar-label">Workspace</div>',

        unsafe_allow_html=True,

    )



    page = st.radio(

        "Navigation",

        [

            "Executive Dashboard",

            "Data Explorer",

            "AI Agent Workspace",

            "Data Cleaning Studio",

            "Visualization Studio",

            "Reports Center",

            "Settings",

        ],

        label_visibility="collapsed",

        key="main_navigation",

    )



    uploaded = st.file_uploader(

        "Upload dataset",

        type=["csv", "tsv", "xlsx", "xlsm"],

        key="global_dataset_uploader",

        label_visibility="collapsed",

    )



    if uploaded is not None:

        ingest(uploaded)



    st.markdown(

        '<div class="aura-sidebar-label">Active dataset</div>',

        unsafe_allow_html=True,

    )



    if df() is not None:

        st.caption(

            st.session_state.dataset_name or "Dataset connected"

        )

        st.caption(

            f"{len(df()):,} rows · "

            f"{len(df().columns):,} columns"

        )

    else:

        st.caption("No dataset connected")



    mode = (

        "OpenAI connected"

        if get_api_key()

        else "Local analysis fallback"

    )



    badge_class = "llm" if get_api_key() else "rule"



    st.markdown(

        f'<span class="aura-badge {badge_class}">'

        f'<span class="aura-dot"></span>{mode}</span>',

        unsafe_allow_html=True,

    )



    if df() is not None:



        if st.button(

            "Restore original dataset",

            use_container_width=True,

        ):

            st.session_state.working_df = (

                st.session_state.original_df.copy(deep=True)

            )



            st.session_state.dataset_revision += 1

            st.session_state.pdf_report = None

            st.session_state.last_chart = None



            reset_analysis()



            st.success("Original data restored.")





# =========================================================

# EXECUTIVE DASHBOARD

# =========================================================



if page == "Executive Dashboard":


    header(
        "Autonomous intelligence, grounded in your data.",
        "Connect a dataset and AURA will surface quality signals, patterns, and next actions.",
    )


    if df() is not None:

        info = overview(df())


        # =====================================================
        # 1. DATASET OVERVIEW
        # =====================================================

        st.markdown("## Dataset Overview")
        st.caption("A quick summary of the currently active dataset.")


        cards = st.columns(5)

        values = [
            ("Records", f"{info['rows']:,}", "active rows", ""),
            (
                "Features",
                info["columns"],
                f"{info['numeric_columns']} numeric",
                "",
            ),
            (
                "Missing",
                f"{info['missing_values']:,}",
                f"{info['missing_pct']}% cells",
                "warn",
            ),
            (
                "Duplicates",
                f"{info['duplicate_rows']:,}",
                f"{info['duplicate_pct']}% rows",
                "",
            ),
            (
                "Quality",
                f"{info['quality_score']}/100",
                info["quality_label"],
                "good" if info["quality_score"] >= 75 else "warn",
            ),
        ]

        for col, item in zip(cards, values):
            with col:
                metric(*item)


        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

        detail_cards = st.columns(3)

        detail_values = [
            (
                "Numeric Columns",
                f"{info['numeric_columns']:,}",
                "numeric features",
            ),
            (
                "Categorical Columns",
                f"{info['categorical_columns']:,}",
                "categorical features",
            ),
            (
                "Datetime Columns",
                f"{info['datetime_columns']:,}",
                "date/time features",
            ),
        ]

        for col, (title, value, subtitle) in zip(detail_cards, detail_values):
            with col:
                metric(title, value, subtitle, "")


        # =====================================================
        # 2. WHAT AURA FOUND
        # =====================================================

        st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
        st.markdown("## What AURA Found")
        st.caption("Key quality signals and observations detected from the dataset.")


        findings = cached(
            "findings",
            lambda: generate_findings(df(), 8),
        )


        for finding in findings:
            st.markdown(f"- {finding}")




        # =====================================================
        # 3. DATA VISUALIZATIONS
        # =====================================================

        st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
        st.markdown("## Data Visualizations")
        st.caption("Visual summaries of the most important patterns found in the dataset.")


        st.markdown("### Missing Values by Column")

        st.plotly_chart(
            missing_values_chart(missing_table(df())),
            use_container_width=True,
        )


        nums = get_column_groups(df())["numeric"]

        if nums:
            st.markdown(f"### Distribution of {nums[0]}")

            st.plotly_chart(
                distribution_chart(df(), nums[0]),
                use_container_width=True,
            )


# =========================================================

# DATA EXPLORER

# =========================================================



elif page == "Data Explorer":



    header(

        "Data Explorer",

        "Filter, sort, profile and clean the working copy. Your original upload stays intact.",

    )



    data = must_df()



    if data is not None:



        section = st.radio(

            "Data Explorer section",

            ["Explore", "Profiles", "Quality", "Cleaning"],

            horizontal=True,

            label_visibility="collapsed",

            key="data_explorer_section",

        )



        cols = data.columns.tolist()



        # -------------------------------------------------

        # EXPLORE SECTION

        # -------------------------------------------------



        if section == "Explore":



            c1, c2, c3 = st.columns(3)



            with c1:

                chosen = st.multiselect(

                    "Columns",

                    cols,

                    default=cols[:min(12, len(cols))],

                )



            with c2:

                filter_col = st.selectbox(

                    "Filter column",

                    ["None"] + cols,

                )



            with c3:

                sort_col = st.selectbox(

                    "Sort by",

                    ["None"] + cols,

                )



            view = data[chosen] if chosen else data



            if filter_col != "None":



                if pd.api.types.is_numeric_dtype(data[filter_col]):



                    low = float(data[filter_col].min())

                    high = float(data[filter_col].max())



                    if low < high:

                        bounds = st.slider(

                            "Range",

                            min_value=low,

                            max_value=high,

                            value=(low, high),

                        )



                        view = view[

                            data[filter_col].between(*bounds)

                        ]



                else:



                    options = (

                        data[filter_col]

                        .dropna()

                        .astype(str)

                        .value_counts()

                        .head(500)

                        .index.tolist()

                    )



                    values = st.multiselect(

                        "Values (top 500 by frequency)",

                        options,

                        default=options[:min(25, len(options))],

                    )



                    view = view[

                        data[filter_col].astype(str).isin(values)

                    ]



            if sort_col != "None":

                view = view.sort_values(

                    sort_col,

                    na_position="last",

                )



            st.caption(

                f"Showing {len(view):,} of {len(data):,} rows"

            )



            st.dataframe(

                view,

                use_container_width=True,

                hide_index=True,

                height=500,

            )



            export_signature = (
                "filtered:"
                + str(st.session_state.dataset_revision)
                + ":"
                + "|".join(chosen)
                + ":"
                + str(filter_col)
                + ":"
                + str(sort_col)
            )
            if filter_col != "None" and pd.api.types.is_numeric_dtype(data[filter_col]):
                export_signature += f":{bounds}"
            elif filter_col != "None":
                export_signature += ":" + "|".join(values)

            lazy_export(
                view,
                export_signature,
                "aura_filtered.csv",
                "text/csv",
                dataframe_to_csv_bytes,
            )



        # -------------------------------------------------

        # PROFILES SECTION

        # -------------------------------------------------



        elif section == "Profiles":



            st.dataframe(

                cached(

                    "types",

                    lambda: data_types_table(data),

                ),

                use_container_width=True,

                hide_index=True,

            )



            try:

                stats = cached(

                    "stats",

                    lambda: statistics_table(data).round(4),

                )



                st.dataframe(

                    stats,

                    use_container_width=True,

                    hide_index=True,

                )



            except Exception as exc:

                st.info(str(exc))



            try:

                cats = cached(

                    "categories",

                    lambda: categorical_analysis(data),

                )



                st.dataframe(

                    cats["summary"],

                    use_container_width=True,

                    hide_index=True,

                )



            except Exception as exc:

                st.info(

                    f"Categorical analysis unavailable: {exc}"

                )



        # -------------------------------------------------

        # QUALITY SECTION

        # -------------------------------------------------



        elif section == "Quality":



            st.dataframe(

                missing_table(data),

                use_container_width=True,

                hide_index=True,

            )



            try:

                st.dataframe(

                    cached(

                        "outliers",

                        lambda: outlier_summary(data, "iqr"),

                    ),

                    use_container_width=True,

                    hide_index=True,

                )



            except Exception as exc:

                st.info(str(exc))



            try:

                correlation = cached(

                    "correlation",

                    lambda: correlation_matrix(data),

                )



                st.plotly_chart(

                    correlation_heatmap(correlation),

                    use_container_width=True,

                )



                st.dataframe(

                    cached(

                        "top_correlations",

                        lambda: top_correlations(data, min_abs=0.3),

                    ),

                    use_container_width=True,

                    hide_index=True,

                )



            except Exception as exc:

                st.info(str(exc))



        # -------------------------------------------------

        # CLEANING SECTION

        # -------------------------------------------------



        elif section == "Cleaning":



            st.warning(

                "Each action updates only the working copy; "

                "use Restore original dataset to undo all actions."

            )



            a, b, c = st.columns(3)



            with a:



                if st.button(

                    "Remove duplicate rows",

                    use_container_width=True,

                ):

                    try:

                        commit(remove_duplicates(data))

                    except CleaningError as exc:

                        st.error(str(exc))



            with b:



                strategy = st.selectbox(

                    "Missing-data cleaning",

                    ["none", "auto", "drop_rows"],

                )



                if st.button(

                    "Run standard cleaning",

                    use_container_width=True,

                ):

                    try:

                        commit(

                            clean_dataset(

                                data,

                                missing_strategy=strategy,

                            )

                        )

                    except CleaningError as exc:

                        st.error(str(exc))



            with c:



                col = st.selectbox(

                    "Convert column",

                    data.columns,

                )



                kind = st.selectbox(

                    "To type",

                    [

                        "float",

                        "integer",

                        "string",

                        "category",

                        "datetime",

                        "boolean",

                    ],

                )



                if st.button("Convert type"):

                    try:

                        commit(

                            convert_dtype(

                                data,

                                col,

                                kind,

                                errors="coerce",

                            )

                        )

                    except CleaningError as exc:

                        st.error(str(exc))



            old = st.selectbox(

                "Column to rename",

                data.columns,

            )



            new = st.text_input("New column name")



            if st.button("Rename column") and new.strip():



                try:

                    commit(

                        rename_columns(

                            data,

                            {old: new.strip()},

                        )

                    )

                except CleaningError as exc:

                    st.error(str(exc))



            st.markdown("#### Export cleaned working copy")



            download_csv, download_excel = st.columns(2)



            with download_csv:
                lazy_export(
                    df(),
                    "explorer_clean_csv",
                    "aura_cleaned.csv",
                    "text/csv",
                    dataframe_to_csv_bytes,
                )

            with download_excel:
                lazy_export(
                    df(),
                    "explorer_clean_excel",
                    "aura_cleaned.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    dataframe_to_excel_bytes,
                )





# =========================================================

# AI AGENT WORKSPACE

# =========================================================



elif page == "AI Agent Workspace":



    header(

        "AURA AI Agent",

        "Understand → plan → select registered tools → execute → evaluate → respond.",

    )



    data = must_df()



    if data is not None:



        st.caption(

            "Registered tools: dataset information, missing values, "

            "statistics, outliers, correlations, cleaning, duplicate "

            "removal, visualizations and PDF reports."

        )



        for message in st.session_state.chat:



            with st.chat_message(message["role"]):



                st.markdown(message["content"])



                if message.get("trace"):



                    with st.expander("Execution trace"):

                        st.json(message["trace"])



        with st.form(

            "aura_agent_prompt_form",

            clear_on_submit=True,

        ):



            prompt = st.text_input(

                "Ask AURA",

                placeholder="Ask AURA to analyze the active dataset...",

            )



            submitted = st.form_submit_button(

                "Ask AURA",

                type="primary",

            )



        if submitted and prompt.strip():



            st.session_state.chat.append(

                {

                    "role": "user",

                    "content": prompt.strip(),

                }

            )



            with st.chat_message("user"):

                st.markdown(prompt.strip())



            with st.chat_message("assistant"):



                with st.spinner(

                    "AURA is planning and executing registered analysis tools..."

                ):



                    result = DataAnalysisAgent(

                        data,

                        st.session_state.dataset_name or "Dataset",

                    ).run(prompt.strip())



                st.markdown(result.response)



                with st.expander(

                    "Plan & tool trace",

                    expanded=True,

                ):



                    st.write(

                        "Plan: " + " → ".join(result.plan)

                    )



                    st.json(result.trace)



                if result.cleaned_df is not None:



                    st.session_state.working_df = result.cleaned_df



                    st.session_state.dataset_revision += 1

                    st.session_state.pdf_report = None

                    st.session_state.last_chart = None



                    reset_analysis()



                    st.info(

                        "The produced cleaned dataset is now the working copy."

                    )



                if result.chart is not None:



                    st.session_state.last_chart = result.chart



                    st.plotly_chart(

                        result.chart,

                        use_container_width=True,

                    )



                    st.caption(

                        "This agent-created chart is also available "

                        "in Visualization Studio for the current session."

                    )



                if result.report_bytes:



                    st.download_button(

                        "Download agent PDF report",

                        result.report_bytes,

                        "aura_agent_report.pdf",

                        "application/pdf",

                    )



            st.session_state.chat.append(

                {

                    "role": "assistant",

                    "content": result.response,

                    "trace": result.trace,

                }

            )





# =========================================================

# DATA CLEANING STUDIO

# =========================================================



elif page == "Data Cleaning Studio":

    header(
        "Data Cleaning Studio",
        "Build a safer, analysis-ready dataset while keeping your original upload untouched.",
    )

    data = must_df()

    if data is not None:
        info = overview(data)

        # Summary metrics
        before, after, actions = st.columns(3)

        with before:
            metric("Working records", f"{len(data):,}", "current copy")

        with after:
            metric(
                "Missing cells",
                f"{info['missing_values']:,}",
                "before any new action",
                "warn" if info["missing_values"] else "good",
            )

        with actions:
            metric(
                "Duplicate rows",
                f"{info['duplicate_rows']:,}",
                "available to remove",
            )

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        st.markdown("## Cleaning workspace")
        st.caption(
            "Choose an action below. Each operation updates only the working copy; "
            "you can restore the original dataset from the sidebar."
        )

        # Balanced two-column workspace
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("### Missing data & duplicates")
            st.caption("Handle incomplete records and repeated rows.")

            missing = st.selectbox(
                "Missing-value strategy",
                ["none", "auto", "drop_rows"],
                key="studio_missing_strategy",
            )

            if st.button(
                "Apply standard cleaning",
                type="primary",
                use_container_width=True,
                key="studio_apply_cleaning",
            ):
                try:
                    commit(
                        clean_dataset(
                            data,
                            missing_strategy=missing,
                        )
                    )
                except CleaningError as exc:
                    st.error(str(exc))

            st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

            if st.button(
                "Remove duplicate rows",
                use_container_width=True,
                key="studio_remove_duplicates",
            ):
                try:
                    commit(remove_duplicates(data))
                except CleaningError as exc:
                    st.error(str(exc))

        with right:
            st.markdown("### Column tools")
            st.caption("Change a column's data type or rename it.")

            target = st.selectbox(
                "Column to convert",
                data.columns,
                key="studio_convert_col",
            )

            dtype = st.selectbox(
                "Target data type",
                [
                    "float",
                    "integer",
                    "string",
                    "category",
                    "datetime",
                    "boolean",
                ],
                key="studio_convert_type",
            )

            if st.button(
                "Convert selected column",
                use_container_width=True,
                key="studio_convert_button",
            ):
                try:
                    commit(
                        convert_dtype(
                            data,
                            target,
                            dtype,
                            errors="coerce",
                        )
                    )
                except CleaningError as exc:
                    st.error(str(exc))

            st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

            original = st.selectbox(
                "Column to rename",
                data.columns,
                key="studio_rename_old",
            )

            replacement = st.text_input(
                "New column name",
                key="studio_rename_new",
            )

            if st.button(
                "Rename selected column",
                use_container_width=True,
                key="studio_rename_button",
            ) and replacement.strip():
                try:
                    commit(
                        rename_columns(
                            data,
                            {original: replacement.strip()},
                        )
                    )
                except CleaningError as exc:
                    st.error(str(exc))

        # Export area
        st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
        st.markdown("## Export cleaned working copy")
        st.caption("Prepare the format you need, then download the updated working dataset.")

        export_csv, export_excel = st.columns(2, gap="large")

        with export_csv:
            lazy_export(
                df(),
                "studio_clean_csv",
                "aura_cleaned.csv",
                "text/csv",
                dataframe_to_csv_bytes,
            )

        with export_excel:
            lazy_export(
                df(),
                "studio_clean_excel",
                "aura_cleaned.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                dataframe_to_excel_bytes,
            )


# VISUALIZATION STUDIO

# =========================================================



elif page == "Visualization Studio":



    header(

        "Visualization Studio",

        "Build professional, interactive Plotly charts without writing code.",

    )



    data = must_df()



    if data is not None:



        cols = data.columns.tolist()



        a, b, c, d = st.columns(4)



        with a:

            kind = st.selectbox(

                "Chart type",

                CHART_TYPES,

            )



        with b:

            x = st.selectbox(

                "X / category",

                cols,

            )



        with c:

            y = st.selectbox(

                "Y / value",

                ["None"] + cols,

            )



        with d:

            color = st.selectbox(

                "Color / group",

                ["None"] + cols,

            )



        aggregation = st.selectbox(

            "Aggregation",

            ["none"] + AGGREGATIONS,

            index=1,

        )



        if st.button(

            "Create interactive chart",

            type="primary",

        ):



            try:

                st.session_state.last_chart = create_chart(

                    data,

                    kind,

                    x=x,

                    y=None if y == "None" else y,

                    color=None if color == "None" else color,

                    agg=aggregation,

                )



            except ChartError as exc:

                st.error(str(exc))



        if st.session_state.last_chart is not None:



            st.plotly_chart(

                st.session_state.last_chart,

                use_container_width=True,

            )





# =========================================================

# REPORTS CENTER

# =========================================================



elif page == "Reports Center":



    header(

        "Intelligence Reports",

        "Generate a professional PDF from actual computations on your working dataset.",

    )



    data = must_df()



    if data is not None:



        st.markdown(

            '<div class="aura-card">'

            '<h4>Included in every report</h4>'

            '<p>Executive summary, quality metrics, computed findings, '

            'missing-data details, numeric statistics, and notable correlations.</p>'

            '</div>',

            unsafe_allow_html=True,

        )



        if st.button(

            "Generate PDF report",

            type="primary",

        ):



            with st.spinner("Composing the report..."):



                try:

                    st.session_state.pdf_report = generate_pdf_report(

                        data,

                        st.session_state.dataset_name or "Dataset",

                    )



                except Exception as exc:

                    st.error(f"Could not generate report: {exc}")



        if st.session_state.pdf_report:



            st.download_button(

                "Download AURA PDF report",

                st.session_state.pdf_report,

                f"{safe_filename(st.session_state.dataset_name or 'aura')}_report.pdf",

                "application/pdf",

            )





# =========================================================

# SETTINGS

# =========================================================



else:



    header(

        "Settings",

        "Runtime configuration and local workspace controls.",

    )



    st.markdown(

        '<div class="aura-card"><h4>AI provider</h4></div>',

        unsafe_allow_html=True,

    )



    st.write("OpenAI model:", f"`{OPENAI_MODEL}`")



    st.write(

        "API key status:",

        "Configured from .env"

        if get_api_key()

        else "Not configured — local rule-based tools remain fully functional.",

    )



    st.caption(

        "Set OPENAI_API_KEY and optionally OPENAI_MODEL in .env. "

        "Keys are never displayed or persisted by AURA."

    )



    if st.button("Clear current chat"):

        st.session_state.chat = []

        st.success("Chat history cleared.")





# =========================================================

# FOOTER

# =========================================================



st.markdown(

    f"<hr><p style='text-align:center;color:#8E8A9C;"

    f"font-size:11px;letter-spacing:1px'>"

    f"✦ AURA • AUTONOMOUS DATA INTELLIGENCE • v{APP_VERSION}</p>",

    unsafe_allow_html=True,

)