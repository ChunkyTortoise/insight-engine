"""Insight Engine: Upload CSV/Excel, get instant dashboards, predictive models, and reports."""

from pathlib import Path

import pandas as pd
import streamlit as st

from insight_engine.attribution import run_all_models
from insight_engine.cleaner import clean_dataframe
from insight_engine.dashboard_generator import generate_dashboard
from insight_engine.predictor import train_model
from insight_engine.profiler import profile_dataframe
from insight_engine.report_generator import generate_report, report_to_markdown

DEMO_DIR = Path(__file__).parent / "demo_data"


def load_demo_datasets() -> dict[str, Path]:
    """Discover available demo datasets."""
    datasets = {}
    for f in DEMO_DIR.glob("*.csv"):
        name = f.stem.replace("_", " ").title()
        datasets[name] = f
    return datasets


def load_file(uploaded_file) -> pd.DataFrame:
    """Load uploaded file into DataFrame."""
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def main():
    st.set_page_config(page_title="Insight Engine", page_icon="📊", layout="wide")
    st.title("📊 Insight Engine")
    st.caption("Upload CSV/Excel → get instant dashboards, predictive models, and reports")

    # Sidebar: data source
    st.sidebar.header("Data Source")
    source = st.sidebar.radio("Choose source:", ["Upload File", "Demo Dataset"])

    df = None

    if source == "Demo Dataset":
        demos = load_demo_datasets()
        if not demos:
            st.warning("No demo datasets found. Run `python demo_data/generate_demo_data.py` first.")
            return
        selected = st.sidebar.selectbox("Select dataset:", list(demos.keys()))
        df = pd.read_csv(demos[selected])
        st.sidebar.success(f"Loaded {selected} ({len(df):,} rows)")
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if uploaded:
            df = load_file(uploaded)
            st.sidebar.success(f"Loaded {uploaded.name} ({len(df):,} rows)")

    if df is None:
        st.info("Select a demo dataset or upload a file to get started.")
        return

    # Tabs
    tab_profile, tab_dashboard, tab_clean, tab_predict, tab_attribution, tab_report = st.tabs(
        ["📋 Profile", "📈 Dashboard", "🧹 Clean", "🔮 Predict", "📡 Attribution", "📄 Report"]
    )

    # Profile tab
    with tab_profile:
        st.subheader("Data Profile")
        profile = profile_dataframe(df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{profile.row_count:,}")
        col2.metric("Columns", profile.column_count)
        col3.metric("Duplicates", f"{profile.duplicate_rows:,}")
        col4.metric("Memory", f"{profile.memory_usage_mb:.1f} MB")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Column": c.name,
                        "Type": c.dtype,
                        "Non-Null": c.non_null_count,
                        "Null%": c.null_pct,
                        "Unique": c.unique_count,
                        "Outliers": c.outlier_count,
                    }
                    for c in profile.columns
                ]
            ),
            use_container_width=True,
        )

    # Dashboard tab
    with tab_dashboard:
        st.subheader("Auto-Generated Dashboard")
        profile = profile_dataframe(df)
        charts = generate_dashboard(df, profile)
        for chart in charts:
            st.plotly_chart(chart.fig, use_container_width=True)

    # Clean tab
    with tab_clean:
        st.subheader("Data Cleaning")
        col1, col2 = st.columns(2)
        do_dedup = col1.checkbox("Remove duplicates", value=True)
        do_standardize = col1.checkbox("Standardize columns", value=True)
        do_impute = col2.checkbox("Impute missing values", value=True)
        strategy = col2.selectbox("Imputation strategy", ["smart", "mean", "drop"])

        if st.button("Clean Data"):
            cleaned, report = clean_dataframe(
                df,
                dedup=do_dedup,
                standardize=do_standardize,
                impute=do_impute,
                impute_strategy=strategy,
            )
            st.success(f"Cleaning complete: {report.original_rows} → {report.cleaned_rows} rows")
            for op in report.operations:
                st.write(f"- {op}")
            st.dataframe(cleaned.head(50), use_container_width=True)

    # Predict tab
    with tab_predict:
        st.subheader("Predictive Modeling")
        target = st.selectbox("Select target column:", df.columns.tolist())
        if st.button("Train Model"):
            with st.spinner("Training..."):
                result = train_model(df, target)
            st.success(f"Task: {result.task_type} | Target: {result.target_column}")
            st.json(result.metrics)
            st.bar_chart(
                pd.Series(result.feature_importances).head(15),
            )

    # Attribution tab
    with tab_attribution:
        st.subheader("Marketing Attribution")
        required_cols = {"user_id", "channel", "timestamp"}
        if required_cols.issubset(set(df.columns)):
            converted_col = st.selectbox(
                "Conversion column (1 = converted):",
                [c for c in df.columns if c not in required_cols],
            )
            if st.button("Run Attribution"):
                conversions = set(df[df[converted_col] == 1]["user_id"].unique())
                results = run_all_models(df, conversions)
                for model, result in results.items():
                    st.write(f"**{model.value}** ({result.total_conversions} conversions)")
                    st.dataframe(result.summary, use_container_width=True)
        else:
            st.info(f"Attribution requires columns: {required_cols}. Try the Marketing Touchpoints demo dataset.")

    # Report tab
    with tab_report:
        st.subheader("Generate Report")
        profile = profile_dataframe(df)
        report = generate_report(profile, title="Data Analysis Report")
        markdown = report_to_markdown(report)
        st.markdown(markdown)
        st.download_button("Download Report (Markdown)", markdown, "report.md", "text/markdown")


if __name__ == "__main__":
    main()
