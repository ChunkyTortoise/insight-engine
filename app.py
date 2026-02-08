"""Insight Engine: Upload CSV/Excel, get instant dashboards, predictive models, and reports."""

from pathlib import Path

import pandas as pd
import streamlit as st

from insight_engine.attribution import run_all_models
from insight_engine.cleaner import clean_dataframe
from insight_engine.clustering import Clusterer
from insight_engine.dashboard_generator import generate_dashboard
from insight_engine.feature_lab import FeatureLab
from insight_engine.forecaster import Forecaster
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
    (
        tab_profile,
        tab_dashboard,
        tab_clean,
        tab_predict,
        tab_attribution,
        tab_report,
        tab_forecast,
        tab_cluster,
        tab_features,
    ) = st.tabs(
        [
            "📋 Profile",
            "📈 Dashboard",
            "🧹 Clean",
            "🔮 Predict",
            "📡 Attribution",
            "📄 Report",
            "📉 Forecast",
            "🔵 Cluster",
            "🧪 Features",
        ]
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
            width="stretch",
        )

    # Dashboard tab
    with tab_dashboard:
        st.subheader("Auto-Generated Dashboard")
        profile = profile_dataframe(df)
        charts = generate_dashboard(df, profile)
        for chart in charts:
            st.plotly_chart(chart.fig, width="stretch")

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
            st.dataframe(cleaned.head(50), width="stretch")

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
                    st.dataframe(result.summary, width="stretch")
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

    # Forecast tab
    with tab_forecast:
        st.subheader("Time Series Forecasting")
        ts_path = DEMO_DIR / "time_series_sales.csv"

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        use_ts_demo = False

        if ts_path.exists():
            use_ts = st.checkbox("Use time series demo dataset", value=True)
            if use_ts:
                ts_df = pd.read_csv(ts_path)
                forecast_col = "sales"
                use_ts_demo = True

        if not use_ts_demo:
            if not numeric_cols:
                st.info("No numeric columns available for forecasting.")
            else:
                forecast_col = st.selectbox("Select column to forecast:", numeric_cols)
                ts_df = df

        if numeric_cols or use_ts_demo:
            horizon = st.slider("Forecast horizon (steps)", 1, 30, 5)
            if st.button("Run Forecast"):
                series = ts_df[forecast_col].dropna().tolist()
                forecaster = Forecaster()
                comparison = forecaster.compare_forecasts(series, horizon=horizon)

                # Results table
                rows = []
                for method, result in comparison.results.items():
                    rows.append(
                        {
                            "Method": method,
                            "MAE": round(result.mae, 4),
                            "RMSE": round(result.rmse, 4),
                            "MAPE": round(result.mape, 4),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), width="stretch")
                st.success(f"Best method: **{comparison.best_method}** (MAE={comparison.best_mae:.4f})")

                # Plot actual vs predictions
                for method, result in comparison.results.items():
                    chart_df = pd.DataFrame(
                        {
                            "Step": list(range(len(series))) + list(range(len(series), len(series) + horizon)),
                            "Value": series + result.predictions,
                            "Type": ["Actual"] * len(series) + ["Predicted"] * horizon,
                        }
                    )
                    st.write(f"**{method}**")
                    st.line_chart(chart_df.pivot(index="Step", columns="Type", values="Value"))

    # Cluster tab
    with tab_cluster:
        st.subheader("Clustering Analysis")
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for clustering.")
        else:
            selected_cols = st.multiselect("Select numeric columns:", numeric_cols, default=numeric_cols[:3])

            if len(selected_cols) >= 2 and st.button("Run Clustering"):
                data = df[selected_cols].dropna().values
                clusterer = Clusterer()
                comparison = clusterer.compare(data)

                for method, result in comparison.results.items():
                    st.write(f"**{method}**: {result.n_clusters} clusters, silhouette={result.silhouette:.4f}")
                    st.write(f"Cluster sizes: {result.cluster_sizes}")

                st.success(f"Best method: **{comparison.best_method}** (silhouette={comparison.best_silhouette:.4f})")

                # Scatter plot of first 2 columns colored by best method labels
                best = comparison.results[comparison.best_method]
                clean_data = df[selected_cols].dropna()
                scatter_df = pd.DataFrame(
                    {
                        selected_cols[0]: clean_data[selected_cols[0]].values,
                        selected_cols[1]: clean_data[selected_cols[1]].values,
                        "Cluster": [str(lbl) for lbl in best.labels],
                    }
                )
                st.scatter_chart(scatter_df, x=selected_cols[0], y=selected_cols[1], color="Cluster")

    # Features tab
    with tab_features:
        st.subheader("Feature Engineering")
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if numeric_cols:
            scale_method = st.selectbox("Scaling method:", ["standard", "minmax", "robust"])
            scale_cols = st.multiselect("Columns to scale:", numeric_cols, default=numeric_cols[:2])

            if scale_cols and st.button("Scale Columns"):
                lab = FeatureLab()
                result = lab.scale(df, scale_cols, method=scale_method)
                st.write(f"New columns: {result.new_columns}")
                st.dataframe(result.data[scale_cols + result.new_columns].head(20), width="stretch")

        if st.button("Auto-Engineer Features"):
            lab = FeatureLab()
            result = lab.auto_engineer(df)
            st.success(result.description)
            st.write(f"New columns ({len(result.new_columns)}): {result.new_columns}")
            st.dataframe(result.data.head(20), width="stretch")


if __name__ == "__main__":
    main()
