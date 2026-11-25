"""
dashboard.py:
Streamlit dashboard for Online Shopping Cart Abandonment Prediction.
Reads precomputed outputs from outputs/ using pandas (no PySpark runtime).
Wide layout with 3 tabs:
1. Overview: KPI cards, abandonment rate by hour and by weekday.
2. Behavior: Views before cart and time to cart relationship to abandonment.
3. Model Performance: Model comparison table, ROC curve, confusion matrix, feature importances.
"""

import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure wide layout and title
st.set_page_config(
    page_title="Shopping Cart Abandonment Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

OUTPUT_DIR = "outputs"


@st.cache_data
def load_kpis():
    """Loads KPI summary data."""
    with open(os.path.join(OUTPUT_DIR, "kpis.json"), "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_hourly_data():
    """Loads abandonment rate by hour of day."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "abandonment_by_hour.csv"))


@st.cache_data
def load_weekday_data():
    """Loads abandonment rate by day of week."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "abandonment_by_weekday.csv"))


@st.cache_data
def load_behavior_views():
    """Loads abandonment by views before cart."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "behavior_views.csv"))


@st.cache_data
def load_behavior_time():
    """Loads abandonment by time to cart."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "behavior_time.csv"))


@st.cache_data
def load_model_metrics():
    """Loads model evaluation metrics comparison."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"))


@st.cache_data
def load_confusion_matrices():
    """Loads confusion matrix counts."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "confusion_matrices.csv"))


@st.cache_data
def load_feature_importances():
    """Loads feature importances."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"))


@st.cache_data
def load_roc_curve():
    """Loads ROC curve points."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "roc_curve.csv"))


# Header
st.title("🛒 Online Shopping Cart Abandonment Prediction")
st.caption(
    "PySpark Big Data Analytics & MLlib Pipeline on Retailrocket Dataset (Fedora 44 / Spark 4.3.0)"
)

# Load data
try:
    kpis = load_kpis()
    hourly_df = load_hourly_data()
    weekday_df = load_weekday_data()
    views_df = load_behavior_views()
    time_df = load_behavior_time()
    metrics_df = load_model_metrics()
    cm_df = load_confusion_matrices()
    fi_df = load_feature_importances()
    roc_df = load_roc_curve()
except Exception as e:
    st.error(f"Error loading exported data from {OUTPUT_DIR}/: {e}")
    st.info("Please make sure you have executed the full pipeline (src/01 to src/05) first.")
    st.stop()

# Three tabs as requested
tab_overview, tab_behavior, tab_model = st.tabs([
    "📊 Overview",
    "🔍 User Behavior",
    "🤖 Model Performance",
])

# ----------------- TAB 1: OVERVIEW -----------------
with tab_overview:
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Total Sessions",
            value=f"{kpis['total_sessions']:,}",
            help="Total browsing sessions with 30-min inactivity gap",
        )
    with col2:
        st.metric(
            label="Cart Sessions",
            value=f"{kpis['cart_sessions']:,}",
            help="Sessions that reached at least one addtocart event",
        )
    with col3:
        st.metric(
            label="Cart Abandonment Rate",
            value=f"{kpis['abandonment_rate']}%",
            delta=f"-{kpis['cart_conversion_rate']}% converted",
            delta_color="inverse",
            help="Percentage of cart sessions without a subsequent transaction",
        )
    with col4:
        st.metric(
            label="Active Date Range",
            value=kpis["start_date"][:10],
            delta=f"to {kpis['end_date'][:10]}",
            delta_color="off",
            help="Full dataset coverage period",
        )

    st.markdown("---")
    st.subheader("Temporal Abandonment Trends")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_hour = px.line(
            hourly_df,
            x="hour_of_day",
            y="abandonment_rate",
            markers=True,
            title="Cart Abandonment Rate by Hour of Day (UTC)",
            labels={"hour_of_day": "Hour of Day (0 - 23)", "abandonment_rate": "Abandonment Rate (%)"},
        )
        fig_hour.update_layout(yaxis_range=[50, 100], hovermode="x unified")
        st.plotly_chart(fig_hour, use_container_width=True)

    with chart_col2:
        # Sort weekdays Monday through Sunday
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_df["day_name"] = pd.Categorical(weekday_df["day_name"], categories=weekday_order, ordered=True)
        weekday_df = weekday_df.sort_values("day_name")

        fig_dow = px.bar(
            weekday_df,
            x="day_name",
            y="abandonment_rate",
            color="abandonment_rate",
            color_continuous_scale="Reds",
            title="Cart Abandonment Rate by Day of Week",
            labels={"day_name": "Day of Week", "abandonment_rate": "Abandonment Rate (%)"},
        )
        fig_dow.update_layout(yaxis_range=[50, 100], showlegend=False)
        st.plotly_chart(fig_dow, use_container_width=True)

# ----------------- TAB 2: BEHAVIOR -----------------
with tab_behavior:
    st.subheader("Pre-Cart User Journey & Intent Signals")
    st.markdown(
        """
        Features are computed **strictly prior to or at the first `addtocart`** event in the session,
        preventing any target data leakage while capturing user intent and hesitation signals.
        """
    )
    bcol1, bcol2 = st.columns(2)

    with bcol1:
        fig_views = px.bar(
            views_df,
            x="views_group",
            y="abandonment_rate",
            text="total_sessions",
            color="abandonment_rate",
            color_continuous_scale="Blues_r",
            title="Abandonment Rate by Views Before First Cart",
            labels={"views_group": "Product Views Before Cart", "abandonment_rate": "Abandonment Rate (%)"},
        )
        fig_views.update_traces(texttemplate="%{text} sessions", textposition="outside")
        fig_views.update_layout(yaxis_range=[0, 100], showlegend=False)
        st.plotly_chart(fig_views, use_container_width=True)

    with bcol2:
        fig_time = px.bar(
            time_df,
            x="time_group",
            y="abandonment_rate",
            text="total_sessions",
            color="abandonment_rate",
            color_continuous_scale="Viridis",
            title="Abandonment Rate by Time Elapsed to First Cart",
            labels={"time_group": "Time to First Cart", "abandonment_rate": "Abandonment Rate (%)"},
        )
        fig_time.update_traces(texttemplate="%{text} sessions", textposition="outside")
        fig_time.update_layout(yaxis_range=[0, 100], showlegend=False)
        st.plotly_chart(fig_time, use_container_width=True)

    st.info(
        "💡 **Key Insight**: Direct buyers (immediate cart addition with 0-1 prior views) exhibit distinct "
        "checkout velocity compared to research-heavy browsers exploring multiple catalog items before carting."
    )

# ----------------- TAB 3: MODEL PERFORMANCE -----------------
with tab_model:
    st.subheader("Model Evaluation & Comparison (Test Set)")
    st.caption("Temporal split: Earliest 80% sessions for training, latest 20% for testing.")

    # Formatted Metrics Table
    display_metrics = metrics_df[[
        "model", "auc_roc", "auc_pr", "f1_score", "precision", "recall",
        "abandoned_precision", "abandoned_recall", "abandoned_f1"
    ]].copy()
    display_metrics.columns = [
        "Model", "AUC-ROC", "AUC-PR", "Weighted F1", "Precision", "Recall",
        "Abandoned Prec.", "Abandoned Rec.", "Abandoned F1"
    ]
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    mcol1, mcol2 = st.columns(2)

    with mcol1:
        # ROC Curve Plot
        best_model_name = roc_df["best_model"].iloc[0] if "best_model" in roc_df.columns else "GBTClassifier"
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=roc_df["fpr"],
                y=roc_df["tpr"],
                mode="lines",
                name=f"{best_model_name}",
                line=dict(color="#1f77b4", width=3),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Chance (0.50)",
                line=dict(color="grey", dash="dash"),
            )
        )
        fig_roc.update_layout(
            title=f"ROC Curve - {best_model_name}",
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall)",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with mcol2:
        # Confusion Matrix Heatmap for best model
        best_cm_row = cm_df[cm_df["model"] == best_model_name].iloc[0]
        z_vals = [
            [best_cm_row["true_negative"], best_cm_row["false_positive"]],
            [best_cm_row["false_negative"], best_cm_row["true_positive"]],
        ]
        fig_cm = px.imshow(
            z_vals,
            text_auto=True,
            x=["Predicted Converted (0)", "Predicted Abandoned (1)"],
            y=["Actual Converted (0)", "Actual Abandoned (1)"],
            color_continuous_scale="Blues",
            title=f"Confusion Matrix ({best_model_name})",
        )
        fig_cm.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    # Feature Importance Chart
    st.subheader("Feature Importances")
    gbt_fi = fi_df[fi_df["model"] == "GBTClassifier"].sort_values("importance", ascending=True)

    fig_fi = px.bar(
        gbt_fi,
        x="importance",
        y="feature",
        orientation="h",
        title="GBTClassifier Relative Feature Importances",
        labels={"importance": "Importance Score", "feature": "Feature"},
        color="importance",
        color_continuous_scale="Teal",
    )
    fig_fi.update_layout(showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

