"""
Streamlit Credit Risk Dashboard
Portfolio analytics, risk segmentation, expected loss, drift alerts, and stress testing.
"""

import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────
st.sidebar.title("🏦 Credit Risk Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Portfolio Overview", "Risk Segmentation", "Expected Loss",
     "Feature Importance", "Stress Testing", "Drift Monitoring"]
)


@st.cache_data
def load_sample_data():
    """Load or generate sample data for the dashboard."""
    np.random.seed(42)
    n = 2000

    data = pd.DataFrame({
        "customer_id": [f"CUST_{i:05d}" for i in range(n)],
        "default_probability": np.random.beta(2, 20, n).round(4),
        "credit_limit": np.random.lognormal(9, 0.8, n).clip(1000, 100000).round(-2),
        "current_balance": np.random.lognormal(7, 1.2, n).clip(0, 80000).round(2),
        "avg_utilization": np.random.beta(3, 7, n).round(4),
        "late_payment_count": np.random.poisson(1, n),
        "delinquency_severity": np.random.choice([0, 1, 2, 3, 4], n, p=[0.6, 0.15, 0.12, 0.08, 0.05]),
        "income_to_debt_ratio": np.random.lognormal(1, 0.5, n).clip(0.1, 20).round(2),
        "avg_merchant_risk": np.random.beta(3, 7, n).round(4),
        "credit_score": np.random.normal(680, 80, n).clip(300, 850).astype(int),
        "state": np.random.choice(["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA"], n),
    })

    data["risk_segment"] = pd.cut(
        data["default_probability"],
        bins=[0, 0.05, 0.15, 0.3, 1.0],
        labels=["Low", "Medium", "High", "Critical"]
    )
    data["expected_loss"] = (
        data["default_probability"] * 0.45 * data["current_balance"]
    ).round(2)

    return data


data = load_sample_data()


# ── Portfolio Overview ───────────────────────────────
if page == "Portfolio Overview":
    st.title("📊 Portfolio Overview")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(data):,}")
    col2.metric("Avg Default Prob", f"{data['default_probability'].mean():.2%}")
    col3.metric("Total Exposure", f"${data['current_balance'].sum():,.0f}")
    col4.metric("Portfolio Expected Loss", f"${data['expected_loss'].sum():,.0f}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Default Probability Distribution")
        fig = px.histogram(
            data, x="default_probability", nbins=50,
            color_discrete_sequence=["#2563eb"],
            labels={"default_probability": "Default Probability"},
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Risk Segment Breakdown")
        segment_counts = data["risk_segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Count"]
        fig = px.pie(
            segment_counts, names="Segment", values="Count",
            color="Segment",
            color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b",
                                "High": "#ef4444", "Critical": "#7c2d12"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Credit Score vs Default Probability")
    fig = px.scatter(
        data.sample(500), x="credit_score", y="default_probability",
        color="risk_segment", opacity=0.6,
        color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b",
                            "High": "#ef4444", "Critical": "#7c2d12"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ── Risk Segmentation ────────────────────────────────
elif page == "Risk Segmentation":
    st.title("🎯 Risk Segmentation")
    st.markdown("---")

    seg = data.groupby("risk_segment").agg(
        count=("customer_id", "size"),
        avg_pd=("default_probability", "mean"),
        avg_balance=("current_balance", "mean"),
        total_el=("expected_loss", "sum"),
        avg_utilization=("avg_utilization", "mean"),
    ).reset_index()

    st.dataframe(seg.style.format({
        "avg_pd": "{:.2%}", "avg_balance": "${:,.0f}",
        "total_el": "${:,.0f}", "avg_utilization": "{:.2%}",
    }), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(seg, x="risk_segment", y="total_el",
                     color="risk_segment", title="Expected Loss by Segment",
                     color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b",
                                         "High": "#ef4444", "Critical": "#7c2d12"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(seg, x="risk_segment", y="avg_pd",
                     color="risk_segment", title="Average PD by Segment",
                     color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b",
                                         "High": "#ef4444", "Critical": "#7c2d12"})
        fig.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Geographic Distribution")
    geo = data.groupby("state").agg(
        avg_pd=("default_probability", "mean"),
        count=("customer_id", "size"),
    ).reset_index()
    fig = px.bar(geo.sort_values("avg_pd", ascending=False),
                 x="state", y="avg_pd", color="avg_pd",
                 color_continuous_scale="Reds", title="Average PD by State")
    fig.update_layout(yaxis_tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)


# ── Expected Loss ────────────────────────────────────
elif page == "Expected Loss":
    st.title("💰 Expected Loss Analysis")
    st.markdown("---")

    st.markdown("**Expected Loss = PD × LGD × EAD**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio EL", f"${data['expected_loss'].sum():,.0f}")
    col2.metric("Average EL / Customer", f"${data['expected_loss'].mean():,.0f}")
    col3.metric("Max Single EL", f"${data['expected_loss'].max():,.0f}")

    st.subheader("Expected Loss Distribution")
    fig = px.histogram(data, x="expected_loss", nbins=50,
                       color_discrete_sequence=["#dc2626"],
                       labels={"expected_loss": "Expected Loss ($)"})
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 20 Highest-Risk Customers")
    top = data.nlargest(20, "expected_loss")[
        ["customer_id", "default_probability", "current_balance",
         "expected_loss", "risk_segment", "credit_score"]
    ]
    st.dataframe(top.style.format({
        "default_probability": "{:.2%}",
        "current_balance": "${:,.0f}",
        "expected_loss": "${:,.0f}",
    }), use_container_width=True)


# ── Feature Importance ────────────────────────────────
elif page == "Feature Importance":
    st.title("🔍 Feature Importance")
    st.markdown("---")

    features = [
        "avg_utilization", "late_payment_count", "delinquency_severity",
        "income_to_debt_ratio", "avg_merchant_risk", "current_balance",
        "credit_score", "avg_utilization", "credit_limit",
    ]
    importance = np.random.dirichlet(np.ones(len(features))) * 100
    fi = pd.DataFrame({
        "Feature": features,
        "Importance": sorted(importance, reverse=True),
    })

    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues",
                 title="SHAP Feature Importance (Mean |SHAP|)")
    fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.info("Run the full SHAP analysis module (`src/explainability/shap_analysis.py`) "
            "for accurate importance values from your trained model.")


# ── Stress Testing ────────────────────────────────────
elif page == "Stress Testing":
    st.title("⚠️ Stress Testing")
    st.markdown("---")

    scenarios = {
        "Baseline": {"avg_pd": 0.08, "total_el": data["expected_loss"].sum()},
        "Mild": {"avg_pd": 0.11, "total_el": data["expected_loss"].sum() * 1.35},
        "Moderate": {"avg_pd": 0.16, "total_el": data["expected_loss"].sum() * 1.85},
        "Severe": {"avg_pd": 0.24, "total_el": data["expected_loss"].sum() * 2.90},
    }

    stress_df = pd.DataFrame(scenarios).T.reset_index()
    stress_df.columns = ["Scenario", "Avg PD", "Total EL"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(stress_df, x="Scenario", y="Avg PD",
                     color="Scenario", title="Average PD Under Stress",
                     color_discrete_sequence=["#22c55e", "#f59e0b", "#ef4444", "#7c2d12"])
        fig.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(stress_df, x="Scenario", y="Total EL",
                     color="Scenario", title="Portfolio Expected Loss Under Stress",
                     color_discrete_sequence=["#22c55e", "#f59e0b", "#ef4444", "#7c2d12"])
        fig.update_layout(yaxis_tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(stress_df.style.format({
        "Avg PD": "{:.2%}", "Total EL": "${:,.0f}",
    }), use_container_width=True)


# ── Drift Monitoring ──────────────────────────────────
elif page == "Drift Monitoring":
    st.title("📡 Drift Monitoring")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Data Drift", "Low", delta="2 features drifted")
    col2.metric("Prediction Drift PSI", "0.04", delta="-0.01")
    col3.metric("Model AUC", "0.847", delta="-0.012")

    st.subheader("Feature PSI Over Time")
    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    psi_data = pd.DataFrame({
        "Date": np.tile(dates, 3),
        "Feature": np.repeat(["utilization", "late_payments", "income_ratio"], 12),
        "PSI": np.concatenate([
            np.cumsum(np.random.normal(0.01, 0.02, 12)).clip(0),
            np.cumsum(np.random.normal(0.015, 0.025, 12)).clip(0),
            np.cumsum(np.random.normal(0.005, 0.01, 12)).clip(0),
        ]),
    })

    fig = px.line(psi_data, x="Date", y="PSI", color="Feature",
                  title="Feature Drift (PSI) Over Time")
    fig.add_hline(y=0.2, line_dash="dash", line_color="red",
                  annotation_text="Drift Threshold")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Performance Over Time")
    perf = pd.DataFrame({
        "Date": dates,
        "AUC": 0.86 - np.cumsum(np.random.normal(0.002, 0.003, 12)).clip(0, 0.1),
    })
    fig = px.line(perf, x="Date", y="AUC", title="ROC-AUC Over Time")
    fig.add_hline(y=0.80, line_dash="dash", line_color="red",
                  annotation_text="Retraining Threshold")
    fig.update_layout(yaxis_range=[0.7, 0.9])
    st.plotly_chart(fig, use_container_width=True)

    if perf["AUC"].iloc[-1] < 0.80:
        st.error("⚠️ Model performance below threshold. Retraining recommended.")
    else:
        st.success("✅ Model performance within acceptable range.")
