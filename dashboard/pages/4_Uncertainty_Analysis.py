"""Uncertainty Analysis — quantile confidence bands and historical vs forecast."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ensemble import EnsembleForecaster
from src.features.airline_features import engineer_airline_features
from src.features.ecommerce_features import engineer_ecommerce_features
from src.features.payment_features import engineer_payment_features

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

st.title("Uncertainty & Forecast Visualization")
st.caption("Quantile-based confidence intervals (10th / 50th / 90th percentile)")


@st.cache_resource
def load_ensemble():
    ens = EnsembleForecaster()
    ens.load(str(MODELS_DIR), ["airline", "ecommerce", "payment"])
    return ens


@st.cache_data
def get_domain_data(_ens, domain):
    eng_fns = {
        "airline": (pd.read_csv(DATA_DIR / "airline_bookings.csv"), engineer_airline_features),
        "ecommerce": (pd.read_csv(DATA_DIR / "ecommerce_demand.csv"), engineer_ecommerce_features),
        "payment": (pd.read_csv(DATA_DIR / "payment_volume.csv"), engineer_payment_features),
    }
    df, eng_fn = eng_fns[domain]
    features, target = eng_fn(df)
    n = len(features)
    split = int(n * 0.8)
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = target.iloc[:split], target.iloc[split:]
    pred = _ens.predict(domain, X_test)
    return y_train.values, y_test.values, pred


try:
    ens = load_ensemble()
except Exception as e:
    st.error(f"Load models first: {e}")
    st.stop()

domain = st.selectbox("Domain", ["airline", "ecommerce", "payment"])
y_train, y_test, pred = get_domain_data(ens, domain)
forecast, lower, upper = pred["forecast"], pred["lower_bound"], pred["upper_bound"]
median = forecast  # Ridge meta-learner output serves as median estimate

# --- Confidence Band Plot ---
st.subheader("Prediction Intervals (80% Confidence)")
n_show = min(200, len(y_test))
idx = np.arange(n_show)

fig = go.Figure()
fig.add_trace(go.Scatter(x=idx, y=upper[:n_show], mode="lines", line=dict(width=0),
                         showlegend=False, name="Upper"))
fig.add_trace(go.Scatter(x=idx, y=lower[:n_show], mode="lines", line=dict(width=0),
                         fill="tonexty", fillcolor="rgba(78, 205, 196, 0.2)",
                         name="80% Interval (P10–P90)"))
fig.add_trace(go.Scatter(x=idx, y=median[:n_show], mode="lines",
                         line=dict(color="#FFD93D", width=2), name="Median Forecast"))
fig.add_trace(go.Scatter(x=idx, y=y_test[:n_show], mode="markers",
                         marker=dict(size=3, color="#FF6B6B", opacity=0.6), name="Actual"))
fig.update_layout(xaxis_title="Test Sample", yaxis_title="Value",
                  template="plotly_dark", height=450,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)

# --- Coverage Stats ---
coverage = np.mean((y_test >= lower) & (y_test <= upper))
avg_width = np.mean(upper - lower)
col1, col2, col3 = st.columns(3)
col1.metric("Interval Coverage", f"{coverage:.1%}", help="% of actuals within P10–P90 band")
col2.metric("Avg Interval Width", f"{avg_width:.2f}")
col3.metric("Samples", f"{len(y_test):,}")

# --- Historical vs Forecast ---
st.subheader("Historical vs Forecast")
n_hist = min(100, len(y_train))
hist_idx = np.arange(n_hist)
fore_idx = np.arange(n_hist, n_hist + n_show)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=hist_idx, y=y_train[-n_hist:], mode="lines",
                          line=dict(color="#45B7D1", width=1.5), name="Historical"))
# Forecast with confidence band
fig2.add_trace(go.Scatter(x=fore_idx, y=upper[:n_show], mode="lines", line=dict(width=0), showlegend=False))
fig2.add_trace(go.Scatter(x=fore_idx, y=lower[:n_show], mode="lines", line=dict(width=0),
                          fill="tonexty", fillcolor="rgba(192, 132, 252, 0.2)", name="Forecast Interval"))
fig2.add_trace(go.Scatter(x=fore_idx, y=median[:n_show], mode="lines",
                          line=dict(color="#C084FC", width=2), name="Forecast"))
fig2.add_trace(go.Scatter(x=fore_idx, y=y_test[:n_show], mode="lines",
                          line=dict(color="#FF6B6B", width=1, dash="dot"), name="Actual (test)"))
fig2.add_vline(x=n_hist, line_dash="dash", line_color="white", opacity=0.5)
fig2.add_annotation(x=n_hist, y=max(y_train[-n_hist:].max(), forecast[:n_show].max()),
                    text="Train / Test Split", showarrow=False, font=dict(color="white", size=11))
fig2.update_layout(xaxis_title="Time Index", yaxis_title="Value",
                   template="plotly_dark", height=450,
                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig2, use_container_width=True)
