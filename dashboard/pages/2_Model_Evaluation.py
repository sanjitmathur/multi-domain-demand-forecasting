"""Model Evaluation — metrics, residuals, actual vs predicted."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ensemble import EnsembleForecaster
from src.features.airline_features import engineer_airline_features
from src.features.ecommerce_features import engineer_ecommerce_features
from src.features.payment_features import engineer_payment_features
from src.evaluation.metrics import rmse, mae, r_squared, rmse_pct

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

st.title("Model Evaluation")
st.caption("Per-domain metrics, residual analysis, and actual vs predicted scatter")


@st.cache_resource
def load_ensemble():
    ens = EnsembleForecaster()
    ens.load(str(MODELS_DIR), ["airline", "ecommerce", "payment"])
    return ens


@st.cache_data
def get_eval_data(_ens):
    datasets = {
        "airline": (pd.read_csv(DATA_DIR / "airline_bookings.csv"), engineer_airline_features),
        "ecommerce": (pd.read_csv(DATA_DIR / "ecommerce_demand.csv"), engineer_ecommerce_features),
        "payment": (pd.read_csv(DATA_DIR / "payment_volume.csv"), engineer_payment_features),
    }
    results = {}
    for domain, (df, eng_fn) in datasets.items():
        features, target = eng_fn(df)
        split = int(len(features) * 0.8)
        X_test, y_test = features.iloc[split:], target.iloc[split:]
        pred = _ens.predict(domain, X_test)
        xgb_p = _ens.predict_single_xgb(domain, X_test)
        lgbm_p = _ens.predict_single_lgbm(domain, X_test)
        results[domain] = {
            "y_test": y_test.values, "pred": pred["forecast"],
            "lower": pred["lower_bound"], "upper": pred["upper_bound"],
            "xgb": xgb_p, "lgbm": lgbm_p,
        }
    return results


try:
    ens = load_ensemble()
    data = get_eval_data(ens)
except Exception as e:
    st.error(f"Load models first: {e}")
    st.stop()

domain = st.selectbox("Domain", ["airline", "ecommerce", "payment"])
d = data[domain]
y, p = d["y_test"], d["pred"]
residuals = y - p

# --- Metrics Cards ---
st.subheader("Performance Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("RMSE", f"{rmse(y, p):.4f}")
c2.metric("MAE", f"{mae(y, p):.4f}")
c3.metric("R²", f"{r_squared(y, p):.4f}")
c4.metric("RMSE %", f"{rmse_pct(y, p):.2f}%")

# --- Model Comparison Table ---
st.subheader("Model Comparison")
rows = []
for name, preds in [("XGBoost", d["xgb"]), ("LightGBM", d["lgbm"]), ("Ensemble+Meta", p)]:
    rows.append({"Model": name, "RMSE": round(rmse(y, preds), 4),
                 "MAE": round(mae(y, preds), 4), "R²": round(r_squared(y, preds), 4)})
st.dataframe(pd.DataFrame(rows), hide_index=True)

# --- Charts ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Actual vs Predicted")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y, y=p, mode="markers", marker=dict(size=3, opacity=0.5, color="#4ECDC4"), name="Predictions"))
    mn, mx = min(y.min(), p.min()), max(y.max(), p.max())
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", line=dict(dash="dash", color="#FF6B6B"), name="Perfect"))
    fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted", template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Residual Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=50, marker_color="#C084FC", opacity=0.8))
    fig.add_vline(x=0, line_dash="dash", line_color="#FFD93D")
    fig.update_layout(xaxis_title="Residual (Actual - Predicted)", yaxis_title="Count",
                      template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

# --- Residual over index ---
st.subheader("Residuals Over Test Set")
fig = go.Figure()
fig.add_trace(go.Scatter(y=residuals, mode="markers", marker=dict(size=2, color="#45B7D1", opacity=0.5)))
fig.add_hline(y=0, line_dash="dash", line_color="#FF6B6B")
fig.update_layout(xaxis_title="Sample Index", yaxis_title="Residual", template="plotly_dark", height=300)
st.plotly_chart(fig, use_container_width=True)
