"""Feature Importance — XGBoost & LightGBM built-in importance."""

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

st.title("Feature Importance Analysis")
st.caption("XGBoost gain-based and LightGBM split-based feature importance per domain")


@st.cache_resource
def load_ensemble():
    ens = EnsembleForecaster()
    ens.load(str(MODELS_DIR), ["airline", "ecommerce", "payment"])
    return ens


@st.cache_data
def get_feature_names():
    airline_df = pd.read_csv(DATA_DIR / "airline_bookings.csv")
    ecom_df = pd.read_csv(DATA_DIR / "ecommerce_demand.csv")
    payment_df = pd.read_csv(DATA_DIR / "payment_volume.csv")
    return {
        "airline": list(engineer_airline_features(airline_df)[0].columns),
        "ecommerce": list(engineer_ecommerce_features(ecom_df)[0].columns),
        "payment": list(engineer_payment_features(payment_df)[0].columns),
    }


def importance_chart(importances, names, title, color):
    df = pd.DataFrame({"feature": names, "importance": importances})
    df = df.sort_values("importance", ascending=True).tail(15)
    fig = go.Figure(go.Bar(x=df["importance"], y=df["feature"], orientation="h",
                           marker_color=color, opacity=0.85))
    fig.update_layout(title=title, template="plotly_dark", height=500,
                      margin=dict(l=200), yaxis=dict(tickfont=dict(size=11)))
    return fig


try:
    ens = load_ensemble()
    feat_names = get_feature_names()
except Exception as e:
    st.error(f"Load models first: {e}")
    st.stop()

domain = st.selectbox("Domain", ["airline", "ecommerce", "payment"])
model = ens.domain_models[domain]
names = feat_names[domain]

col1, col2 = st.columns(2)

with col1:
    xgb_imp = model.xgb_model.feature_importances_
    st.plotly_chart(importance_chart(xgb_imp, names, "XGBoost Feature Importance (Gain)", "#FF6B6B"),
                    use_container_width=True)

with col2:
    lgbm_imp = model.lgbm_model.feature_importances_
    st.plotly_chart(importance_chart(lgbm_imp, names, "LightGBM Feature Importance (Split)", "#4ECDC4"),
                    use_container_width=True)

# --- Combined ranking ---
st.subheader("Combined Importance Ranking")
xgb_norm = xgb_imp / (xgb_imp.sum() + 1e-8)
lgbm_norm = lgbm_imp / (lgbm_imp.sum() + 1e-8)
combined = (xgb_norm + lgbm_norm) / 2
ranking = pd.DataFrame({"Feature": names, "XGBoost (norm)": np.round(xgb_norm, 4),
                         "LightGBM (norm)": np.round(lgbm_norm, 4),
                         "Combined": np.round(combined, 4)})
ranking = ranking.sort_values("Combined", ascending=False).reset_index(drop=True)
ranking.index += 1
st.dataframe(ranking, use_container_width=True)

st.info("**Interpretation**: High-importance lag/rolling features indicate strong autoregressive signal. "
        "Temporal features (cyclical encodings) capture seasonality. Price/promo features drive cross-sectional variation.")

# --- SHAP Summary ---
st.divider()
st.subheader("SHAP Feature Impact")
st.caption("Mean absolute SHAP values — shows direction and magnitude of each feature's effect on predictions")

import shap


@st.cache_data
def compute_shap(_model, domain, _feat_names):
    eng_fns = {
        "airline": (pd.read_csv(DATA_DIR / "airline_bookings.csv"), engineer_airline_features),
        "ecommerce": (pd.read_csv(DATA_DIR / "ecommerce_demand.csv"), engineer_ecommerce_features),
        "payment": (pd.read_csv(DATA_DIR / "payment_volume.csv"), engineer_payment_features),
    }
    df, eng_fn = eng_fns[domain]
    features, _ = eng_fn(df)
    split = int(len(features) * 0.8)
    X_sample = features.iloc[split:].sample(n=min(200, len(features) - split), random_state=42)

    explainer = shap.TreeExplainer(_model.xgb_model)
    shap_values = explainer.shap_values(X_sample)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return mean_abs, shap_values, X_sample


shap_mean, shap_vals, X_sample = compute_shap(model, domain, names)

# Bar chart of mean |SHAP|
shap_df = pd.DataFrame({"feature": names, "mean_abs_shap": shap_mean})
shap_df = shap_df.sort_values("mean_abs_shap", ascending=True).tail(15)
fig = go.Figure(go.Bar(x=shap_df["mean_abs_shap"], y=shap_df["feature"], orientation="h",
                       marker_color="#C084FC", opacity=0.85))
fig.update_layout(title="SHAP Mean |Impact| (XGBoost)", template="plotly_dark", height=500,
                  margin=dict(l=200), xaxis_title="Mean |SHAP value|")
st.plotly_chart(fig, use_container_width=True)

# Beeswarm-style: top features scatter
st.caption("SHAP value distribution for top 10 features (each dot = one test sample)")
top_idx = np.argsort(shap_mean)[-10:][::-1]
import plotly.express as px

rows = []
for i in top_idx:
    for j in range(len(X_sample)):
        rows.append({"Feature": names[i], "SHAP Value": shap_vals[j, i],
                      "Feature Value": X_sample.iloc[j, i]})
scatter_df = pd.DataFrame(rows)
fig2 = px.strip(scatter_df, x="SHAP Value", y="Feature", color="Feature Value",
                orientation="h")
fig2.update_coloraxes(colorscale="RdBu_r")
fig2.update_layout(template="plotly_dark", height=450, showlegend=False)
st.plotly_chart(fig2, use_container_width=True)
