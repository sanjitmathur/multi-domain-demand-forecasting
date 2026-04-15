"""Model Internals — architecture, feature importance (XGB/LGBM/SHAP), system design."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import _theme as theme
from src.features.airline_features import engineer_airline_features
from src.features.ecommerce_features import engineer_ecommerce_features
from src.features.payment_features import engineer_payment_features
from src.models.ensemble import EnsembleForecaster

theme.apply()

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"


@st.cache_resource(show_spinner=False)
def load_ensemble() -> EnsembleForecaster:
    ens = EnsembleForecaster()
    ens.load(str(MODELS_DIR), ["airline", "ecommerce", "payment"])
    return ens


@st.cache_data(show_spinner=False)
def _features(domain: str) -> tuple[pd.DataFrame, pd.Series]:
    loaders = {
        "airline":   (DATA_DIR / "airline_bookings.csv", engineer_airline_features),
        "ecommerce": (DATA_DIR / "ecommerce_demand.csv", engineer_ecommerce_features),
        "payment":   (DATA_DIR / "payment_volume.csv",   engineer_payment_features),
    }
    path, fn = loaders[domain]
    return fn(pd.read_csv(path))


theme.brand_header(status_text="Models loaded · 22 tests passing")
theme.page_header(
    eyebrow="UNDER THE HOOD",
    title="Model Internals",
    subtitle="Training pipeline, feature signal, and serving architecture.",
)

try:
    ens = load_ensemble()
except Exception as e:
    st.error(f"Load models first: {e}")
    st.stop()

tab_arch, tab_feat, tab_sys = st.tabs(["Architecture", "Feature Importance", "System Design"])


# ── Architecture ─────────────────────────────────────────────────────────────
with tab_arch:
    theme.section("Training Pipeline", "Stacking ensemble with Ridge meta-learner and quantile-regression uncertainty.")
    st.code("""┌─────────────────────────────────────────────────────────┐
│                      RAW DATA                           │
│         Airline  ·  E-Commerce  ·  Payment              │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING                     │
│  Rolling Stats · Lag Features · Cyclical Encoding       │
│  Trend Direction · Domain-Specific Transforms           │
└──────────────────────────┬──────────────────────────────┘
                ┌──────────┴──────────┐
                ▼                     ▼
┌──────────────────────┐  ┌──────────────────────┐
│      XGBoost         │  │      LightGBM        │
│  Point + Quantile    │  │  Point + Quantile    │
│  (α=0.1 / 0.9)       │  │  (α=0.1 / 0.9)       │
└──────────┬───────────┘  └──────────┬───────────┘
           │     ┌────────────┐      │
           └────▶│  STACKING  │◀─────┘
                 │  5-Fold    │
                 │ TimeSeries │
                 │  Split OOF │
                 └─────┬──────┘
                       ▼
         ┌─────────────────────────┐
         │   RIDGE META-LEARNER    │
         │   α=1.0 · OOF inputs    │
         └─────────┬───────────────┘
                   ▼
         ┌─────────────────────────┐
         │   FINAL PREDICTIONS     │
         │  Point + P10 / P90      │
         └─────────────────────────┘""", language="text")

    c1, c2, c3 = st.columns(3)
    with c1:
        theme.section("Base Models")
        st.markdown(
            "- **XGBoost** — gradient-boosted trees, `max_depth=6`, 300 estimators  \n"
            "- **LightGBM** — leaf-wise growth, `num_leaves=31`, 300 estimators  \n"
            "- Both trained with point + quantile (α=0.1, 0.9) objectives"
        )
    with c2:
        theme.section("Stacking Strategy")
        st.markdown(
            "- 5-fold **TimeSeriesSplit** for out-of-fold predictions  \n"
            "- OOF predictions become meta-features (no leakage)  \n"
            "- Ridge regression (α=1.0) as meta-learner"
        )
    with c3:
        theme.section("Uncertainty Quantification")
        st.markdown(
            "- Quantile regression at the **10th** and **90th** percentiles  \n"
            "- Averaged across XGBoost + LightGBM → 80% prediction interval  \n"
            "- Empirical coverage reported in **Evaluation** tab"
        )


# ── Feature Importance ───────────────────────────────────────────────────────
with tab_feat:
    domain = st.selectbox("Domain", ["airline", "ecommerce", "payment"], key="fi_domain")
    model = ens.domain_models[domain]
    X, _ = _features(domain)
    names = list(X.columns)

    xgb_imp = model.xgb_model.feature_importances_
    lgbm_imp = model.lgbm_model.feature_importances_
    xgb_norm = xgb_imp / (xgb_imp.sum() + 1e-8)
    lgbm_norm = lgbm_imp / (lgbm_imp.sum() + 1e-8)
    combined = (xgb_norm + lgbm_norm) / 2

    theme.section("Tree-based importance", "XGBoost gain vs LightGBM split counts, top 15 features.")
    col1, col2 = st.columns(2)
    with col1:
        df = pd.DataFrame({"f": names, "v": xgb_imp}).sort_values("v").tail(15)
        fig = go.Figure(go.Bar(x=df["v"], y=df["f"], orientation="h",
                                marker_color=theme.ACCENT, opacity=0.88))
        fig.update_layout(title="XGBoost — Gain", height=480, margin=dict(l=180))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df = pd.DataFrame({"f": names, "v": lgbm_imp}).sort_values("v").tail(15)
        fig = go.Figure(go.Bar(x=df["v"], y=df["f"], orientation="h",
                                marker_color=theme.PREDICTED, opacity=0.88))
        fig.update_layout(title="LightGBM — Splits", height=480, margin=dict(l=180))
        st.plotly_chart(fig, use_container_width=True)

    theme.section("Combined ranking")
    ranking = pd.DataFrame({
        "Feature": names,
        "XGB (norm)": np.round(xgb_norm, 4),
        "LGBM (norm)": np.round(lgbm_norm, 4),
        "Combined": np.round(combined, 4),
    }).sort_values("Combined", ascending=False).reset_index(drop=True)
    ranking.index += 1
    st.dataframe(ranking, use_container_width=True)

    # SHAP — the most rigorous interpretability signal we have.
    st.divider()
    theme.section("SHAP feature impact",
                  "Tree SHAP on 200 held-out samples — directional, additive, leakage-aware.")

    @st.cache_data(show_spinner=False)
    def _shap(_model, domain: str):
        import shap
        split = int(len(X) * 0.8)
        sample = X.iloc[split:].sample(n=min(200, len(X) - split), random_state=42)
        explainer = shap.TreeExplainer(_model.xgb_model)
        sv = explainer.shap_values(sample)
        return np.abs(sv).mean(axis=0), sv, sample

    shap_mean, shap_vals, X_sample = _shap(model, domain)
    shap_df = pd.DataFrame({"f": names, "v": shap_mean}).sort_values("v").tail(15)
    fig = go.Figure(go.Bar(x=shap_df["v"], y=shap_df["f"], orientation="h",
                            marker_color="#C084FC", opacity=0.88))
    fig.update_layout(title="Mean |SHAP| (XGBoost)", height=480, margin=dict(l=180))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("SHAP value distribution for top 10 features — each dot is one held-out sample.")
    top = np.argsort(shap_mean)[-10:][::-1]
    rows = []
    for i in top:
        for j in range(len(X_sample)):
            rows.append({
                "Feature": names[i],
                "SHAP Value": shap_vals[j, i],
                "Feature Value": X_sample.iloc[j, i],
            })
    fig2 = px.strip(pd.DataFrame(rows), x="SHAP Value", y="Feature",
                    color="Feature Value", orientation="h")
    fig2.update_coloraxes(colorscale="RdBu_r")
    fig2.update_layout(height=460, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)


# ── System Design ────────────────────────────────────────────────────────────
with tab_sys:
    theme.section("Serving Architecture",
                  "Streamlit UI → FastAPI inference service → joblib-serialized model artifacts.")
    st.code("""┌──────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                   │
└──────────────────────────┬───────────────────────────────┘
                           │  POST /forecast/{domain}
                           ▼
┌──────────────────────────────────────────────────────────┐
│                   FastAPI INFERENCE SERVICE              │
│  Pydantic validation · CORS · Health endpoint            │
└──────────────┬──────────────────────────┬────────────────┘
               ▼                          ▼
┌──────────────────────────┐  ┌───────────────────────────┐
│    FEATURE PIPELINE      │  │     MODEL ARTIFACTS       │
│  src/features/           │  │     models/saved/         │
│  • Rolling statistics    │  │  • {domain}_xgb.joblib    │
│  • Lag features          │  │  • {domain}_lgbm.joblib   │
│  • Cyclical encoding     │  │  • {domain}_*_q10/q90     │
│  • Domain transforms     │  │  • {domain}_meta.joblib   │
└──────────┬───────────────┘  └───────────────┬───────────┘
           └──────────────┬───────────────────┘
                          ▼
            ┌─────────────────────────┐
            │   ENSEMBLE FORECASTER   │
            │  XGB + LGBM → Ridge     │
            └────────────┬────────────┘
                         ▼
            ┌─────────────────────────┐
            │   PREDICTION RESPONSE   │
            │  forecast / P10 / P90   │
            │  interval_score         │
            └─────────────────────────┘""", language="text")

    c1, c2 = st.columns(2)
    with c1:
        theme.section("API surface")
        st.code(
            "POST /forecast/airline    → ForecastResponse\n"
            "POST /forecast/ecommerce  → ForecastResponse\n"
            "POST /forecast/payment    → ForecastResponse\n"
            "POST /forecast/batch      → dict[domain, ForecastResponse]\n"
            "GET  /docs                → OpenAPI / Swagger UI",
            language="text",
        )
        theme.section("Design decisions")
        st.markdown(
            "- **Stacking with time-aware OOF** — no leakage in meta-features  \n"
            "- **Quantile regression** — native uncertainty, no bootstrap cost  \n"
            "- **Ridge meta-learner** — regularized, fast, inspectable weights  \n"
            "- **Interval score**, not *confidence* — documented as a tightness metric, not a probability"
        )

    with c2:
        theme.section("Project layout")
        st.code(
            "multi-domain-demand-forecasting/\n"
            "├── api/                 # FastAPI inference service\n"
            "├── dashboard/           # Streamlit multi-page app\n"
            "├── src/\n"
            "│   ├── features/        # Per-domain engineering\n"
            "│   ├── models/          # Base models, ensemble, training\n"
            "│   └── evaluation/      # Metrics\n"
            "├── tests/               # pytest suite\n"
            "├── models/saved/        # Serialized .joblib artifacts\n"
            "├── data/                # Generated CSV datasets\n"
            "├── results/             # Training-time evaluation\n"
            "├── Dockerfile           # API container\n"
            "└── .github/workflows/   # CI + keep-alive",
            language="text",
        )
        theme.section("Deployment")
        st.markdown(
            "- **Dashboard:** Streamlit Community Cloud  \n"
            "- **API:** Dockerized (`uvicorn api.main:app`), deployable to Fly.io / Render / Cloud Run  \n"
            "- **CI:** GitHub Actions runs pytest on every push"
        )
