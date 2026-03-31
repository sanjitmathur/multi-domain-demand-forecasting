"""System Design — architecture overview of the forecasting platform."""

import streamlit as st

st.title("System Design")
st.caption("Production architecture: Streamlit UI → FastAPI → Model Artifacts → Predictions")

st.markdown("""
```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                      │
│  Interactive UI · Domain Tabs · Charts · Metrics Display    │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTP POST /forecast/{domain}
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI INFERENCE SERVICE                 │
│  /forecast/airline · /forecast/ecommerce · /forecast/payment│
│  /forecast/batch · Request validation (Pydantic)            │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────┐  ┌───────────────────────────────┐
│    FEATURE PIPELINE      │  │     MODEL ARTIFACTS           │
│  src/features/           │  │     models/saved/             │
│                          │  │                               │
│  • Rolling statistics    │  │  • {domain}_xgb.joblib        │
│  • Lag features          │  │  • {domain}_lgbm.joblib       │
│  • Cyclical encoding     │  │  • {domain}_xgb_q10.joblib    │
│  • Domain transforms     │  │  • {domain}_xgb_q90.joblib    │
│                          │  │  • {domain}_lgbm_q10.joblib   │
│                          │  │  • {domain}_lgbm_q90.joblib   │
│                          │  │  • {domain}_meta.joblib       │
└──────────┬───────────────┘  └───────────────┬───────────────┘
           │                                  │
           └──────────────┬───────────────────┘
                          ▼
            ┌─────────────────────────┐
            │   ENSEMBLE FORECASTER   │
            │  XGB + LGBM → Ridge     │
            │  Meta-learner blend     │
            └────────────┬────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │   PREDICTION RESPONSE   │
            │  forecast: float        │
            │  lower_bound: float     │
            │  upper_bound: float     │
            │  confidence: float      │
            └─────────────────────────┘
```
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Request Flow")
    st.markdown("""
1. User selects domain + parameters in **Streamlit**
2. Streamlit calls **FastAPI** endpoint (`/forecast/{domain}`)
3. FastAPI builds feature vector from request params
4. **Feature pipeline** applies rolling stats, lags, cyclical encoding
5. **Ensemble** runs XGBoost + LightGBM → Ridge meta-learner
6. Returns forecast + P10/P90 bounds + confidence score
""")

    st.markdown("### API Endpoints")
    st.code("""POST /forecast/airline    → ForecastResponse
POST /forecast/ecommerce  → ForecastResponse
POST /forecast/payment    → ForecastResponse
POST /forecast/batch      → dict[domain, ForecastResponse]""", language="text")

with col2:
    st.markdown("### Project Structure")
    st.code("""multi-domain-demand-forecasting/
├── api/
│   ├── main.py              # FastAPI service
│   └── schemas.py           # Pydantic models
├── dashboard/
│   ├── app.py               # Navigation entry
│   └── pages/               # Multi-page views
├── src/
│   ├── features/            # Per-domain feature engineering
│   ├── models/
│   │   ├── base_model.py    # XGB + LGBM + quantile wrapper
│   │   ├── ensemble.py      # Stacking + Ridge meta-learner
│   │   └── train_all.py     # Training orchestrator
│   └── evaluation/
│       └── metrics.py       # RMSE, MAE, R², MAPE, sMAPE
├── models/saved/            # Serialized .joblib artifacts
├── data/                    # Domain datasets
├── results/                 # Evaluation CSVs
└── tests/                   # pytest suite""", language="text")

    st.markdown("### Design Decisions")
    st.markdown("""
- **Stacking over blending** — OOF predictions prevent leakage
- **Quantile regression** — native uncertainty without bootstrapping
- **Ridge meta-learner** — regularized, fast, interpretable weights
- **TimeSeriesSplit** — respects temporal ordering in CV
- **Joblib serialization** — efficient model persistence
""")
