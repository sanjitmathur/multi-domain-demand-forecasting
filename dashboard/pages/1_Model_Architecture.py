"""Model Architecture — visual ML pipeline diagram."""

import streamlit as st

st.title("Model Architecture")
st.caption("End-to-end ML pipeline for multi-domain demand forecasting")

st.markdown("""
```
┌─────────────────────────────────────────────────────────┐
│                      RAW DATA                           │
│         Airline  ·  E-Commerce  ·  Payment              │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING                     |
│  Rolling Stats · Lag Features · Cyclical Encoding       │
│  Trend Direction · Domain-Specific Transforms           │
└──────────────────────────┬──────────────────────────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
┌──────────────────────┐  ┌──────────────────────┐
│      XGBoost         │  │      LightGBM        │
│  Point + Quantile    │  │  Point + Quantile    │
│  (α=0.1, 0.9)        │  │  (α=0.1, 0.9)        │
│  max_depth=6         │  │  num_leaves=31       │
│  300 estimators      │  │  300 estimators      │
└──────────┬───────────┘  └──────────┬───────────┘
           │                         │
           │    ┌────────────┐       │
           └───►│  STACKING  │◄──────┘
                │  5-Fold    │
                │  TimeSeries│
                │  Split OOF │
                └─────┬──────┘
                      │
                      ▼
        ┌─────────────────────────┐
        │   RIDGE META-LEARNER    │
        │   α=1.0 · OOF inputs    │
        │   Regularized blend     │
        └─────────┬───────────────┘
                  │
                  ▼
        ┌─────────────────────────┐
        │   FINAL PREDICTIONS     │
        │  Forecast + P10 / P90   │
        │  Confidence Score       │
        └─────────────────────────┘
```
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### Base Models")
    st.markdown("""
- **XGBoost** — gradient-boosted trees, `max_depth=6`, 300 estimators
- **LightGBM** — leaf-wise growth, `num_leaves=31`, 300 estimators
- Both trained with point + quantile (α=0.1, 0.9) objectives
""")
with col2:
    st.markdown("### Stacking Strategy")
    st.markdown("""
- 5-fold **TimeSeriesSplit** for out-of-fold predictions
- OOF predictions become meta-features (no data leakage)
- Ridge regression (α=1.0) as meta-learner
""")
with col3:
    st.markdown("### Uncertainty Quantification")
    st.markdown("""
- Quantile regression at **10th** and **90th** percentiles
- Both XGBoost and LightGBM produce quantile estimates
- Averaged quantile bounds → 80% prediction interval
""")
