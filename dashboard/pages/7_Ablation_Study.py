"""Ablation Study — measure impact of removing feature groups."""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ensemble import EnsembleForecaster
from src.features.airline_features import engineer_airline_features
from src.features.ecommerce_features import engineer_ecommerce_features
from src.features.payment_features import engineer_payment_features
from src.evaluation.metrics import rmse, mae, r_squared

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

st.title("Ablation Study")
st.caption("Drop feature groups and measure prediction degradation to understand what drives model performance")

FEATURE_GROUPS = {
    "airline": {
        "Lag Features": ["bookings_lag_1", "bookings_lag_7", "bookings_lag_14"],
        "Rolling Stats": ["bookings_rolling_mean_7", "bookings_rolling_std_7", "bookings_rolling_max_7",
                          "bookings_rolling_mean_14", "bookings_rolling_mean_30"],
        "Price Features": ["price", "competitor_price", "price_ratio", "price_per_seat"],
        "Temporal": ["day_of_week", "month", "is_weekend", "is_holiday", "dow_sin", "dow_cos", "month_sin", "month_cos"],
        "Trend": ["bookings_trend"],
    },
    "ecommerce": {
        "Lag Features": ["quantity_sold_lag_1", "quantity_sold_lag_7", "quantity_sold_lag_14"],
        "Rolling Stats": ["quantity_sold_rolling_mean_7", "quantity_sold_rolling_std_7",
                          "quantity_sold_rolling_max_7", "quantity_sold_rolling_mean_30"],
        "Price Features": ["price", "competitor_price", "price_ratio", "discount_pct", "price_x_promotion"],
        "Temporal": ["day_of_week", "month", "is_weekend", "is_holiday", "season_encoded",
                     "dow_sin", "dow_cos", "month_sin", "month_cos"],
        "Trend": ["quantity_sold_trend"],
    },
    "payment": {
        "Lag Features": ["prev_hour_volume", "prev_day_same_hour", "prev_week_same_hour"],
        "Rolling Stats": ["volume_rolling_mean_6h", "volume_rolling_std_6h", "volume_rolling_max_6h",
                          "volume_rolling_mean_24", "volume_rolling_std_24", "volume_rolling_max_24",
                          "volume_rolling_mean_7d", "volume_rolling_std_7d", "volume_rolling_max_7d"],
        "Temporal": ["hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday",
                     "hour_sin", "hour_cos", "dow_sin", "dow_cos"],
        "Trend": ["volume_trend", "volume_volatility_7d"],
    },
}


@st.cache_resource
def load_ensemble():
    ens = EnsembleForecaster()
    ens.load(str(MODELS_DIR), ["airline", "ecommerce", "payment"])
    return ens


@st.cache_data
def run_ablation(_ens, domain):
    eng_fns = {
        "airline": (pd.read_csv(DATA_DIR / "airline_bookings.csv"), engineer_airline_features),
        "ecommerce": (pd.read_csv(DATA_DIR / "ecommerce_demand.csv"), engineer_ecommerce_features),
        "payment": (pd.read_csv(DATA_DIR / "payment_volume.csv"), engineer_payment_features),
    }
    df, eng_fn = eng_fns[domain]
    features, target = eng_fn(df)
    split = int(len(features) * 0.8)
    X_test, y_test = features.iloc[split:], target.iloc[split:].values

    # Baseline
    baseline_pred = _ens.predict(domain, X_test)["forecast"]
    baseline_rmse = rmse(y_test, baseline_pred)
    baseline_mae = mae(y_test, baseline_pred)
    baseline_r2 = r_squared(y_test, baseline_pred)

    results = [{"Group": "ALL FEATURES (baseline)", "RMSE": round(baseline_rmse, 4),
                "MAE": round(baseline_mae, 4), "R²": round(baseline_r2, 4),
                "RMSE Δ%": 0.0}]

    groups = FEATURE_GROUPS[domain]
    for group_name, cols in groups.items():
        # Zero out the feature group (keeps shape, simulates removal)
        X_ablated = X_test.copy()
        existing = [c for c in cols if c in X_ablated.columns]
        if not existing:
            continue
        X_ablated[existing] = 0
        pred = _ens.predict(domain, X_ablated)["forecast"]
        r = rmse(y_test, pred)
        m = mae(y_test, pred)
        r2 = r_squared(y_test, pred)
        delta = (r - baseline_rmse) / baseline_rmse * 100
        results.append({"Group": f"Drop: {group_name}", "RMSE": round(r, 4),
                        "MAE": round(m, 4), "R²": round(r2, 4),
                        "RMSE Δ%": round(delta, 2)})
    return pd.DataFrame(results)


try:
    ens = load_ensemble()
except Exception as e:
    st.error(f"Load models first: {e}")
    st.stop()

domain = st.selectbox("Domain", ["airline", "ecommerce", "payment"])

st.markdown("Zeroing out each feature group and measuring RMSE increase. "
            "Higher **RMSE Δ%** = more critical feature group.")

results = run_ablation(ens, domain)
st.dataframe(results, hide_index=True, use_container_width=True)

# Highlight most impactful
if len(results) > 1:
    worst = results.iloc[1:].sort_values("RMSE Δ%", ascending=False).iloc[0]
    st.warning(f"Most impactful group: **{worst['Group']}** — removing it increases RMSE by **{worst['RMSE Δ%']:.1f}%**")
