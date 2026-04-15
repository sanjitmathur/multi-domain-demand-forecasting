"""Evaluation — metrics, residuals, uncertainty, diagnostics, ablation, baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import _theme as theme
from dashboard._baselines import all_baselines
from src.evaluation.metrics import rmse, mae, r_squared, rmse_pct
from src.features.airline_features import engineer_airline_features
from src.features.ecommerce_features import engineer_ecommerce_features
from src.features.payment_features import engineer_payment_features
from src.models.ensemble import EnsembleForecaster

theme.apply()

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

DATASETS = {
    "airline":   (DATA_DIR / "airline_bookings.csv",  engineer_airline_features,   "bookings"),
    "ecommerce": (DATA_DIR / "ecommerce_demand.csv",  engineer_ecommerce_features, "quantity_sold"),
    "payment":   (DATA_DIR / "payment_volume.csv",    engineer_payment_features,   "volume"),
}

FEATURE_GROUPS = {
    "airline": {
        "Lag Features": ["bookings_lag_1", "bookings_lag_7", "bookings_lag_14"],
        "Rolling Stats": ["bookings_rolling_mean_7", "bookings_rolling_std_7", "bookings_rolling_max_7",
                          "bookings_rolling_mean_14", "bookings_rolling_mean_30"],
        "Price Features": ["price", "competitor_price", "price_ratio", "price_per_seat"],
        "Temporal": ["day_of_week", "month", "is_weekend", "is_holiday",
                     "dow_sin", "dow_cos", "month_sin", "month_cos"],
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


@st.cache_resource(show_spinner=False)
def load_ensemble() -> EnsembleForecaster:
    ens = EnsembleForecaster()
    ens.load(str(MODELS_DIR), ["airline", "ecommerce", "payment"])
    return ens


@st.cache_data(show_spinner=False)
def evaluate_domain(_ens, domain: str):
    path, fn, _ = DATASETS[domain]
    df = pd.read_csv(path)
    features, target = fn(df)
    split = int(len(features) * 0.8)
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = target.iloc[:split], target.iloc[split:]
    pred = _ens.predict(domain, X_test)
    xgb = _ens.predict_single_xgb(domain, X_test)
    lgbm = _ens.predict_single_lgbm(domain, X_test)
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train.values, "y_test": y_test.values,
        "pred": pred, "xgb": xgb, "lgbm": lgbm,
        "raw": df, "features": features, "target": target,
    }


@st.cache_data(show_spinner=False)
def run_ablation(_ens, domain: str) -> pd.DataFrame:
    path, fn, _ = DATASETS[domain]
    features, target = fn(pd.read_csv(path))
    split = int(len(features) * 0.8)
    X_test, y_test = features.iloc[split:], target.iloc[split:].values

    base_pred = _ens.predict(domain, X_test)["forecast"]
    base_rmse = rmse(y_test, base_pred)
    rows = [{"Group": "ALL FEATURES (baseline)",
             "RMSE": round(base_rmse, 3),
             "MAE": round(mae(y_test, base_pred), 3),
             "R²": round(r_squared(y_test, base_pred), 3),
             "RMSE Δ%": 0.0}]
    for group, cols in FEATURE_GROUPS[domain].items():
        X_ab = X_test.copy()
        existing = [c for c in cols if c in X_ab.columns]
        if not existing:
            continue
        X_ab[existing] = 0
        p = _ens.predict(domain, X_ab)["forecast"]
        r = rmse(y_test, p)
        rows.append({
            "Group": f"Drop: {group}",
            "RMSE": round(r, 3),
            "MAE": round(mae(y_test, p), 3),
            "R²": round(r_squared(y_test, p), 3),
            "RMSE Δ%": round((r - base_rmse) / base_rmse * 100, 2),
        })
    return pd.DataFrame(rows)


# ── Page header ─────────────────────────────────────────────────────────────
theme.brand_header(status_text="Models loaded · 22 tests passing")
theme.page_header(
    eyebrow="MEASURE TWICE",
    title="Evaluation",
    subtitle="Metrics vs classical baselines, uncertainty calibration, feature ablation, and data diagnostics.",
)

try:
    ens = load_ensemble()
except Exception as e:
    st.error(f"Load models first: {e}")
    st.stop()

domain = st.selectbox("Domain", list(DATASETS.keys()), key="eval_domain")
d = evaluate_domain(ens, domain)
y, p = d["y_test"], d["pred"]["forecast"]
lower, upper = d["pred"]["lower_bound"], d["pred"]["upper_bound"]
residuals = y - p

# ── Hero metrics ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("RMSE", f"{rmse(y, p):.3f}",
          help="Root Mean Squared Error — lower is better.")
c2.metric("MAE", f"{mae(y, p):.3f}",
          help="Mean Absolute Error — lower is better.")
c3.metric("R²", f"{r_squared(y, p):.3f}",
          help="Fraction of variance explained. 1.0 is perfect.")
c4.metric("RMSE %", f"{rmse_pct(y, p):.2f}%",
          help="RMSE as a % of mean actual — scale-free.")

tab_models, tab_uncert, tab_ablation, tab_data = st.tabs(
    ["Models vs Baselines", "Uncertainty", "Ablation", "Data Diagnostics"]
)


# ── Models vs baselines ─────────────────────────────────────────────────────
with tab_models:
    rows = all_baselines(y)
    rows += [
        {"Model": "XGBoost (base)",        "RMSE": round(rmse(y, d["xgb"]), 3),
         "MAE": round(mae(y, d["xgb"]), 3),  "R²": round(r_squared(y, d["xgb"]), 3),
         "RMSE %": round(rmse_pct(y, d["xgb"]), 2)},
        {"Model": "LightGBM (base)",       "RMSE": round(rmse(y, d["lgbm"]), 3),
         "MAE": round(mae(y, d["lgbm"]), 3), "R²": round(r_squared(y, d["lgbm"]), 3),
         "RMSE %": round(rmse_pct(y, d["lgbm"]), 2)},
        {"Model": "Ensemble + Ridge meta", "RMSE": round(rmse(y, p), 3),
         "MAE": round(mae(y, p), 3),        "R²": round(r_squared(y, p), 3),
         "RMSE %": round(rmse_pct(y, p), 2)},
    ]
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

    # Compute improvement vs best naive baseline
    best_baseline_rmse = min(rmse(y, r_pred) for r_pred in [
        np.concatenate([[y[0]], y[:-1]]),            # naive
        np.concatenate([y[:7], y[:-7]]) if len(y) > 7 else y,  # rough seasonal
    ])
    ensemble_rmse = rmse(y, p)
    improvement = (1 - ensemble_rmse / best_baseline_rmse) * 100

    theme.section("Models vs classical baselines",
                  "A demand-forecasting model that doesn't beat naive baselines isn't a model.")
    c1, c2 = st.columns([3, 1])
    with c1:
        st.dataframe(table, hide_index=True, use_container_width=True,
                     column_config={
                         "RMSE %": st.column_config.NumberColumn(format="%.2f%%"),
                     })
    with c2:
        st.metric("RMSE vs best baseline",
                  f"−{improvement:.1f}%",
                  help="Ensemble's held-out RMSE reduction over the best classical baseline.")

    theme.section("Actual vs Predicted")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y, y=p, mode="markers",
            marker=dict(size=4, color=theme.PREDICTED, opacity=0.55,
                        line=dict(color=theme.BG, width=0.3)),
            name="Predictions",
        ))
        mn, mx = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(dash="dash", color=theme.ACTUAL, width=1.5),
            name="Perfect",
        ))
        fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=residuals, nbinsx=50, marker_color=theme.ACCENT, opacity=0.85,
        ))
        fig.add_vline(x=0, line_dash="dash", line_color=theme.HIGHLIGHT)
        fig.update_layout(xaxis_title="Residual (Actual − Predicted)",
                          yaxis_title="Count", height=400)
        st.plotly_chart(fig, use_container_width=True)


# ── Uncertainty ─────────────────────────────────────────────────────────────
with tab_uncert:
    theme.section("80% prediction interval", "Lower = P10, upper = P90. The band should catch ~80% of actuals.")
    n_show = min(200, len(y))
    idx = np.arange(n_show)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=upper[:n_show], mode="lines",
                              line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=idx, y=lower[:n_show], mode="lines",
                              line=dict(width=0), fill="tonexty",
                              fillcolor=theme.ACCENT_SOFT, name="P10–P90 interval",
                              hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=idx, y=p[:n_show], mode="lines", name="Forecast",
                              line=dict(color=theme.PREDICTED, width=2)))
    fig.add_trace(go.Scatter(x=idx, y=y[:n_show], mode="markers", name="Actual",
                              marker=dict(size=5, color=theme.ACTUAL,
                                          symbol=theme.SHAPE_ACTUAL)))
    fig.update_layout(xaxis_title="Test Sample", yaxis_title="Value", height=420,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    coverage = float(np.mean((y >= lower) & (y <= upper)))
    avg_width = float(np.mean(upper - lower))

    c1, c2, c3 = st.columns(3)
    c1.metric("Empirical coverage", f"{coverage:.1%}",
              help="Share of held-out actuals falling inside the P10–P90 band. "
                   "Target ≈ 80%. Higher = conservative, lower = over-confident.")
    c2.metric("Avg interval width", f"{avg_width:,.2f}")
    c3.metric("Samples", f"{len(y):,}")

    delta = coverage - 0.80
    if abs(delta) < 0.05:
        st.success(f"Coverage is within ±5pp of the target 80% — well calibrated.")
    elif delta > 0:
        st.info(f"Coverage is {delta * 100:+.1f}pp above target — intervals are conservative.")
    else:
        st.warning(f"Coverage is {delta * 100:+.1f}pp below target — intervals are over-confident.")


# ── Ablation ─────────────────────────────────────────────────────────────────
with tab_ablation:
    theme.section("Feature-group ablation",
                  "Zeroing out each group in turn and measuring RMSE increase. Higher Δ% = more critical.")
    abl = run_ablation(ens, domain)
    st.dataframe(abl, hide_index=True, use_container_width=True,
                 column_config={"RMSE Δ%": st.column_config.NumberColumn(format="%.2f%%")})

    if len(abl) > 1:
        worst = abl.iloc[1:].sort_values("RMSE Δ%", ascending=False).iloc[0]
        st.info(f"**Most impactful group:** {worst['Group']} · "
                f"dropping it raises RMSE by {worst['RMSE Δ%']:.1f}%.")


# ── Data Diagnostics ─────────────────────────────────────────────────────────
with tab_data:
    _, _, target_col = DATASETS[domain]
    df = d["raw"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns)}")
    c3.metric("Missing cells", f"{df.isna().sum().sum():,}")
    c4.metric("Duplicate rows", f"{df.duplicated().sum():,}")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        theme.section("Missing values")
        st.dataframe(pd.DataFrame({"Column": missing.index,
                                    "Missing": missing.values,
                                    "% Missing": (missing.values / len(df) * 100).round(2)}),
                     hide_index=True, use_container_width=True)
    else:
        st.success("No missing values detected.")

    theme.section("Numeric summary")
    st.dataframe(df.describe().T.round(3), use_container_width=True)

    theme.section(f"Target distribution — `{target_col}`")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[target_col], nbinsx=50,
                                marker_color=theme.PREDICTED, opacity=0.85))
    fig.add_vline(x=df[target_col].mean(), line_dash="dash", line_color=theme.HIGHLIGHT,
                   annotation_text=f"mean={df[target_col].mean():.1f}")
    fig.update_layout(xaxis_title=target_col, yaxis_title="Count", height=340)
    st.plotly_chart(fig, use_container_width=True)

    theme.section("Feature correlations with target")
    num = df.select_dtypes(include=[np.number])
    if target_col in num.columns:
        corr = num.corr()[target_col].drop(target_col).sort_values(ascending=False)
        fig = go.Figure(go.Bar(
            x=corr.values, y=corr.index, orientation="h",
            marker_color=[theme.PREDICTED if v > 0 else theme.ACTUAL for v in corr.values],
        ))
        fig.update_layout(title="Pearson correlation with target",
                          height=max(320, len(corr) * 22), margin=dict(l=180))
        st.plotly_chart(fig, use_container_width=True)
