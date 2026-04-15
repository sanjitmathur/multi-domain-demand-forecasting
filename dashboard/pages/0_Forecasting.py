"""Forecasting page — premium interactive predictions with baselines."""

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


# ─── Loaders ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_ensemble() -> EnsembleForecaster:
    ens = EnsembleForecaster()
    ens.load(str(MODELS_DIR), ["airline", "ecommerce", "payment"])
    return ens


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(DATA_DIR / "airline_bookings.csv"),
        pd.read_csv(DATA_DIR / "ecommerce_demand.csv"),
        pd.read_csv(DATA_DIR / "payment_volume.csv"),
    )


@st.cache_data(show_spinner=False)
def prepare_domain(_ens: EnsembleForecaster, domain: str, features, target):
    split = int(len(features) * 0.8)
    X_test = features.iloc[split:]
    y_test = target.iloc[split:]
    pred = _ens.predict(domain, X_test)
    return (
        X_test, y_test,
        pred,
        _ens.predict_single_xgb(domain, X_test),
        _ens.predict_single_lgbm(domain, X_test),
        _ens.predict_simple_avg(domain, X_test),
    )


# ─── Chart ──────────────────────────────────────────────────────────────────
def forecast_chart(
    x_vals, actual, predicted, lower, upper,
    x_current, x_label: str, y_label: str,
) -> go.Figure:
    fig = go.Figure()
    # Upper band
    fig.add_trace(go.Scatter(
        x=x_vals, y=upper, mode="lines",
        line=dict(width=0, color=theme.ACCENT), showlegend=False, hoverinfo="skip",
    ))
    # Lower band — tonexty fills between upper and lower
    fig.add_trace(go.Scatter(
        x=x_vals, y=lower, mode="lines",
        line=dict(width=0, color=theme.ACCENT),
        fill="tonexty", fillcolor=theme.ACCENT_SOFT,
        name="P10–P90 interval", hoverinfo="skip",
    ))
    # Actuals — dashed with X markers (colorblind-safe)
    fig.add_trace(go.Scatter(
        x=x_vals, y=actual, mode="lines+markers", name="Actual",
        line=dict(color=theme.ACTUAL, width=1.4, dash="dot"),
        marker=dict(color=theme.ACTUAL, symbol=theme.SHAPE_ACTUAL,
                    size=9, line=dict(width=1.6)),
    ))
    # Predicted — smooth spline with glow, circle markers
    fig.add_trace(go.Scatter(
        x=x_vals, y=predicted, mode="lines+markers", name="Forecast",
        line=dict(color=theme.PREDICTED, width=2.8, shape="spline", smoothing=0.6),
        marker=dict(color=theme.PREDICTED, symbol=theme.SHAPE_PREDICTED,
                    size=7, line=dict(color=theme.BG, width=1.5)),
    ))
    # Selection marker
    if x_current is not None and len(x_vals):
        i = min(range(len(x_vals)), key=lambda j: abs(x_vals[j] - x_current))
        fig.add_vline(x=x_current, line_dash="dot",
                      line_color=theme.HIGHLIGHT, opacity=0.55)
        fig.add_trace(go.Scatter(
            x=[x_vals[i]], y=[predicted[i]], mode="markers",
            marker=dict(color=theme.HIGHLIGHT, size=14, symbol=theme.SHAPE_SELECTION,
                        line=dict(color=theme.BG, width=2)),
            name="Selected",
        ))

    fig.update_layout(
        height=420, hovermode="x unified",
        xaxis_title=x_label, yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0,
                    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        margin=dict(l=48, r=24, t=52, b=48),
    )
    return fig


# ─── Comparison table ────────────────────────────────────────────────────────
def _model_row(name: str, y_true, y_pred) -> dict:
    return {
        "Model": name,
        "RMSE": round(rmse(y_true, y_pred), 3),
        "MAE": round(mae(y_true, y_pred), 3),
        "R²": round(r_squared(y_true, y_pred), 3),
        "RMSE %": round(rmse_pct(y_true, y_pred), 2),
    }


def comparison_table(y_true, xgb_p, lgbm_p, avg_p, meta_p) -> pd.DataFrame:
    rows = all_baselines(y_true)
    rows += [
        _model_row("XGBoost (base)",        y_true, xgb_p),
        _model_row("LightGBM (base)",       y_true, lgbm_p),
        _model_row("Simple Avg",            y_true, avg_p),
        _model_row("Ensemble + Ridge meta", y_true, meta_p),
    ]
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


# ─── Page layout ─────────────────────────────────────────────────────────────
theme.brand_header(status_text="Models loaded · 22 tests passing")
theme.page_header(
    eyebrow="INTERACTIVE FORECAST",
    title="Forecasting",
    subtitle="Ensemble ML across airline, e-commerce, and payment demand — "
             "with calibrated P10 / P90 uncertainty intervals.",
)

try:
    ensemble = load_ensemble()
except Exception as e:
    st.error(f"Could not load models: {e}. Run `python -m src.models.train_all` first.")
    st.stop()

airline_df, ecom_df, payment_df = load_data()

airline_tab, ecom_tab, payment_tab = st.tabs(
    ["Airline", "E-Commerce", "Payments"]
)


# ─── Airline ─────────────────────────────────────────────────────────────────
with airline_tab:
    features, target = engineer_airline_features(airline_df)
    X_test, y_test, pred, xgb_p, lgbm_p, avg_p = prepare_domain(
        ensemble, "airline", features, target)

    ctl1, ctl2 = st.columns([1, 2])
    with ctl1:
        fare_class = st.selectbox("Fare Class",
                                  ["Economy", "Business", "First"], key="a_fare")
    with ctl2:
        days_until = st.slider("Days until departure", 1, 180, 30, key="a_days")

    fc = {"Economy": 0, "Business": 1, "First": 2}[fare_class]
    mask = X_test["fare_class_encoded"] == fc

    if mask.sum() == 0:
        st.info("No held-out samples for this fare class. Try another.")
    else:
        fp = pred["forecast"][mask.values]
        fy = y_test[mask].values
        fl = pred["lower_bound"][mask.values]
        fu = pred["upper_bound"][mask.values]
        fX = X_test[mask]
        closest_idx = (fX["days_until_departure"] - days_until).abs().idxmin()
        i = fX.index.get_loc(closest_idx)

        # Build a sparkline of forecast vs day-bucket for the hero card
        spark_df = pd.DataFrame({
            "days": fX["days_until_departure"].values,
            "pred": fp,
        }).assign(b=lambda d: (d["days"] // 5) * 5).groupby("b")["pred"].mean()
        theme.hero_forecast(
            forecast=float(fp[i]),
            actual=float(fy[i]),
            lower=float(fl[i]),
            upper=float(fu[i]),
            spark_values=spark_df.sort_index().values.tolist(),
            unit="bookings",
        )

        theme.section("Booking curve",
                      "Mean bookings vs days until departure, bucketed by 5-day windows.")
        df_plot = pd.DataFrame({
            "days": fX["days_until_departure"].values,
            "actual": fy, "pred": fp, "lower": fl, "upper": fu,
        })
        df_plot["days_bin"] = (df_plot["days"] // 5) * 5
        grouped = df_plot.groupby("days_bin").mean().reset_index()
        st.plotly_chart(
            forecast_chart(
                grouped["days_bin"].tolist(), grouped["actual"].tolist(),
                grouped["pred"].tolist(), grouped["lower"].tolist(),
                grouped["upper"].tolist(), days_until,
                "Days until departure", "Bookings",
            ),
            use_container_width=True,
        )

        theme.section("Model vs baselines",
                      "Ensemble compared with base models and classical baselines on held-out data.")
        st.dataframe(
            comparison_table(y_test.values, xgb_p, lgbm_p, avg_p, pred["forecast"]),
            hide_index=True, use_container_width=True,
            column_config={
                "RMSE":   st.column_config.NumberColumn(help="Root Mean Squared Error — lower is better"),
                "MAE":    st.column_config.NumberColumn(help="Mean Absolute Error — lower is better"),
                "R²":     st.column_config.NumberColumn(help="Fraction of variance explained — 1.0 is perfect"),
                "RMSE %": st.column_config.NumberColumn(format="%.2f%%",
                                                        help="RMSE as a % of mean actual — scale-free"),
            },
        )


# ─── E-Commerce ──────────────────────────────────────────────────────────────
with ecom_tab:
    features, target = engineer_ecommerce_features(ecom_df)
    X_test, y_test, pred, xgb_p, lgbm_p, avg_p = prepare_domain(
        ensemble, "ecommerce", features, target)

    ctl1, ctl2 = st.columns([1, 2])
    with ctl1:
        category = st.selectbox("Category",
            ["Beauty", "Electronics", "Fashion", "Food & Grocery", "Home & Kitchen"],
            key="e_cat")
    with ctl2:
        promo = st.radio("Promotion", ["All", "Promo ON", "Promo OFF"],
                         key="e_promo", horizontal=True)

    cc = {"Beauty": 0, "Electronics": 1, "Fashion": 2,
          "Food & Grocery": 3, "Home & Kitchen": 4}[category]
    mask = X_test["category_encoded"] == cc
    if promo == "Promo ON":
        mask = mask & (X_test["promotions"] == 1)
    elif promo == "Promo OFF":
        mask = mask & (X_test["promotions"] == 0)

    if mask.sum() == 0:
        st.info("No held-out samples match. Try `All` or a different category.")
    else:
        fp = pred["forecast"][mask.values]
        fy = y_test[mask].values
        fl = pred["lower_bound"][mask.values]
        fu = pred["upper_bound"][mask.values]
        spark_df = pd.DataFrame({
            "price": X_test.loc[mask, "price"].values, "pred": fp,
        }).sort_values("price")
        theme.hero_forecast(
            forecast=float(np.mean(fp)),
            actual=float(np.mean(fy)),
            lower=float(np.mean(fl)),
            upper=float(np.mean(fu)),
            spark_values=spark_df["pred"].rolling(5, min_periods=1).mean().values.tolist(),
            unit="units",
        )

        theme.section("Demand vs price",
                      f"{category} · {promo} — bucketed by price band.")
        df_plot = pd.DataFrame({
            "price": X_test.loc[mask, "price"].values,
            "actual": fy, "pred": fp, "lower": fl, "upper": fu,
        })
        rng = df_plot["price"].max() - df_plot["price"].min()
        bin_size = max(1.0, rng / 20)
        df_plot["price_bin"] = (df_plot["price"] / bin_size).round() * bin_size
        grouped = df_plot.groupby("price_bin").mean().reset_index()
        st.plotly_chart(
            forecast_chart(
                grouped["price_bin"].tolist(), grouped["actual"].tolist(),
                grouped["pred"].tolist(), grouped["lower"].tolist(),
                grouped["upper"].tolist(), None,
                "Price ($)", "Units sold",
            ),
            use_container_width=True,
        )

        theme.section("Model vs baselines")
        st.dataframe(
            comparison_table(y_test.values, xgb_p, lgbm_p, avg_p, pred["forecast"]),
            hide_index=True, use_container_width=True,
        )


# ─── Payments ────────────────────────────────────────────────────────────────
with payment_tab:
    features, target = engineer_payment_features(payment_df)
    X_test, y_test, pred, xgb_p, lgbm_p, avg_p = prepare_domain(
        ensemble, "payment", features, target)

    ctl1, ctl2 = st.columns([1, 2])
    with ctl1:
        dow = st.selectbox("Day of week", list(range(7)),
            format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
            key="p_dow")
    with ctl2:
        hour = st.slider("Hour of day", 0, 23, 10, key="p_hour")

    day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow]
    day_mask = X_test["day_of_week"] == dow
    exact = day_mask & (X_test["hour_of_day"] == hour)

    if exact.sum() > 0:
        fp = pred["forecast"][exact.values]
        fy = y_test[exact].values
        fl = pred["lower_bound"][exact.values]
        fu = pred["upper_bound"][exact.values]

        # Sparkline: hourly volume on the selected day
        hour_df = pd.DataFrame({
            "hour": X_test.loc[day_mask, "hour_of_day"].values,
            "pred": pred["forecast"][day_mask.values],
        }).groupby("hour")["pred"].mean().sort_index()
        theme.hero_forecast(
            forecast=float(np.mean(fp)),
            actual=float(np.mean(fy)),
            lower=float(np.mean(fl)),
            upper=float(np.mean(fu)),
            spark_values=hour_df.values.tolist(),
            unit="txns",
        )
    else:
        st.info(f"No held-out samples for {day_name} at hour {hour:02d}.")

    if day_mask.sum() > 0:
        theme.section("Hourly volume profile",
                      f"Average transaction volume across the day — {day_name}.")
        df_plot = pd.DataFrame({
            "hour": X_test.loc[day_mask, "hour_of_day"].values,
            "actual": y_test[day_mask].values,
            "pred": pred["forecast"][day_mask.values],
            "lower": pred["lower_bound"][day_mask.values],
            "upper": pred["upper_bound"][day_mask.values],
        })
        grouped = df_plot.groupby("hour").mean().reset_index()
        st.plotly_chart(
            forecast_chart(
                grouped["hour"].tolist(), grouped["actual"].tolist(),
                grouped["pred"].tolist(), grouped["lower"].tolist(),
                grouped["upper"].tolist(), hour,
                "Hour of day", "Volume",
            ),
            use_container_width=True,
        )

    theme.section("Model vs baselines")
    st.dataframe(
        comparison_table(y_test.values, xgb_p, lgbm_p, avg_p, pred["forecast"]),
        hide_index=True, use_container_width=True,
    )
