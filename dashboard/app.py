"""Streamlit dashboard for multi-domain demand forecasting."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ensemble import EnsembleForecaster
from src.features.airline_features import engineer_airline_features
from src.features.ecommerce_features import engineer_ecommerce_features
from src.features.payment_features import engineer_payment_features
from src.evaluation.metrics import rmse, mae, r_squared, rmse_pct
from api.schemas import AirlineRequest, ECommerceRequest, PaymentRequest
from api.main import _build_airline_features, _build_ecommerce_features, _build_payment_features

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"


@st.cache_resource
def load_ensemble():
    ens = EnsembleForecaster()
    ens.load(str(MODELS_DIR), ["airline", "ecommerce", "payment"])
    return ens


@st.cache_data
def load_data():
    airline = pd.read_csv(DATA_DIR / "airline_bookings.csv")
    ecom = pd.read_csv(DATA_DIR / "ecommerce_demand.csv")
    payment = pd.read_csv(DATA_DIR / "payment_volume.csv")
    return airline, ecom, payment


def sweep_predict(ensemble, domain, build_fn, requests):
    """Run predictions for a list of requests and return arrays."""
    frames = [build_fn(r) for r in requests]
    X = pd.concat(frames, ignore_index=True)
    result = ensemble.predict(domain, X)
    return result


def plot_sweep(x_vals, forecast, lower, upper, x_current, title, x_label, y_label):
    """Plot forecast curve with confidence band and a vertical marker at current value."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals, y=upper, mode="lines", line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=lower, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(68, 68, 255, 0.15)",
        name="80% Confidence Band",
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=forecast, mode="lines",
        name="Ensemble+Meta Forecast", line=dict(color="#4ECDC4", width=2.5),
    ))

    # Marker at current slider value
    idx = None
    for i, v in enumerate(x_vals):
        if v == x_current:
            idx = i
            break
    if idx is not None:
        fig.add_trace(go.Scatter(
            x=[x_current], y=[forecast[idx]],
            mode="markers", marker=dict(color="#FF6B6B", size=14, symbol="diamond"),
            name=f"Current ({x_label}={x_current})",
        ))
        # Vertical line
        fig.add_vline(x=x_current, line_dash="dot", line_color="#FF6B6B", opacity=0.5)

    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title=y_label,
        template="plotly_dark", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_actual_vs_predicted(actual, predicted, lower, upper, title):
    """Static test-set chart: actual vs predicted."""
    fig = go.Figure()
    x = list(range(len(actual)))

    fig.add_trace(go.Scatter(
        x=x, y=upper, mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=lower, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(68, 68, 255, 0.1)",
        name="80% Confidence Band",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=actual, mode="lines",
        name="Actual", line=dict(color="#FF6B6B", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=predicted, mode="lines",
        name="Predicted", line=dict(color="#4ECDC4", width=1.5),
    ))

    fig.update_layout(
        title=title, xaxis_title="Test Sample", yaxis_title="Value",
        template="plotly_dark", height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def metrics_table(y_true, xgb_pred, lgbm_pred, avg_pred, meta_pred):
    rows = []
    for name, pred in [("XGBoost", xgb_pred), ("LightGBM", lgbm_pred),
                        ("Simple Avg", avg_pred), ("Ensemble+Meta", meta_pred)]:
        rows.append({
            "Model": name,
            "RMSE": round(rmse(y_true, pred), 4),
            "MAE": round(mae(y_true, pred), 4),
            "R²": round(r_squared(y_true, pred), 4),
            "RMSE %": round(rmse_pct(y_true, pred), 2),
        })
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")
    st.title("Multi-Domain Demand Forecasting")

    try:
        ensemble = load_ensemble()
    except Exception as e:
        st.error(f"Could not load models: {e}. Run `python -m src.models.train_all` first.")
        return

    airline_df, ecom_df, payment_df = load_data()

    tab1, tab2, tab3 = st.tabs(["Airline Bookings", "E-Commerce Demand", "Payment Volume"])

    # ── Airline Tab ──
    with tab1:
        st.header("Airline Booking Forecasting")

        col1, col2 = st.columns([2, 1])
        with col2:
            st.subheader("Parameters")
            fare_class = st.selectbox("Fare Class", ["Economy", "Business", "First"], key="a_fare")
            comp_price = st.slider("Competitor Price ($)", 100, 3000, 300, step=50, key="a_comp")
            days_until = st.slider("Days Until Departure", 1, 180, 30, key="a_days")

            # Single-point forecast at current slider
            feat = _build_airline_features(AirlineRequest(
                days_until_departure=days_until, fare_class=fare_class,
                competitor_price=float(comp_price),
            ))
            result = ensemble.predict("airline", feat)
            st.metric("Forecast", f"{result['forecast'][0]:.1f} bookings")
            st.metric("Confidence", f"{result['confidence'][0]:.1%}")
            st.caption(f"Range: [{result['lower_bound'][0]:.1f}, {result['upper_bound'][0]:.1f}]")

        with col1:
            # Sweep: forecast vs days_until_departure
            sweep_days = list(range(1, 181, 3))
            reqs = [AirlineRequest(days_until_departure=d, fare_class=fare_class,
                                   competitor_price=float(comp_price)) for d in sweep_days]
            sw = sweep_predict(ensemble, "airline", _build_airline_features, reqs)
            fig = plot_sweep(
                sweep_days, sw["forecast"], sw["lower_bound"], sw["upper_bound"],
                days_until, "Bookings vs Days Until Departure",
                "Days Until Departure", "Predicted Bookings",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Test-set accuracy section
        with st.expander("Test Set: Actual vs Predicted"):
            features, target = engineer_airline_features(airline_df)
            split = int(len(features) * 0.8)
            X_test, y_test = features.iloc[split:], target.iloc[split:]
            pred_result = ensemble.predict("airline", X_test)
            sl = min(200, len(y_test))
            fig2 = plot_actual_vs_predicted(
                y_test.values[:sl], pred_result["forecast"][:sl],
                pred_result["lower_bound"][:sl], pred_result["upper_bound"][:sl],
                "Airline: Actual vs Predicted (Test Set)",
            )
            st.plotly_chart(fig2, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Model Comparison")
                xgb_p = ensemble.predict_single_xgb("airline", X_test)
                lgbm_p = ensemble.predict_single_lgbm("airline", X_test)
                avg_p = ensemble.predict_simple_avg("airline", X_test)
                st.dataframe(metrics_table(y_test.values, xgb_p, lgbm_p, avg_p, pred_result["forecast"]), hide_index=True)

    # ── E-Commerce Tab ──
    with tab2:
        st.header("E-Commerce Demand Forecasting")

        col1, col2 = st.columns([2, 1])
        with col2:
            st.subheader("Parameters")
            category = st.selectbox("Category",
                ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Food & Grocery"], key="e_cat")
            promo = st.checkbox("Promotion Active", key="e_promo")
            price = st.slider("Price ($)", 5, 1000, 100, step=5, key="e_price")

            feat = _build_ecommerce_features(ECommerceRequest(
                product_category=category, price=float(price), promotion_active=promo,
            ))
            result = ensemble.predict("ecommerce", feat)
            st.metric("Forecast", f"{result['forecast'][0]:.0f} units")
            st.metric("Confidence", f"{result['confidence'][0]:.1%}")
            st.caption(f"Range: [{result['lower_bound'][0]:.0f}, {result['upper_bound'][0]:.0f}]")

        with col1:
            # Sweep: forecast vs price
            sweep_prices = list(range(5, 1001, 10))
            reqs = [ECommerceRequest(product_category=category, price=float(p),
                                     promotion_active=promo) for p in sweep_prices]
            sw = sweep_predict(ensemble, "ecommerce", _build_ecommerce_features, reqs)
            fig = plot_sweep(
                sweep_prices, sw["forecast"], sw["lower_bound"], sw["upper_bound"],
                price, f"Demand vs Price ({category}, {'Promo ON' if promo else 'Promo OFF'})",
                "Price ($)", "Predicted Units Sold",
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Test Set: Actual vs Predicted"):
            features, target = engineer_ecommerce_features(ecom_df)
            split = int(len(features) * 0.8)
            X_test, y_test = features.iloc[split:], target.iloc[split:]
            pred_result = ensemble.predict("ecommerce", X_test)
            sl = min(200, len(y_test))
            fig2 = plot_actual_vs_predicted(
                y_test.values[:sl], pred_result["forecast"][:sl],
                pred_result["lower_bound"][:sl], pred_result["upper_bound"][:sl],
                "E-Commerce: Actual vs Predicted (Test Set)",
            )
            st.plotly_chart(fig2, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Model Comparison")
                xgb_p = ensemble.predict_single_xgb("ecommerce", X_test)
                lgbm_p = ensemble.predict_single_lgbm("ecommerce", X_test)
                avg_p = ensemble.predict_simple_avg("ecommerce", X_test)
                st.dataframe(metrics_table(y_test.values, xgb_p, lgbm_p, avg_p, pred_result["forecast"]), hide_index=True)

    # ── Payment Tab ──
    with tab3:
        st.header("Payment Volume Forecasting")

        col1, col2 = st.columns([2, 1])
        with col2:
            st.subheader("Parameters")
            dow = st.selectbox("Day of Week", list(range(7)),
                format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x], key="p_dow")
            hour = st.slider("Hour of Day", 0, 23, 10, key="p_hour")

            feat = _build_payment_features(PaymentRequest(hour_of_day=hour, day_of_week=dow))
            result = ensemble.predict("payment", feat)
            st.metric("Forecast", f"{result['forecast'][0]:.0f} transactions")
            st.metric("Confidence", f"{result['confidence'][0]:.1%}")
            st.caption(f"Range: [{result['lower_bound'][0]:.0f}, {result['upper_bound'][0]:.0f}]")

        with col1:
            # Sweep: forecast vs hour_of_day
            sweep_hours = list(range(0, 24))
            reqs = [PaymentRequest(hour_of_day=h, day_of_week=dow) for h in sweep_hours]
            sw = sweep_predict(ensemble, "payment", _build_payment_features, reqs)
            day_name = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][dow]
            fig = plot_sweep(
                sweep_hours, sw["forecast"], sw["lower_bound"], sw["upper_bound"],
                hour, f"Transaction Volume by Hour ({day_name})",
                "Hour of Day", "Predicted Volume",
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Test Set: Actual vs Predicted"):
            features, target = engineer_payment_features(payment_df)
            split = int(len(features) * 0.8)
            X_test, y_test = features.iloc[split:], target.iloc[split:]
            pred_result = ensemble.predict("payment", X_test)
            sl = min(300, len(y_test))
            fig2 = plot_actual_vs_predicted(
                y_test.values[:sl], pred_result["forecast"][:sl],
                pred_result["lower_bound"][:sl], pred_result["upper_bound"][:sl],
                "Payment: Actual vs Predicted (Test Set)",
            )
            st.plotly_chart(fig2, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Model Comparison")
                xgb_p = ensemble.predict_single_xgb("payment", X_test)
                lgbm_p = ensemble.predict_single_lgbm("payment", X_test)
                avg_p = ensemble.predict_simple_avg("payment", X_test)
                st.dataframe(metrics_table(y_test.values, xgb_p, lgbm_p, avg_p, pred_result["forecast"]), hide_index=True)


if __name__ == "__main__":
    main()
