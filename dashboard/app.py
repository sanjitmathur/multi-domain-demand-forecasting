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


def plot_forecast(actual, predicted, lower, upper, title, x_label="Index"):
    fig = go.Figure()
    x = list(range(len(actual)))

    fig.add_trace(go.Scatter(
        x=x, y=upper, mode="lines", line=dict(width=0),
        showlegend=False, name="Upper Bound",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=lower, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(68, 68, 255, 0.15)",
        name="80% Confidence Band",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=actual, mode="lines",
        name="Actual", line=dict(color="#FF6B6B", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=predicted, mode="lines",
        name="Predicted (Ensemble+Meta)", line=dict(color="#4ECDC4", width=2),
    ))

    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title="Value",
        template="plotly_dark", height=450,
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
            "R-squared": round(r_squared(y_true, pred), 4),
            "RMSE %": round(rmse_pct(y_true, pred), 2),
        })
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")
    st.title("Multi-Domain Demand Forecasting")

    try:
        ensemble = load_ensemble()
    except Exception as e:
        st.error(f"Could not load models: {e}. Please run `python -m src.models.train_all` first.")
        return

    airline_df, ecom_df, payment_df = load_data()

    tab1, tab2, tab3 = st.tabs(["Airline Bookings", "E-Commerce Demand", "Payment Volume"])

    # --- Airline Tab ---
    with tab1:
        st.header("Airline Booking Forecasting")

        col1, col2 = st.columns([2, 1])
        with col2:
            st.subheader("Interactive Forecast")
            days_until = st.slider("Days Until Departure", 1, 180, 30, key="airline_days")
            fare_class = st.selectbox("Fare Class", ["Economy", "Business", "First"], key="airline_fare")
            comp_price = st.slider("Competitor Price ($)", 100, 3000, 300, key="airline_comp")

            from api.schemas import AirlineRequest
            from api.main import _build_airline_features
            feat = _build_airline_features(AirlineRequest(
                days_until_departure=days_until,
                fare_class=fare_class,
                competitor_price=float(comp_price),
            ))
            result = ensemble.predict("airline", feat)
            st.metric("Forecast", f"{result['forecast'][0]:.1f} bookings")
            st.metric("Confidence", f"{result['confidence'][0]:.1%}")
            st.caption(f"Range: [{result['lower_bound'][0]:.1f}, {result['upper_bound'][0]:.1f}]")

        with col1:
            features, target = engineer_airline_features(airline_df)
            n = len(features)
            split = int(n * 0.8)
            X_test = features.iloc[split:]
            y_test = target.iloc[split:]

            pred_result = ensemble.predict("airline", X_test)
            test_slice = min(200, len(y_test))

            fig = plot_forecast(
                y_test.values[:test_slice],
                pred_result["forecast"][:test_slice],
                pred_result["lower_bound"][:test_slice],
                pred_result["upper_bound"][:test_slice],
                "Airline: Actual vs Predicted Bookings",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Comparison")
        xgb_p = ensemble.predict_single_xgb("airline", X_test)
        lgbm_p = ensemble.predict_single_lgbm("airline", X_test)
        avg_p = ensemble.predict_simple_avg("airline", X_test)
        meta_p = pred_result["forecast"]
        st.dataframe(metrics_table(y_test.values, xgb_p, lgbm_p, avg_p, meta_p), hide_index=True)

    # --- E-Commerce Tab ---
    with tab2:
        st.header("E-Commerce Demand Forecasting")

        col1, col2 = st.columns([2, 1])
        with col2:
            st.subheader("Interactive Forecast")
            category = st.selectbox("Category", ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Food & Grocery"], key="ecom_cat")
            price = st.slider("Price ($)", 5, 1000, 100, key="ecom_price")
            promo = st.checkbox("Promotion Active", key="ecom_promo")

            from api.schemas import ECommerceRequest
            from api.main import _build_ecommerce_features
            feat = _build_ecommerce_features(ECommerceRequest(
                product_category=category, price=float(price), promotion_active=promo,
            ))
            result = ensemble.predict("ecommerce", feat)
            st.metric("Forecast", f"{result['forecast'][0]:.0f} units")
            st.metric("Confidence", f"{result['confidence'][0]:.1%}")
            st.caption(f"Range: [{result['lower_bound'][0]:.0f}, {result['upper_bound'][0]:.0f}]")

        with col1:
            features, target = engineer_ecommerce_features(ecom_df)
            n = len(features)
            split = int(n * 0.8)
            X_test = features.iloc[split:]
            y_test = target.iloc[split:]

            pred_result = ensemble.predict("ecommerce", X_test)
            test_slice = min(200, len(y_test))

            fig = plot_forecast(
                y_test.values[:test_slice],
                pred_result["forecast"][:test_slice],
                pred_result["lower_bound"][:test_slice],
                pred_result["upper_bound"][:test_slice],
                "E-Commerce: Actual vs Predicted Demand",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Comparison")
        xgb_p = ensemble.predict_single_xgb("ecommerce", X_test)
        lgbm_p = ensemble.predict_single_lgbm("ecommerce", X_test)
        avg_p = ensemble.predict_simple_avg("ecommerce", X_test)
        meta_p = pred_result["forecast"]
        st.dataframe(metrics_table(y_test.values, xgb_p, lgbm_p, avg_p, meta_p), hide_index=True)

    # --- Payment Tab ---
    with tab3:
        st.header("Payment Volume Forecasting")

        col1, col2 = st.columns([2, 1])
        with col2:
            st.subheader("Interactive Forecast")
            hour = st.slider("Hour of Day", 0, 23, 10, key="pay_hour")
            dow = st.selectbox("Day of Week", list(range(7)),
                               format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
                               key="pay_dow")

            from api.schemas import PaymentRequest
            from api.main import _build_payment_features
            feat = _build_payment_features(PaymentRequest(
                hour_of_day=hour, day_of_week=dow,
            ))
            result = ensemble.predict("payment", feat)
            st.metric("Forecast", f"{result['forecast'][0]:.0f} transactions")
            st.metric("Confidence", f"{result['confidence'][0]:.1%}")
            st.caption(f"Range: [{result['lower_bound'][0]:.0f}, {result['upper_bound'][0]:.0f}]")

        with col1:
            features, target = engineer_payment_features(payment_df)
            n = len(features)
            split = int(n * 0.8)
            X_test = features.iloc[split:]
            y_test = target.iloc[split:]

            pred_result = ensemble.predict("payment", X_test)
            test_slice = min(300, len(y_test))

            fig = plot_forecast(
                y_test.values[:test_slice],
                pred_result["forecast"][:test_slice],
                pred_result["lower_bound"][:test_slice],
                pred_result["upper_bound"][:test_slice],
                "Payment: Actual vs Predicted Volume",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Comparison")
        xgb_p = ensemble.predict_single_xgb("payment", X_test)
        lgbm_p = ensemble.predict_single_lgbm("payment", X_test)
        avg_p = ensemble.predict_simple_avg("payment", X_test)
        meta_p = pred_result["forecast"]
        st.dataframe(metrics_table(y_test.values, xgb_p, lgbm_p, avg_p, meta_p), hide_index=True)


if __name__ == "__main__":
    main()
