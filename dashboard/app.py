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


@st.cache_data
def prepare_domain(_ensemble, domain, features, target):
    """Run predictions on test set once, return everything needed."""
    n = len(features)
    split = int(n * 0.8)
    X_test = features.iloc[split:]
    y_test = target.iloc[split:]
    pred = _ensemble.predict(domain, X_test)
    xgb_p = _ensemble.predict_single_xgb(domain, X_test)
    lgbm_p = _ensemble.predict_single_lgbm(domain, X_test)
    avg_p = _ensemble.predict_simple_avg(domain, X_test)
    return X_test, y_test, pred, xgb_p, lgbm_p, avg_p


def plot_grouped(x_vals, actual_means, pred_means, lower_means, upper_means,
                 x_current, title, x_label, y_label, x_format=None):
    """Plot actual vs predicted grouped by a parameter, with current selection marked."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals, y=upper_means, mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=lower_means, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(68, 68, 255, 0.15)",
        name="Confidence Band",
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=actual_means, mode="lines+markers",
        name="Actual (avg)", line=dict(color="#FF6B6B", width=2.5),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=pred_means, mode="lines+markers",
        name="Predicted (avg)", line=dict(color="#4ECDC4", width=2.5),
        marker=dict(size=5),
    ))

    # Mark current selection
    if x_current is not None:
        fig.add_vline(x=x_current, line_dash="dot", line_color="#FFD93D", opacity=0.7)
        # Find nearest x value for annotation
        idx = min(range(len(x_vals)), key=lambda i: abs(x_vals[i] - x_current))
        fig.add_trace(go.Scatter(
            x=[x_vals[idx]], y=[pred_means[idx]],
            mode="markers", marker=dict(color="#FFD93D", size=14, symbol="diamond"),
            name=f"Selected",
        ))

    fmt = x_format or {}
    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title=y_label,
        template="plotly_dark", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **fmt,
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

    METRICS_LEGEND = (
        "**RMSE** = avg prediction error in original units (lower is better) | "
        "**MAE** = avg absolute error (lower is better) | "
        "**R\u00b2** = variance explained by model, 1.0 = perfect | "
        "**RMSE %** = error as % of mean actual value"
    )

    # ── Airline Tab ──
    with tab1:
        st.header("Airline Booking Forecasting")
        st.caption(
            "Predicts flight booking demand based on days until departure, fare class, and competitor pricing. "
            "Uses an XGBoost + LightGBM ensemble with a ridge meta-learner. "
            "The chart shows average actual vs predicted bookings grouped by departure window for the selected fare class."
        )
        features, target = engineer_airline_features(airline_df)
        X_test, y_test, pred, xgb_p, lgbm_p, avg_p = prepare_domain(
            ensemble, "airline", features, target)

        col1, col2 = st.columns([2, 1])

        with col2:
            st.subheader("Parameters")
            fare_class = st.selectbox("Fare Class", ["Economy", "Business", "First"], key="a_fare")
            days_until = st.slider("Days Until Departure", 1, 180, 30, key="a_days")

            fare_map = {"Economy": 0, "Business": 1, "First": 2}
            fc = fare_map[fare_class]

            # Filter test data by fare class
            mask = X_test["fare_class_encoded"] == fc
            if mask.sum() > 0:
                # Find closest days_until_departure
                filtered_X = X_test[mask]
                filtered_y = y_test[mask]
                filtered_pred = pred["forecast"][mask.values]
                filtered_lower = pred["lower_bound"][mask.values]
                filtered_upper = pred["upper_bound"][mask.values]
                filtered_conf = pred["confidence"][mask.values]

                closest_idx = (filtered_X["days_until_departure"] - days_until).abs().idxmin()
                i = filtered_X.index.get_loc(closest_idx)
                st.metric("Forecast", f"{filtered_pred[i]:.1f} bookings")
                st.metric("Actual", f"{filtered_y.iloc[i]:.0f} bookings")
                st.metric("Confidence", f"{filtered_conf[i]:.1%}")
                st.caption(f"Range: [{filtered_lower[i]:.1f}, {filtered_upper[i]:.1f}]")
            else:
                st.info("No test data for this fare class")

        with col1:
            # Group by days_until_departure for selected fare class
            mask = X_test["fare_class_encoded"] == fc
            if mask.sum() > 0:
                df_plot = pd.DataFrame({
                    "days": X_test.loc[mask, "days_until_departure"].values,
                    "actual": y_test[mask].values,
                    "pred": pred["forecast"][mask.values],
                    "lower": pred["lower_bound"][mask.values],
                    "upper": pred["upper_bound"][mask.values],
                })
                # Bin days into groups for smoother plot
                df_plot["days_bin"] = (df_plot["days"] // 5) * 5
                grouped = df_plot.groupby("days_bin").mean().reset_index()

                fig = plot_grouped(
                    grouped["days_bin"].tolist(),
                    grouped["actual"].tolist(),
                    grouped["pred"].tolist(),
                    grouped["lower"].tolist(),
                    grouped["upper"].tolist(),
                    days_until,
                    f"Bookings vs Days Until Departure ({fare_class})",
                    "Days Until Departure", "Bookings",
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Comparison")
        st.caption(METRICS_LEGEND)
        st.dataframe(metrics_table(y_test.values, xgb_p, lgbm_p, avg_p, pred["forecast"]), hide_index=True)

    # ── E-Commerce Tab ──
    with tab2:
        st.header("E-Commerce Demand Forecasting")
        st.caption(
            "Forecasts product demand by category using price, promotions, inventory levels, and seasonality. "
            "An ensemble of XGBoost + LightGBM with a meta-learner combines predictions. "
            "The chart shows average actual vs predicted demand grouped by price range for the selected category and promotion filter."
        )
        features, target = engineer_ecommerce_features(ecom_df)
        X_test, y_test, pred, xgb_p, lgbm_p, avg_p = prepare_domain(
            ensemble, "ecommerce", features, target)

        col1, col2 = st.columns([2, 1])

        with col2:
            st.subheader("Parameters")
            category = st.selectbox("Category",
                ["Beauty", "Electronics", "Fashion", "Food & Grocery", "Home & Kitchen"], key="e_cat")
            promo_filter = st.radio("Promotion", ["All", "Promo ON", "Promo OFF"], key="e_promo", horizontal=True)

            cat_map = {"Beauty": 0, "Electronics": 1, "Fashion": 2, "Food & Grocery": 3, "Home & Kitchen": 4}
            cc = cat_map[category]

            mask = X_test["category_encoded"] == cc
            if promo_filter == "Promo ON":
                mask = mask & (X_test["promotions"] == 1)
            elif promo_filter == "Promo OFF":
                mask = mask & (X_test["promotions"] == 0)

            if mask.sum() > 0:
                filtered_pred = pred["forecast"][mask.values]
                filtered_y = y_test[mask]
                filtered_conf = pred["confidence"][mask.values]
                avg_forecast = np.mean(filtered_pred)
                avg_actual = np.mean(filtered_y.values)
                st.metric("Avg Forecast", f"{avg_forecast:.0f} units")
                st.metric("Avg Actual", f"{avg_actual:.0f} units")
                st.metric("Avg Confidence", f"{np.mean(filtered_conf):.1%}")
            else:
                st.info("No test data for this filter")

        with col1:
            if mask.sum() > 0:
                df_plot = pd.DataFrame({
                    "price": X_test.loc[mask, "price"].values,
                    "actual": y_test[mask].values,
                    "pred": pred["forecast"][mask.values],
                    "lower": pred["lower_bound"][mask.values],
                    "upper": pred["upper_bound"][mask.values],
                })
                # Bin prices
                price_range = df_plot["price"].max() - df_plot["price"].min()
                bin_size = max(1, price_range / 20)
                df_plot["price_bin"] = (df_plot["price"] / bin_size).round() * bin_size
                grouped = df_plot.groupby("price_bin").mean().reset_index()

                fig = plot_grouped(
                    grouped["price_bin"].tolist(),
                    grouped["actual"].tolist(),
                    grouped["pred"].tolist(),
                    grouped["lower"].tolist(),
                    grouped["upper"].tolist(),
                    None,
                    f"Demand vs Price ({category}, {promo_filter})",
                    "Price ($)", "Units Sold",
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Comparison")
        st.caption(METRICS_LEGEND)
        st.dataframe(metrics_table(y_test.values, xgb_p, lgbm_p, avg_p, pred["forecast"]), hide_index=True)

    # ── Payment Tab ──
    with tab3:
        st.header("Payment Volume Forecasting")
        st.caption(
            "Predicts transaction volume by hour and day of week using payment method, amount patterns, and temporal features. "
            "Ensemble of XGBoost + LightGBM with meta-learner. "
            "The chart shows average actual vs predicted volume by hour for the selected day."
        )
        features, target = engineer_payment_features(payment_df)
        X_test, y_test, pred, xgb_p, lgbm_p, avg_p = prepare_domain(
            ensemble, "payment", features, target)

        col1, col2 = st.columns([2, 1])

        with col2:
            st.subheader("Parameters")
            dow = st.selectbox("Day of Week", list(range(7)),
                format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x], key="p_dow")
            hour = st.slider("Hour of Day", 0, 23, 10, key="p_hour")

            mask = X_test["day_of_week"] == dow
            if mask.sum() > 0:
                # Get samples at this hour
                hour_mask = mask & (X_test["hour_of_day"] == hour)
                if hour_mask.sum() > 0:
                    hp = pred["forecast"][hour_mask.values]
                    ha = y_test[hour_mask].values
                    hc = pred["confidence"][hour_mask.values]
                    st.metric("Avg Forecast", f"{np.mean(hp):.0f} txns")
                    st.metric("Avg Actual", f"{np.mean(ha):.0f} txns")
                    st.metric("Confidence", f"{np.mean(hc):.1%}")
                else:
                    st.info("No test data for this hour+day combo")

        with col1:
            day_name = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][dow]
            mask = X_test["day_of_week"] == dow
            if mask.sum() > 0:
                df_plot = pd.DataFrame({
                    "hour": X_test.loc[mask, "hour_of_day"].values,
                    "actual": y_test[mask].values,
                    "pred": pred["forecast"][mask.values],
                    "lower": pred["lower_bound"][mask.values],
                    "upper": pred["upper_bound"][mask.values],
                })
                grouped = df_plot.groupby("hour").mean().reset_index()

                fig = plot_grouped(
                    grouped["hour"].tolist(),
                    grouped["actual"].tolist(),
                    grouped["pred"].tolist(),
                    grouped["lower"].tolist(),
                    grouped["upper"].tolist(),
                    hour,
                    f"Transaction Volume by Hour ({day_name})",
                    "Hour of Day", "Volume",
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Comparison")
        st.caption(METRICS_LEGEND)
        st.dataframe(metrics_table(y_test.values, xgb_p, lgbm_p, avg_p, pred["forecast"]), hide_index=True)


if __name__ == "__main__":
    main()
