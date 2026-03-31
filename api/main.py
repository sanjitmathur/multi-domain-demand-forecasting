"""FastAPI app for demand forecasting predictions."""

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from api.schemas import (
    AirlineRequest, ECommerceRequest, PaymentRequest,
    ForecastResponse, BatchRequest,
)
from src.models.ensemble import EnsembleForecaster

app = FastAPI(title="Multi-Domain Demand Forecasting API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

ensemble: EnsembleForecaster | None = None


@app.on_event("startup")
def load_models():
    global ensemble
    ensemble = EnsembleForecaster()
    try:
        ensemble.load(str(MODELS_DIR), ["airline", "ecommerce", "payment"])
        print("Models loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load models: {e}")
        ensemble = None


def _build_airline_features(req: AirlineRequest) -> pd.DataFrame:
    fare_map = {"Economy": 0, "Business": 1, "First": 2}
    capacity_map = {"Economy": 180, "Business": 40, "First": 12}
    base_price_map = {"Economy": 250, "Business": 800, "First": 2200}

    cap = capacity_map.get(req.fare_class, 180)
    bp = base_price_map.get(req.fare_class, 250)
    d = req.days_until_departure

    # Price increases as departure approaches (booking curve)
    price = bp * (1.0 + (1.0 - d / 180) * 0.8)
    # Booking curve: fraction of capacity sold so far
    curve = np.exp(-0.02 * d) * 0.6 + 0.4  # 0.4 far out -> 1.0 at departure
    load_factor = min(0.95, curve * 0.85)
    cumulative_bookings = cap * load_factor
    # Velocity accelerates closer to departure
    velocity = max(1, cap * 0.08 * curve * 1.2)

    return pd.DataFrame([{
        "days_until_departure": d,
        "fare_class_encoded": fare_map.get(req.fare_class, 0),
        "price": price,
        "competitor_price": req.competitor_price,
        "fuel_price_index": 100.0,
        "capacity": cap,
        "booking_velocity": velocity,
        "load_factor": load_factor,
        "price_ratio": price / max(req.competitor_price, 1),
        "price_per_seat": price / cap,
        "bookings_rolling_mean_7": cumulative_bookings * 0.95,
        "bookings_rolling_std_7": cap * 0.05 + velocity * 0.5,
        "bookings_rolling_max_7": cumulative_bookings * 1.05,
        "bookings_rolling_mean_14": cumulative_bookings * 0.88,
        "bookings_rolling_mean_30": cumulative_bookings * 0.75,
        "bookings_trend": 1.0 if d < 90 else 0.0,
        "bookings_lag_1": cumulative_bookings * 0.97,
        "bookings_lag_7": cumulative_bookings * 0.85,
        "bookings_lag_14": cumulative_bookings * 0.72,
        "day_of_week": 3,
        "month": 6,
        "is_weekend": 0,
        "is_holiday": 0,
        "dow_sin": np.sin(2 * np.pi * 3 / 7),
        "dow_cos": np.cos(2 * np.pi * 3 / 7),
        "month_sin": np.sin(2 * np.pi * 6 / 12),
        "month_cos": np.cos(2 * np.pi * 6 / 12),
    }])


def _build_ecommerce_features(req: ECommerceRequest) -> pd.DataFrame:
    cat_map = {"Beauty": 0, "Electronics": 1, "Fashion": 2, "Food & Grocery": 3, "Home & Kitchen": 4}
    base_demand = {"Electronics": 120, "Fashion": 200, "Home & Kitchen": 80, "Beauty": 150, "Food & Grocery": 300}
    base_price = {"Electronics": 450, "Fashion": 85, "Home & Kitchen": 120, "Beauty": 35, "Food & Grocery": 25}
    bd = base_demand.get(req.product_category, 100)
    bp = base_price.get(req.product_category, 100)
    promo = 1 if req.promotion_active else 0

    # Price elasticity: demand scales inversely with price relative to base
    price_ratio = req.price / max(bp, 1)
    elasticity = max(0.3, 1.5 - price_ratio)
    # Estimated demand given price and promo
    est_demand = bd * elasticity * (1 + promo * 0.5)
    # Competitor price assumed at base price level
    comp_price = bp * 1.05

    return pd.DataFrame([{
        "category_encoded": cat_map.get(req.product_category, 0),
        "price": req.price,
        "competitor_price": comp_price,
        "price_ratio": req.price / comp_price,
        "discount_pct": 1 - req.price / comp_price,
        "promotions": promo,
        "inventory_level": int(est_demand * 2.5),
        "quantity_sold_rolling_mean_7": est_demand * 0.9,
        "quantity_sold_rolling_std_7": est_demand * 0.15,
        "quantity_sold_rolling_max_7": est_demand * 1.3,
        "quantity_sold_rolling_mean_30": est_demand * 0.85,
        "quantity_sold_lag_1": est_demand * 0.95,
        "quantity_sold_lag_7": est_demand * 0.9,
        "quantity_sold_lag_14": est_demand * 0.85,
        "quantity_sold_trend": 1.0 if price_ratio < 1.0 else (-1.0 if price_ratio > 1.2 else 0.0),
        "price_x_promotion": req.price * promo,
        "inventory_x_trend": int(est_demand * 2.5) * (1.0 if price_ratio < 1.0 else 0.0),
        "day_of_week": 3,
        "month": 6,
        "is_weekend": 0,
        "is_holiday": 0,
        "season_encoded": 2,
        "dow_sin": np.sin(2 * np.pi * 3 / 7),
        "dow_cos": np.cos(2 * np.pi * 3 / 7),
        "month_sin": np.sin(2 * np.pi * 6 / 12),
        "month_cos": np.cos(2 * np.pi * 6 / 12),
    }])


def _build_payment_features(req: PaymentRequest) -> pd.DataFrame:
    is_weekend = 1 if req.day_of_week >= 5 else 0
    h = req.hour_of_day

    # Realistic intra-day volume profile (bimodal: morning + evening peaks)
    hour_profile = {
        0: 0.08, 1: 0.05, 2: 0.04, 3: 0.03, 4: 0.04, 5: 0.08,
        6: 0.25, 7: 0.45, 8: 0.70, 9: 1.0, 10: 1.05, 11: 0.95,
        12: 0.80, 13: 0.75, 14: 0.78, 15: 0.82, 16: 0.88,
        17: 1.0, 18: 0.95, 19: 0.85, 20: 0.70, 21: 0.50, 22: 0.30, 23: 0.15,
    }
    hour_factor = hour_profile.get(h, 0.5)
    dow_factor = 1.0 if req.day_of_week < 5 else (0.65 if req.day_of_week == 5 else 0.45)

    base_vol = 400 * hour_factor * dow_factor

    # Previous hour follows the profile
    prev_h = (h - 1) % 24
    prev_vol = 400 * hour_profile.get(prev_h, 0.5) * dow_factor

    # Daily average (weighted across hours)
    daily_avg = 400 * 0.52 * dow_factor  # average of the hour profile ~0.52

    return pd.DataFrame([{
        "hour_of_day": h,
        "day_of_week": req.day_of_week,
        "month": 6,
        "txn_type_encoded": 1,
        "is_weekend": is_weekend,
        "is_holiday": 0,
        "is_month_end": 0,
        "volume_rolling_mean_6h": base_vol * 0.9,
        "volume_rolling_std_6h": base_vol * 0.25,
        "volume_rolling_max_6h": base_vol * 1.3,
        "volume_rolling_mean_24": daily_avg,
        "volume_rolling_std_24": daily_avg * 0.6,
        "volume_rolling_max_24": 400 * 1.05 * dow_factor,
        "volume_rolling_mean_7d": daily_avg * 0.95,
        "volume_rolling_std_7d": daily_avg * 0.5,
        "volume_rolling_max_7d": 400 * 1.05,
        "prev_hour_volume": prev_vol,
        "prev_day_same_hour": base_vol * 0.95,
        "prev_week_same_hour": base_vol * 0.92,
        "volume_trend": 1.0,
        "volume_volatility_7d": daily_avg * 0.5,
        "avg_txn_value": 85.0,
        "fraud_flag": 0,
        "hour_sin": np.sin(2 * np.pi * h / 24),
        "hour_cos": np.cos(2 * np.pi * h / 24),
        "dow_sin": np.sin(2 * np.pi * req.day_of_week / 7),
        "dow_cos": np.cos(2 * np.pi * req.day_of_week / 7),
    }])


def _make_response(result: dict, idx: int = 0) -> ForecastResponse:
    return ForecastResponse(
        forecast=round(float(result["forecast"][idx]), 2),
        lower_bound=round(float(result["lower_bound"][idx]), 2),
        upper_bound=round(float(result["upper_bound"][idx]), 2),
        confidence=round(float(result["confidence"][idx]), 4),
    )


@app.get("/")
def root():
    return {"message": "Multi-Domain Demand Forecasting API", "status": "running"}


@app.post("/forecast/airline", response_model=ForecastResponse)
def forecast_airline(req: AirlineRequest):
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    features = _build_airline_features(req)
    result = ensemble.predict("airline", features)
    return _make_response(result)


@app.post("/forecast/ecommerce", response_model=ForecastResponse)
def forecast_ecommerce(req: ECommerceRequest):
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    features = _build_ecommerce_features(req)
    result = ensemble.predict("ecommerce", features)
    return _make_response(result)


@app.post("/forecast/payment", response_model=ForecastResponse)
def forecast_payment(req: PaymentRequest):
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    features = _build_payment_features(req)
    result = ensemble.predict("payment", features)
    return _make_response(result)


@app.post("/forecast/batch")
def forecast_batch(req: BatchRequest):
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    results = {}
    if req.airline:
        features = _build_airline_features(req.airline)
        result = ensemble.predict("airline", features)
        results["airline"] = _make_response(result).model_dump()
    if req.ecommerce:
        features = _build_ecommerce_features(req.ecommerce)
        result = ensemble.predict("ecommerce", features)
        results["ecommerce"] = _make_response(result).model_dump()
    if req.payment:
        features = _build_payment_features(req.payment)
        result = ensemble.predict("payment", features)
        results["payment"] = _make_response(result).model_dump()
    return results
