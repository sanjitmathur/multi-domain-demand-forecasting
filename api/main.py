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
    price = bp * (1.0 + (1.0 - req.days_until_departure / 180) * 0.8)

    return pd.DataFrame([{
        "days_until_departure": req.days_until_departure,
        "fare_class_encoded": fare_map.get(req.fare_class, 0),
        "price": price,
        "competitor_price": req.competitor_price,
        "fuel_price_index": 100.0,
        "capacity": cap,
        "booking_velocity": 5.0,
        "load_factor": 0.5,
        "price_ratio": price / max(req.competitor_price, 1),
        "price_per_seat": price / cap,
        "bookings_rolling_mean_7": cap * 0.4,
        "bookings_rolling_std_7": cap * 0.1,
        "bookings_rolling_max_7": cap * 0.6,
        "bookings_rolling_mean_14": cap * 0.35,
        "bookings_rolling_mean_30": cap * 0.3,
        "bookings_trend": 1.0,
        "bookings_lag_1": cap * 0.4,
        "bookings_lag_7": cap * 0.35,
        "bookings_lag_14": cap * 0.3,
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
    bd = base_demand.get(req.product_category, 100)
    promo = 1 if req.promotion_active else 0

    return pd.DataFrame([{
        "category_encoded": cat_map.get(req.product_category, 0),
        "price": req.price,
        "competitor_price": req.price * 1.05,
        "price_ratio": 1.0 / 1.05,
        "discount_pct": 1 - 1.0 / 1.05,
        "promotions": promo,
        "inventory_level": bd * 2,
        "quantity_sold_rolling_mean_7": bd * 0.8,
        "quantity_sold_rolling_std_7": bd * 0.15,
        "quantity_sold_rolling_max_7": bd * 1.2,
        "quantity_sold_rolling_mean_30": bd * 0.75,
        "quantity_sold_lag_1": bd * 0.8,
        "quantity_sold_lag_7": bd * 0.75,
        "quantity_sold_lag_14": bd * 0.7,
        "quantity_sold_trend": 1.0,
        "price_x_promotion": req.price * promo,
        "inventory_x_trend": bd * 2 * 1.0,
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

    if 9 <= req.hour_of_day <= 11:
        base_vol = 500
    elif 17 <= req.hour_of_day <= 20:
        base_vol = 450
    elif 0 <= req.hour_of_day <= 5:
        base_vol = 50
    else:
        base_vol = 300

    return pd.DataFrame([{
        "hour_of_day": req.hour_of_day,
        "day_of_week": req.day_of_week,
        "month": 6,
        "txn_type_encoded": 1,
        "is_weekend": is_weekend,
        "is_holiday": 0,
        "is_month_end": 0,
        "volume_rolling_mean_6h": base_vol * 0.9,
        "volume_rolling_std_6h": base_vol * 0.2,
        "volume_rolling_max_6h": base_vol * 1.3,
        "volume_rolling_mean_24": base_vol * 0.7,
        "volume_rolling_std_24": base_vol * 0.3,
        "volume_rolling_max_24": base_vol * 1.5,
        "volume_rolling_mean_7d": base_vol * 0.65,
        "volume_rolling_std_7d": base_vol * 0.25,
        "volume_rolling_max_7d": base_vol * 1.6,
        "prev_hour_volume": base_vol * 0.85,
        "prev_day_same_hour": base_vol * 0.9,
        "prev_week_same_hour": base_vol * 0.88,
        "volume_trend": 1.0,
        "volume_volatility_7d": base_vol * 0.25,
        "avg_txn_value": 85.0,
        "fraud_flag": 0,
        "hour_sin": np.sin(2 * np.pi * req.hour_of_day / 24),
        "hour_cos": np.cos(2 * np.pi * req.hour_of_day / 24),
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
