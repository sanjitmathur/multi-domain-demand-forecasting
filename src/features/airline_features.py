"""Airline-specific feature engineering."""

import pandas as pd
import numpy as np
from src.features.common import add_rolling_features, add_lag_features, add_trend_direction


def engineer_airline_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = df.sort_values(["departure_date", "days_until_departure"], ascending=[True, False])

    df["booking_velocity"] = df.groupby("departure_date")["bookings"].diff().fillna(0)
    df["load_factor"] = df["bookings"] / df["capacity"]

    df["price_ratio"] = df["price"] / df["competitor_price"]
    df["price_per_seat"] = df["price"] / df["capacity"]

    fare_map = {"Economy": 0, "Business": 1, "First": 2}
    df["fare_class_encoded"] = df["fare_class"].map(fare_map)

    df = add_rolling_features(df, "bookings", windows=[7, 14, 30])
    df = add_trend_direction(df, "bookings", window=7)
    df = add_lag_features(df, "bookings", lags=[1, 7, 14])

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.dropna()

    feature_cols = [
        "days_until_departure", "fare_class_encoded", "price", "competitor_price",
        "fuel_price_index", "capacity", "booking_velocity", "load_factor",
        "price_ratio", "price_per_seat",
        "bookings_rolling_mean_7", "bookings_rolling_std_7", "bookings_rolling_max_7",
        "bookings_rolling_mean_14", "bookings_rolling_mean_30",
        "bookings_trend", "bookings_lag_1", "bookings_lag_7", "bookings_lag_14",
        "day_of_week", "month", "is_weekend", "is_holiday",
        "dow_sin", "dow_cos", "month_sin", "month_cos",
    ]

    target = df["bookings"]
    features = df[feature_cols]

    return features, target
