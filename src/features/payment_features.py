"""Payment/transaction volume feature engineering."""

import pandas as pd
import numpy as np
from src.features.common import add_rolling_features, add_lag_features, add_trend_direction


def engineer_payment_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = df.sort_values("timestamp")

    type_map = {t: i for i, t in enumerate(sorted(df["transaction_type"].unique()))}
    df["txn_type_encoded"] = df["transaction_type"].map(type_map)

    df = add_rolling_features(df, "volume", windows=[6, 24, 168])

    df = df.rename(columns={
        "volume_rolling_mean_6": "volume_rolling_mean_6h",
        "volume_rolling_std_6": "volume_rolling_std_6h",
        "volume_rolling_max_6": "volume_rolling_max_6h",
        "volume_rolling_mean_168": "volume_rolling_mean_7d",
        "volume_rolling_std_168": "volume_rolling_std_7d",
        "volume_rolling_max_168": "volume_rolling_max_7d",
    })

    df["prev_hour_volume"] = df["volume"].shift(1)
    df["prev_day_same_hour"] = df["volume"].shift(24)
    df["prev_week_same_hour"] = df["volume"].shift(168)

    df = add_trend_direction(df, "volume", window=24)
    df["volume_volatility_7d"] = df["volume"].rolling(168).std()

    df["avg_txn_value"] = df["value"] / df["volume"].replace(0, np.nan)
    df["avg_txn_value"] = df["avg_txn_value"].fillna(0)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df = df.dropna()

    feature_cols = [
        "hour_of_day", "day_of_week", "month", "txn_type_encoded",
        "is_weekend", "is_holiday", "is_month_end",
        "volume_rolling_mean_6h", "volume_rolling_std_6h", "volume_rolling_max_6h",
        "volume_rolling_mean_24", "volume_rolling_std_24", "volume_rolling_max_24",
        "volume_rolling_mean_7d", "volume_rolling_std_7d", "volume_rolling_max_7d",
        "prev_hour_volume", "prev_day_same_hour", "prev_week_same_hour",
        "volume_trend", "volume_volatility_7d",
        "avg_txn_value", "fraud_flag",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]

    target = df["volume"]
    features = df[feature_cols]

    return features, target
