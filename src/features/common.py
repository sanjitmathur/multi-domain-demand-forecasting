"""Shared feature engineering helpers used across all domains."""

import pandas as pd
import numpy as np


def add_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: list[int] = [7, 14, 30],
) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"{column}_rolling_mean_{w}"] = df[column].rolling(w).mean()
        df[f"{column}_rolling_std_{w}"] = df[column].rolling(w).std()
        df[f"{column}_rolling_max_{w}"] = df[column].rolling(w).max()
    return df


def add_lag_features(
    df: pd.DataFrame,
    column: str,
    lags: list[int] = [1, 7, 14],
) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df


def add_temporal_features(
    df: pd.DataFrame,
    date_column: str,
) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_column])
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["day_of_year"] = dt.dt.dayofyear
    return df


def add_trend_direction(
    df: pd.DataFrame,
    column: str,
    window: int = 7,
) -> pd.DataFrame:
    df = df.copy()
    rolling_mean = df[column].rolling(window).mean()
    diff = rolling_mean.diff()
    df[f"{column}_trend"] = np.sign(diff)
    return df
