"""Baseline forecast models for benchmarking."""

import numpy as np


def naive_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    return np.full(horizon, series[-1])


def seasonal_naive(
    series: np.ndarray, season_length: int, horizon: int
) -> np.ndarray:
    last_season = series[-season_length:]
    repeats = (horizon // season_length) + 1
    return np.tile(last_season, repeats)[:horizon]


def simple_moving_average(
    series: np.ndarray, window: int, horizon: int
) -> np.ndarray:
    avg = np.mean(series[-window:])
    return np.full(horizon, avg)
