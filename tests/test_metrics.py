"""Unit tests for evaluation metrics — these are load-bearing; every other
test and dashboard surface depends on them being correct."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.metrics import (
    mae,
    mape,
    percentile_error,
    r_squared,
    rmse,
    rmse_pct,
    smape,
)


def test_rmse_zero_when_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_matches_hand_calc():
    # Residuals: [1, -1, 1, -1] → RMSE = sqrt(mean(1)) = 1.0
    y = np.array([2.0, 2.0, 2.0, 2.0])
    p = np.array([1.0, 3.0, 1.0, 3.0])
    assert rmse(y, p) == pytest.approx(1.0)


def test_mae_matches_hand_calc():
    y = np.array([5.0, 5.0, 5.0, 5.0])
    p = np.array([4.0, 6.0, 7.0, 3.0])
    # |1| + |-1| + |-2| + |2| == 6 → mean == 1.5
    assert mae(y, p) == pytest.approx(1.5)


def test_r_squared_perfect_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == pytest.approx(1.0)


def test_r_squared_constant_target_returns_one():
    # Branch-coverage guard: ss_tot == 0 should not divide by zero.
    y = np.array([5.0, 5.0, 5.0])
    p = np.array([5.0, 5.0, 5.0])
    assert r_squared(y, p) == pytest.approx(1.0)


def test_r_squared_can_be_negative():
    # A model worse than the mean should give R² < 0.
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert r_squared(y, p) < 0


def test_mape_handles_zero_targets():
    y = np.array([0.0, 10.0, 20.0])
    p = np.array([5.0, 11.0, 22.0])
    # Zero targets dropped → mean(|1/10|, |2/20|) = mean(0.1, 0.1) = 0.1
    assert mape(y, p) == pytest.approx(0.1)


def test_smape_is_symmetric_and_bounded():
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 18.0, 33.0])
    forward = smape(y, p)
    backward = smape(p, y)
    assert forward == pytest.approx(backward)
    assert 0.0 <= forward <= 2.0


def test_rmse_pct_is_scale_free():
    # Scaling y and p by the same factor shouldn't change RMSE %.
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([11.0, 19.0, 33.0])
    base = rmse_pct(y, p)
    scaled = rmse_pct(y * 100, p * 100)
    assert base == pytest.approx(scaled)


def test_rmse_pct_zero_mean_is_safe():
    y = np.zeros(3)
    p = np.array([0.1, -0.1, 0.0])
    assert rmse_pct(y, p) == 0.0


def test_percentile_error_monotone():
    errs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.zeros_like(errs)
    assert percentile_error(y, errs, p=50) <= percentile_error(y, errs, p=95)


def test_metrics_return_finite_floats():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 2.2, 2.9])
    for fn in (rmse, mae, r_squared, rmse_pct):
        result = fn(y, p)
        assert isinstance(result, float)
        assert math.isfinite(result)
