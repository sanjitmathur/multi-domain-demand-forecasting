"""Tests for the baseline predictors used in the dashboard."""

from __future__ import annotations

import numpy as np
import pytest

from dashboard._baselines import compute_naive, compute_seasonal_naive, compute_sma


def test_naive_shifts_by_one():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p = compute_naive(y)
    # First sample has no previous value → falls back to self.
    assert p[0] == y[0]
    assert np.allclose(p[1:], y[:-1])


def test_seasonal_naive_shifts_by_season():
    y = np.arange(14.0)
    p = compute_seasonal_naive(y, season=7)
    # Beyond the warm-up, the forecast is y(t - 7).
    assert np.allclose(p[7:], y[:7])


def test_sma_equals_shifted_mean():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    p = compute_sma(y, window=3)
    # SMA(t) = mean(y[t-3:t]). At t=4 that's mean(y[1:4]) = mean([2, 3, 4]) = 3.
    assert p[4] == pytest.approx(3.0)
    # Output length matches input.
    assert len(p) == len(y)


def test_baselines_output_is_finite():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    for fn in (compute_naive, compute_seasonal_naive, compute_sma):
        p = fn(y) if fn is compute_naive else fn(y, 3)
        assert np.all(np.isfinite(p))
        assert p.shape == y.shape
