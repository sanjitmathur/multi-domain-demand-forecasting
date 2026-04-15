"""Helpers for computing and loading naive/seasonal-naive/SMA baselines.

The training run already persists baselines to `results/base_model_evaluation.csv`
but the format is awkward (embedded dict-as-string per domain). We compute
baselines live on the held-out split so they stay in sync with the models the
dashboard is actually using.
"""

from __future__ import annotations

import ast
import numpy as np
import pandas as pd
from pathlib import Path

from src.evaluation.metrics import rmse, mae, r_squared, rmse_pct


def compute_naive(y_test: np.ndarray) -> np.ndarray:
    """Persistence baseline: prediction(t) = y(t-1)."""
    pred = np.empty_like(y_test, dtype=float)
    pred[0] = y_test[0]
    pred[1:] = y_test[:-1]
    return pred


def compute_seasonal_naive(y_test: np.ndarray, season: int = 7) -> np.ndarray:
    """Seasonal naive: prediction(t) = y(t-season). Falls back to naive for leading samples."""
    pred = np.empty_like(y_test, dtype=float)
    for i in range(len(y_test)):
        pred[i] = y_test[i - season] if i >= season else y_test[max(0, i - 1)]
    return pred


def compute_sma(y_test: np.ndarray, window: int = 7) -> np.ndarray:
    """Simple moving average baseline."""
    pred = np.empty_like(y_test, dtype=float)
    s = pd.Series(y_test).shift(1).rolling(window, min_periods=1).mean()
    pred[:] = s.fillna(y_test[0]).values
    return pred


def baseline_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "Model": name,
        "RMSE": round(rmse(y_true, y_pred), 3),
        "MAE": round(mae(y_true, y_pred), 3),
        "R²": round(r_squared(y_true, y_pred), 3),
        "RMSE %": round(rmse_pct(y_true, y_pred), 2),
    }


def all_baselines(y_test: np.ndarray, season: int = 7) -> list[dict]:
    """Return metric rows for Naive / Seasonal-Naive / SMA baselines."""
    return [
        baseline_row("Naive (t−1)", y_test, compute_naive(y_test)),
        baseline_row(f"Seasonal Naive ({season})", y_test, compute_seasonal_naive(y_test, season)),
        baseline_row(f"SMA ({season})", y_test, compute_sma(y_test, season)),
    ]


def load_persisted_baselines(results_csv: Path) -> dict | None:
    """Parse the train-time baseline CSV into {domain: {baseline_name: metrics_dict}}."""
    if not results_csv.exists():
        return None
    df = pd.read_csv(results_csv, index_col=0)
    out: dict[str, dict] = {}
    for domain in df.index:
        out[domain] = {}
        for col in df.columns:
            if col.startswith(f"{domain}_"):
                val = df.at[domain, col]
                if isinstance(val, str) and val.startswith("{"):
                    try:
                        out[domain][col.replace(f"{domain}_", "")] = ast.literal_eval(val)
                    except (ValueError, SyntaxError):
                        pass
    return out
