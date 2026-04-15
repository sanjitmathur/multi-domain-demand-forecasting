"""Stacking ensemble with Ridge meta-learner and uncertainty quantification."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.models.base_model import DomainModel
import joblib
from pathlib import Path


class EnsembleForecaster:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.domain_models: dict[str, DomainModel] = {}
        self.meta_learners: dict[str, Ridge] = {}
        self.oof_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit_domain(
        self,
        model: DomainModel,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> None:
        model.fit(X, y)
        self.domain_models[model.name] = model

        oof_preds = model.get_oof_predictions(X, y, n_splits=n_splits)
        n_oof = len(oof_preds)
        y_oof = y.values[-n_oof:]

        self.oof_data[model.name] = (oof_preds, y_oof)

    def fit_meta_learner(self) -> None:
        for name, (oof_preds, y_oof) in self.oof_data.items():
            meta = Ridge(alpha=self.alpha)
            meta.fit(oof_preds, y_oof)
            self.meta_learners[name] = meta

    def predict(self, domain: str, X: pd.DataFrame) -> dict:
        model = self.domain_models[domain]
        meta = self.meta_learners[domain]

        xgb_pred = model.predict_xgb(X)
        lgbm_pred = model.predict_lgbm(X)
        meta_features = np.column_stack([xgb_pred, lgbm_pred])

        forecast = meta.predict(meta_features)

        uncertainty = model.predict_with_uncertainty(X)
        lower = uncertainty["lower_10"]
        upper = uncertainty["upper_90"]

        # interval_score: 1 - relative-interval-width, clipped to [0, 1].
        # This is an interval-tightness heuristic, NOT a probability.
        # Use the empirical coverage on the test set (shown in the dashboard) to
        # evaluate calibration.
        width = upper - lower
        interval_score = np.clip(1.0 - width / (np.abs(forecast) + 1e-8), 0, 1)

        return {
            "forecast": forecast,
            "lower_bound": lower,
            "upper_bound": upper,
            "interval_score": interval_score,
            # Backwards-compat alias used by older dashboard code paths.
            "confidence": interval_score,
        }

    def predict_single_xgb(self, domain: str, X: pd.DataFrame) -> np.ndarray:
        return self.domain_models[domain].predict_xgb(X)

    def predict_single_lgbm(self, domain: str, X: pd.DataFrame) -> np.ndarray:
        return self.domain_models[domain].predict_lgbm(X)

    def predict_simple_avg(self, domain: str, X: pd.DataFrame) -> np.ndarray:
        return self.domain_models[domain].predict(X)

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for name, model in self.domain_models.items():
            model.save(str(path / name))
        for name, meta in self.meta_learners.items():
            joblib.dump(meta, path / f"{name}_meta.joblib")

    def load(self, directory: str, domain_names: list[str]) -> None:
        path = Path(directory)
        for name in domain_names:
            model = DomainModel(name=name)
            model.load(str(path / name))
            self.domain_models[name] = model
            self.meta_learners[name] = joblib.load(path / f"{name}_meta.joblib")
