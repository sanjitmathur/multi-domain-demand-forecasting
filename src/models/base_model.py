"""Domain model wrapper: XGBoost + LightGBM with OOF and quantile support."""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
import joblib
from pathlib import Path


class DomainModel:
    def __init__(
        self,
        name: str,
        xgb_params: dict | None = None,
        lgbm_params: dict | None = None,
    ):
        self.name = name
        self.xgb_params = xgb_params or {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "random_state": 42,
            "verbosity": 0,
        }
        self.lgbm_params = lgbm_params or {
            "num_leaves": 31,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "random_state": 42,
            "verbose": -1,
        }
        self.xgb_model = None
        self.lgbm_model = None
        self.xgb_q10 = None
        self.xgb_q90 = None
        self.lgbm_q10 = None
        self.lgbm_q90 = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.xgb_model = XGBRegressor(**self.xgb_params)
        self.xgb_model.fit(X, y)

        self.lgbm_model = LGBMRegressor(**self.lgbm_params)
        self.lgbm_model.fit(X, y)

        xgb_q_params = {**self.xgb_params, "objective": "reg:quantileerror"}
        self.xgb_q10 = XGBRegressor(**xgb_q_params, quantile_alpha=0.1)
        self.xgb_q10.fit(X, y)
        self.xgb_q90 = XGBRegressor(**xgb_q_params, quantile_alpha=0.9)
        self.xgb_q90.fit(X, y)

        lgbm_q_params = {**self.lgbm_params, "objective": "quantile"}
        self.lgbm_q10 = LGBMRegressor(**lgbm_q_params, alpha=0.1)
        self.lgbm_q10.fit(X, y)
        self.lgbm_q90 = LGBMRegressor(**lgbm_q_params, alpha=0.9)
        self.lgbm_q90.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        xgb_pred = self.xgb_model.predict(X)
        lgbm_pred = self.lgbm_model.predict(X)
        return (xgb_pred + lgbm_pred) / 2

    def predict_xgb(self, X: pd.DataFrame) -> np.ndarray:
        return self.xgb_model.predict(X)

    def predict_lgbm(self, X: pd.DataFrame) -> np.ndarray:
        return self.lgbm_model.predict(X)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> dict:
        forecast = self.predict(X)
        lower = (self.xgb_q10.predict(X) + self.lgbm_q10.predict(X)) / 2
        upper = (self.xgb_q90.predict(X) + self.lgbm_q90.predict(X)) / 2
        return {
            "forecast": forecast,
            "lower_10": lower,
            "upper_90": upper,
        }

    def get_oof_predictions(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
    ) -> np.ndarray:
        oof_xgb = np.full(len(X), np.nan)
        oof_lgbm = np.full(len(X), np.nan)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y.iloc[train_idx]

            xgb = XGBRegressor(**self.xgb_params)
            xgb.fit(X_tr, y_tr)
            oof_xgb[val_idx] = xgb.predict(X_val)

            lgbm = LGBMRegressor(**self.lgbm_params)
            lgbm.fit(X_tr, y_tr)
            oof_lgbm[val_idx] = lgbm.predict(X_val)

        mask = ~np.isnan(oof_xgb)
        return np.column_stack([oof_xgb[mask], oof_lgbm[mask]])

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.xgb_model, path / f"{self.name}_xgb.joblib")
        joblib.dump(self.lgbm_model, path / f"{self.name}_lgbm.joblib")
        joblib.dump(self.xgb_q10, path / f"{self.name}_xgb_q10.joblib")
        joblib.dump(self.xgb_q90, path / f"{self.name}_xgb_q90.joblib")
        joblib.dump(self.lgbm_q10, path / f"{self.name}_lgbm_q10.joblib")
        joblib.dump(self.lgbm_q90, path / f"{self.name}_lgbm_q90.joblib")

    def load(self, directory: str) -> None:
        path = Path(directory)
        self.xgb_model = joblib.load(path / f"{self.name}_xgb.joblib")
        self.lgbm_model = joblib.load(path / f"{self.name}_lgbm.joblib")
        self.xgb_q10 = joblib.load(path / f"{self.name}_xgb_q10.joblib")
        self.xgb_q90 = joblib.load(path / f"{self.name}_xgb_q90.joblib")
        self.lgbm_q10 = joblib.load(path / f"{self.name}_lgbm_q10.joblib")
        self.lgbm_q90 = joblib.load(path / f"{self.name}_lgbm_q90.joblib")
