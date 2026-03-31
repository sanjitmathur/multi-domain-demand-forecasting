"""Orchestrate training for all 3 domains."""

import pandas as pd
import numpy as np
from pathlib import Path
from src.features.airline_features import engineer_airline_features
from src.features.ecommerce_features import engineer_ecommerce_features
from src.features.payment_features import engineer_payment_features
from src.models.base_model import DomainModel
from src.models.ensemble import EnsembleForecaster
from src.models.baselines import naive_forecast, seasonal_naive, simple_moving_average
from src.evaluation.metrics import rmse, mae, mape, smape, r_squared, rmse_pct, percentile_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"


def load_and_split(
    features: pd.DataFrame, target: pd.Series, train_ratio: float = 0.8
) -> tuple:
    n = len(features)
    split = int(n * train_ratio)
    return (
        features.iloc[:split], features.iloc[split:],
        target.iloc[:split], target.iloc[split:],
    )


def evaluate_baselines(y_train: np.ndarray, y_test: np.ndarray, domain: str) -> dict:
    horizon = len(y_test)
    baselines = {
        "naive": naive_forecast(y_train, horizon),
        "seasonal_naive_7": seasonal_naive(y_train, season_length=7, horizon=horizon),
        "sma_7": simple_moving_average(y_train, window=7, horizon=horizon),
    }
    results = {}
    for name, preds in baselines.items():
        results[f"{domain}_{name}"] = {
            "rmse": rmse(y_test, preds),
            "mae": mae(y_test, preds),
            "rmse_pct": rmse_pct(y_test, preds),
        }
    return results


def train_domain(
    domain_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    ensemble: EnsembleForecaster,
) -> dict:
    X_train, X_test, y_train, y_test = load_and_split(features, target)

    print(f"\n{'='*60}")
    print(f"Training {domain_name} | train={len(X_train)}, test={len(X_test)}")
    print(f"{'='*60}")

    model = DomainModel(name=domain_name)
    ensemble.fit_domain(model, X_train, y_train, n_splits=5)

    # Fit meta-learner for this domain
    ensemble.fit_meta_learner()
    result = ensemble.predict(domain_name, X_test)

    ens_forecast = result["forecast"]
    xgb_pred = ensemble.predict_single_xgb(domain_name, X_test)
    lgbm_pred = ensemble.predict_single_lgbm(domain_name, X_test)
    avg_pred = ensemble.predict_simple_avg(domain_name, X_test)

    metrics = {
        "xgb_rmse": rmse(y_test.values, xgb_pred),
        "lgbm_rmse": rmse(y_test.values, lgbm_pred),
        "avg_rmse": rmse(y_test.values, avg_pred),
        "ensemble_meta_rmse": rmse(y_test.values, ens_forecast),
        "xgb_mae": mae(y_test.values, xgb_pred),
        "lgbm_mae": mae(y_test.values, lgbm_pred),
        "ensemble_meta_mae": mae(y_test.values, ens_forecast),
        "xgb_rmse_pct": rmse_pct(y_test.values, xgb_pred),
        "ensemble_meta_rmse_pct": rmse_pct(y_test.values, ens_forecast),
    }

    baseline_metrics = evaluate_baselines(y_train.values, y_test.values, domain_name)

    print(f"\n  XGBoost RMSE:        {metrics['xgb_rmse']:.4f} ({metrics['xgb_rmse_pct']:.2f}%)")
    print(f"  LightGBM RMSE:      {metrics['lgbm_rmse']:.4f}")
    print(f"  Simple Avg RMSE:    {metrics['avg_rmse']:.4f}")
    print(f"  Ensemble+Meta RMSE: {metrics['ensemble_meta_rmse']:.4f} ({metrics['ensemble_meta_rmse_pct']:.2f}%)")

    best_single = min(metrics["xgb_rmse"], metrics["lgbm_rmse"])
    improvement = (best_single - metrics["ensemble_meta_rmse"]) / best_single * 100
    print(f"  Improvement over best single: {improvement:.1f}%")

    return {**metrics, **baseline_metrics}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ensemble = EnsembleForecaster()
    all_metrics = {}

    print("\nLoading airline data...")
    airline_df = pd.read_csv(DATA_DIR / "airline_bookings.csv")
    airline_features, airline_target = engineer_airline_features(airline_df)
    metrics = train_domain("airline", airline_features, airline_target, ensemble)
    all_metrics["airline"] = metrics

    print("\nLoading e-commerce data...")
    ecom_df = pd.read_csv(DATA_DIR / "ecommerce_demand.csv")
    ecom_features, ecom_target = engineer_ecommerce_features(ecom_df)
    metrics = train_domain("ecommerce", ecom_features, ecom_target, ensemble)
    all_metrics["ecommerce"] = metrics

    print("\nLoading payment data...")
    payment_df = pd.read_csv(DATA_DIR / "payment_volume.csv")
    payment_features, payment_target = engineer_payment_features(payment_df)
    metrics = train_domain("payment", payment_features, payment_target, ensemble)
    all_metrics["payment"] = metrics

    ensemble.fit_meta_learner()

    ensemble.save(str(MODELS_DIR))
    print(f"\nModels saved to {MODELS_DIR}")

    results_df = pd.DataFrame(all_metrics).T
    results_df.to_csv(RESULTS_DIR / "base_model_evaluation.csv")
    print(f"Evaluation results saved to {RESULTS_DIR / 'base_model_evaluation.csv'}")

    return ensemble, all_metrics


if __name__ == "__main__":
    main()
