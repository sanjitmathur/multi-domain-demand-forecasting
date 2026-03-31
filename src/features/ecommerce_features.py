"""E-commerce-specific feature engineering."""

import pandas as pd
import numpy as np
from src.features.common import add_rolling_features, add_lag_features, add_trend_direction


def engineer_ecommerce_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = df.sort_values(["product_category", "timestamp"])

    cat_map = {c: i for i, c in enumerate(sorted(df["product_category"].unique()))}
    df["category_encoded"] = df["product_category"].map(cat_map)

    season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
    df["season_encoded"] = df["season"].map(season_map)

    df["price_ratio"] = df["price"] / df["competitor_price"].replace(0, np.nan).fillna(df["price"])
    df["discount_pct"] = 1 - df["price"] / df["competitor_price"].replace(0, np.nan).fillna(df["price"])

    dfs = []
    for cat in df["product_category"].unique():
        cat_df = df[df["product_category"] == cat].copy()
        cat_df = add_rolling_features(cat_df, "quantity_sold", windows=[7, 30])
        cat_df = add_lag_features(cat_df, "quantity_sold", lags=[1, 7, 14])
        cat_df = add_trend_direction(cat_df, "quantity_sold", window=7)
        dfs.append(cat_df)
    df = pd.concat(dfs, ignore_index=True)

    df["price_x_promotion"] = df["price"] * df["promotions"]
    df["inventory_x_trend"] = df["inventory_level"] * df["quantity_sold_trend"].fillna(0)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.dropna()

    feature_cols = [
        "category_encoded", "price", "competitor_price", "price_ratio", "discount_pct",
        "promotions", "inventory_level",
        "quantity_sold_rolling_mean_7", "quantity_sold_rolling_std_7", "quantity_sold_rolling_max_7",
        "quantity_sold_rolling_mean_30",
        "quantity_sold_lag_1", "quantity_sold_lag_7", "quantity_sold_lag_14",
        "quantity_sold_trend",
        "price_x_promotion", "inventory_x_trend",
        "day_of_week", "month", "is_weekend", "is_holiday", "season_encoded",
        "dow_sin", "dow_cos", "month_sin", "month_cos",
    ]

    target = df["quantity_sold"]
    features = df[feature_cols]

    return features, target
