"""Generate realistic e-commerce demand data modeled on UCI Online Retail patterns."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_ecommerce_data(
    n_days: int = 730,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    categories = ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Food & Grocery"]
    base_demand = {"Electronics": 120, "Fashion": 200, "Home & Kitchen": 80, "Beauty": 150, "Food & Grocery": 300}
    base_price = {"Electronics": 450, "Fashion": 85, "Home & Kitchen": 120, "Beauty": 35, "Food & Grocery": 25}

    start_date = pd.Timestamp("2023-01-01")
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    rows = []
    for date in dates:
        for cat in categories:
            dow = date.dayofweek
            month = date.month
            day = date.day
            day_of_year = date.dayofyear

            if cat in ("Electronics", "Home & Kitchen"):
                dow_factor = 1.1 if dow < 5 else 0.75
            else:
                dow_factor = 1.0 if dow < 5 else 1.15

            seasonal_factor = 1.0
            if month in (11, 12):
                seasonal_factor = 1.5
            elif month in (3, 4):
                seasonal_factor = 1.25 if cat in ("Food & Grocery", "Fashion") else 1.1
            elif month in (8, 9):
                seasonal_factor = 1.2 if cat in ("Electronics", "Fashion") else 1.0
            elif month in (1, 2):
                seasonal_factor = 0.7

            yoy_growth = 1.0 + (day_of_year + (date.year - 2023) * 365) / 365 * rng.uniform(0.05, 0.15)

            is_promotion = int(rng.random() < 0.15)
            if month == 11 and 24 <= day <= 30:
                is_promotion = 1
            promo_factor = 1.0 + is_promotion * rng.uniform(0.3, 0.8)

            price = base_price[cat] * (1 - is_promotion * rng.uniform(0.1, 0.3))
            price *= (1 + rng.normal(0, 0.03))
            competitor_price = base_price[cat] * (1 + rng.normal(0, 0.1))

            inventory = int(base_demand[cat] * rng.uniform(1.5, 3.0))
            if is_promotion:
                inventory = int(inventory * 1.5)

            demand = base_demand[cat] * dow_factor * seasonal_factor * yoy_growth * promo_factor
            price_ratio = price / base_price[cat]
            demand *= max(0.3, 1.5 - price_ratio)
            quantity_sold = max(0, int(rng.poisson(max(1, demand))))

            if month in (12, 1, 2):
                season = "Winter"
            elif month in (3, 4, 5):
                season = "Spring"
            elif month in (6, 7, 8):
                season = "Summer"
            else:
                season = "Fall"

            rows.append({
                "timestamp": date,
                "product_category": cat,
                "quantity_sold": quantity_sold,
                "price": round(price, 2),
                "competitor_price": round(competitor_price, 2),
                "promotions": is_promotion,
                "inventory_level": inventory,
                "day_of_week": dow,
                "month": month,
                "is_weekend": int(dow >= 5),
                "is_holiday": int((month == 11 and 24 <= day <= 30) or (month == 12 and day >= 20)),
                "season": season,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["timestamp", "product_category"]).reset_index(drop=True)
    return df


def main():
    output_dir = Path(__file__).resolve().parent.parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    df = generate_ecommerce_data()
    df.to_csv(output_dir / "ecommerce_demand.csv", index=False)
    print(f"Generated {len(df)} e-commerce demand records -> data/ecommerce_demand.csv")


if __name__ == "__main__":
    main()
