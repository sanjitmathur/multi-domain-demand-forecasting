"""Generate realistic airline booking data with seasonal patterns."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_airline_data(
    n_flights: int = 300,
    booking_window_days: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start_date = pd.Timestamp("2024-01-01")
    departure_dates = pd.date_range(start_date, periods=n_flights, freq="D")
    departure_dates = rng.choice(departure_dates, size=n_flights, replace=True)
    departure_dates = pd.DatetimeIndex(np.sort(departure_dates))

    fare_classes = ["Economy", "Business", "First"]
    fare_class_weights = [0.7, 0.2, 0.1]
    capacity_by_class = {"Economy": 180, "Business": 40, "First": 12}
    base_price = {"Economy": 250, "Business": 800, "First": 2200}

    rows = []
    for dep_date in departure_dates:
        fare_class = rng.choice(fare_classes, p=fare_class_weights)
        cap = capacity_by_class[fare_class]
        bp = base_price[fare_class]

        snapshot_days = sorted(
            rng.choice(range(1, booking_window_days + 1), size=12, replace=False),
            reverse=True,
        )

        cumulative_bookings = 0
        for days_until in snapshot_days:
            dow = dep_date.dayofweek
            month = dep_date.month

            seasonal_factor = 1.0
            if month in (6, 7, 8):
                seasonal_factor = 1.35
            elif month == 12:
                seasonal_factor = 1.45
            elif month in (1, 2):
                seasonal_factor = 0.75

            dow_factor = 1.0
            if dow in (4, 6):
                dow_factor = 1.2
            elif dow in (1, 2):
                dow_factor = 0.85

            curve_factor = np.exp(-0.02 * days_until) * 0.6 + 0.4

            price_multiplier = 1.0 + (1.0 - days_until / booking_window_days) * 0.8
            price = bp * price_multiplier * (1 + rng.normal(0, 0.05))
            competitor_price = price * (1 + rng.normal(0, 0.15))
            fuel_index = 100 + 20 * np.sin(2 * np.pi * month / 12) + rng.normal(0, 5)

            demand_rate = cap * 0.08 * seasonal_factor * dow_factor * curve_factor
            price_elasticity = max(0.5, 1.0 - (price - bp) / (bp * 2))
            demand_rate *= price_elasticity
            new_bookings = max(0, int(rng.poisson(max(1, demand_rate))))
            cumulative_bookings = min(cap, cumulative_bookings + new_bookings)

            is_holiday = 1 if month == 12 and dep_date.day >= 20 else 0
            if month == 1 and dep_date.day <= 5:
                is_holiday = 1

            rows.append({
                "date": dep_date - pd.Timedelta(days=int(days_until)),
                "departure_date": dep_date,
                "days_until_departure": int(days_until),
                "fare_class": fare_class,
                "price": round(price, 2),
                "competitor_price": round(competitor_price, 2),
                "fuel_price_index": round(fuel_index, 2),
                "capacity": cap,
                "bookings": cumulative_bookings,
                "day_of_week": dow,
                "month": month,
                "is_weekend": int(dow >= 5),
                "is_holiday": is_holiday,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["departure_date", "days_until_departure"], ascending=[True, False])
    df = df.reset_index(drop=True)
    return df


def main():
    output_dir = Path(__file__).resolve().parent.parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    df = generate_airline_data()
    df.to_csv(output_dir / "airline_bookings.csv", index=False)
    print(f"Generated {len(df)} airline booking records -> data/airline_bookings.csv")


if __name__ == "__main__":
    main()
