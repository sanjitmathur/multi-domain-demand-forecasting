"""Generate simulated hourly payment/transaction volume data."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_payment_data(
    n_days: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start_date = pd.Timestamp("2024-01-01")
    hours = pd.date_range(start_date, periods=n_days * 24, freq="h")

    transaction_types = ["card_payment", "bank_transfer", "mobile_wallet", "direct_debit"]
    type_weights = [0.45, 0.20, 0.25, 0.10]
    base_volume = {"card_payment": 500, "bank_transfer": 200, "mobile_wallet": 350, "direct_debit": 120}
    avg_value = {"card_payment": 85, "bank_transfer": 1200, "mobile_wallet": 45, "direct_debit": 250}

    rows = []
    for hour_ts in hours:
        hour = hour_ts.hour
        dow = hour_ts.dayofweek
        month = hour_ts.month
        day = hour_ts.day

        txn_type = rng.choice(transaction_types, p=type_weights)
        bv = base_volume[txn_type]
        av = avg_value[txn_type]

        if 9 <= hour <= 11:
            hour_factor = 1.5
        elif 17 <= hour <= 20:
            hour_factor = 1.35
        elif 12 <= hour <= 14:
            hour_factor = 1.1
        elif 0 <= hour <= 5:
            hour_factor = 0.15
        elif 6 <= hour <= 8:
            hour_factor = 0.6
        else:
            hour_factor = 0.9

        if dow < 5:
            dow_factor = 1.0
        elif dow == 5:
            dow_factor = 0.65
        else:
            dow_factor = 0.45

        month_end_factor = 1.0
        if day >= 25:
            month_end_factor = 1.4 if txn_type in ("bank_transfer", "direct_debit") else 1.15

        is_holiday = 0
        holiday_factor = 1.0
        if (month == 12 and day >= 20) or (month == 1 and day <= 5):
            is_holiday = 1
            holiday_factor = 1.3 if txn_type in ("card_payment", "mobile_wallet") else 0.7

        trend = 1.0 + (hour_ts - start_date).days / 365 * 0.25
        volume = bv * hour_factor * dow_factor * month_end_factor * holiday_factor * trend
        volume = max(0, int(rng.poisson(max(1, volume))))

        value = volume * av * (1 + rng.normal(0, 0.1))
        value = max(0, round(value, 2))

        fraud_base = 0.003
        if 0 <= hour <= 5:
            fraud_base *= 3
        if txn_type == "card_payment":
            fraud_base *= 1.5
        fraud_flag = int(rng.random() < fraud_base)

        rows.append({
            "timestamp": hour_ts,
            "hour_of_day": hour,
            "day_of_week": dow,
            "month": month,
            "transaction_type": txn_type,
            "volume": volume,
            "value": value,
            "is_weekend": int(dow >= 5),
            "is_holiday": is_holiday,
            "is_month_end": int(day >= 25),
            "fraud_flag": fraud_flag,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    output_dir = Path(__file__).resolve().parent.parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    df = generate_payment_data()
    df.to_csv(output_dir / "payment_volume.csv", index=False)
    print(f"Generated {len(df)} payment volume records -> data/payment_volume.csv")


if __name__ == "__main__":
    main()
