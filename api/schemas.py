"""Pydantic request/response models for the forecasting API."""

from pydantic import BaseModel, Field


class AirlineRequest(BaseModel):
    days_until_departure: int = Field(30, ge=1, le=365, description="Days until flight departs")
    fare_class: str = Field("Economy", description="Economy | Business | First")
    competitor_price: float = Field(300.0, gt=0, description="Competitor fare in USD")


class ECommerceRequest(BaseModel):
    product_category: str = Field("Electronics", description="Beauty | Electronics | Fashion | Food & Grocery | Home & Kitchen")
    price: float = Field(450.0, gt=0, description="Unit price in USD")
    promotion_active: bool = Field(False, description="Whether a promotion is running")


class PaymentRequest(BaseModel):
    hour_of_day: int = Field(10, ge=0, le=23)
    day_of_week: int = Field(2, ge=0, le=6, description="0=Mon ... 6=Sun")


class ForecastResponse(BaseModel):
    forecast: float = Field(..., description="Point forecast (Ridge meta-learner output)")
    lower_bound: float = Field(..., description="10th percentile (quantile regression P10)")
    upper_bound: float = Field(..., description="90th percentile (quantile regression P90)")
    interval_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Relative interval tightness in [0, 1]. "
            "Computed as 1 - (upper - lower) / |forecast|. "
            "Higher = tighter interval relative to forecast magnitude. "
            "NOT a probability of correctness."
        ),
    )


class BatchRequest(BaseModel):
    airline: AirlineRequest | None = None
    ecommerce: ECommerceRequest | None = None
    payment: PaymentRequest | None = None
