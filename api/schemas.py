"""Pydantic request/response models for the forecasting API."""

from pydantic import BaseModel


class AirlineRequest(BaseModel):
    days_until_departure: int = 30
    fare_class: str = "Economy"
    competitor_price: float = 300.0


class ECommerceRequest(BaseModel):
    product_category: str = "Electronics"
    price: float = 450.0
    promotion_active: bool = False


class PaymentRequest(BaseModel):
    hour_of_day: int = 10
    day_of_week: int = 2


class ForecastResponse(BaseModel):
    forecast: float
    lower_bound: float
    upper_bound: float
    confidence: float


class BatchRequest(BaseModel):
    airline: AirlineRequest | None = None
    ecommerce: ECommerceRequest | None = None
    payment: PaymentRequest | None = None
