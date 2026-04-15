"""API surface tests — schema shape, validation, and response contract.

These use FastAPI's TestClient so they run without uvicorn. When trained
model artifacts are absent, the `/forecast/*` endpoints should degrade
gracefully with a 503 — we test that explicitly so CI is green on clean
clones.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import AirlineRequest, ECommerceRequest, PaymentRequest


@pytest.fixture(scope="module")
def client() -> TestClient:
    # TestClient triggers startup hooks, so ensemble loading is attempted.
    with TestClient(app) as c:
        yield c


def test_root_returns_status(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "running"


def test_openapi_schema_lists_all_endpoints(client: TestClient):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"].keys()
    for route in ("/forecast/airline", "/forecast/ecommerce",
                  "/forecast/payment", "/forecast/batch"):
        assert route in paths


def test_forecast_response_schema_fields():
    from api.schemas import ForecastResponse

    fields = ForecastResponse.model_fields
    for field in ("forecast", "lower_bound", "upper_bound", "interval_score"):
        assert field in fields


def test_airline_request_validates_ranges():
    # Out-of-range days_until_departure must fail validation.
    with pytest.raises(Exception):
        AirlineRequest(days_until_departure=-1, fare_class="Economy", competitor_price=300)
    with pytest.raises(Exception):
        AirlineRequest(days_until_departure=30, fare_class="Economy", competitor_price=0)


def test_payment_request_validates_ranges():
    with pytest.raises(Exception):
        PaymentRequest(hour_of_day=24, day_of_week=0)
    with pytest.raises(Exception):
        PaymentRequest(hour_of_day=10, day_of_week=7)


def test_forecast_endpoint_returns_valid_payload_or_503(client: TestClient):
    # If models loaded, payload is valid; if not, a 503 is the documented path.
    payload = {"days_until_departure": 30, "fare_class": "Economy", "competitor_price": 300}
    r = client.post("/forecast/airline", json=payload)
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert {"forecast", "lower_bound", "upper_bound", "interval_score"} <= set(body)
        assert body["lower_bound"] <= body["forecast"] <= body["upper_bound"] or \
               abs(body["forecast"] - body["lower_bound"]) < 1e-6  # tolerate boundary
        assert 0.0 <= body["interval_score"] <= 1.0
