from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    status: Literal["ok"] = "ok"
    message: str = "InsightRetail API is running"


class SummaryResponse(BaseModel):
    total_customers: int
    total_products: int
    total_orders: int
    total_revenue: float
    latest_forecast_date: date | None
    latest_forecast_value: float | None


class DailySaleItem(BaseModel):
    sale_date: date
    revenue: float
    orders_count: int


class ProductTopItem(BaseModel):
    product_id: str
    description: str | None = None
    total_revenue: float
    total_quantity: int


class CustomerTopItem(BaseModel):
    customer_id: int
    country: str | None = None
    total_revenue: float
    total_orders: int


class SegmentItem(BaseModel):
    customer_id: int
    segment: str | None = None
    recency: int | None = None
    frequency: int | None = None
    monetary: float | None = None
    r_score: int | None = None
    f_score: int | None = None
    m_score: int | None = None


class ForecastItem(BaseModel):
    forecast_date: date
    forecast_value: float
    model_name: str


class PredictRequest(BaseModel):
    prev_day_revenue: float = Field(..., ge=0)
    revenue_7_days_ago: float = Field(..., ge=0)
    revenue_14_days_ago: float = Field(..., ge=0)
    rolling_7d_avg: float = Field(..., ge=0)
    rolling_30d_avg: float = Field(..., ge=0)
    rolling_7d_std: float = Field(..., ge=0)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    weekend: int = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    predicted_revenue: float
    model_name: str = "best_forecast_model"
    message: str = "Prediction generated successfully"
