from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from api.schemas import ForecastItem, CustomerTopItem, DailySaleItem, ProductTopItem, SegmentItem, SummaryResponse


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_forecast_model.joblib"


class DatabaseService:
    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url, future=True)
        self.Session = sessionmaker(bind=self.engine, future=True)

    def fetch_summary(self) -> SummaryResponse:
        query = text(
            """
            SELECT
                (SELECT COUNT(*) FROM customers) AS total_customers,
                (SELECT COUNT(*) FROM products) AS total_products,
                (SELECT COUNT(*) FROM orders) AS total_orders,
                (SELECT COALESCE(SUM(total_amount), 0) FROM orders) AS total_revenue,
                (SELECT forecast_date FROM forecasts ORDER BY forecast_date DESC LIMIT 1) AS latest_forecast_date,
                (SELECT forecast_value FROM forecasts ORDER BY forecast_date DESC LIMIT 1) AS latest_forecast_value
            """
        )
        with self.engine.connect() as conn:
            result = conn.execute(query).mappings().one()

        return SummaryResponse(
            total_customers=int(result["total_customers"]),
            total_products=int(result["total_products"]),
            total_orders=int(result["total_orders"]),
            total_revenue=float(result["total_revenue"]),
            latest_forecast_date=result["latest_forecast_date"],
            latest_forecast_value=(float(result["latest_forecast_value"]) if result["latest_forecast_value"] is not None else None),
        )

    def fetch_daily_sales(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        country: str | None = None,
        limit: int = 30,
    ) -> list[DailySaleItem]:
        filters = ["TRUE"]
        params: dict[str, Any] = {}

        if start_date is not None:
            filters.append("sale_date >= :start_date")
            params["start_date"] = start_date
        if end_date is not None:
            filters.append("sale_date <= :end_date")
            params["end_date"] = end_date

        query = text(
            f"SELECT sale_date, revenue, orders_count FROM daily_sales WHERE {' AND '.join(filters)} ORDER BY sale_date DESC LIMIT :limit"
        )
        params["limit"] = limit

        with self.engine.connect() as conn:
            rows = conn.execute(query, params).mappings().all()

        return [DailySaleItem(**row) for row in rows]

    def fetch_top_products(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 10,
    ) -> list[ProductTopItem]:
        filters = ["TRUE"]
        params: dict[str, Any] = {}

        if start_date is not None:
            filters.append("o.invoice_date::date >= :start_date")
            params["start_date"] = start_date
        if end_date is not None:
            filters.append("o.invoice_date::date <= :end_date")
            params["end_date"] = end_date

        query = text(
            f"""
            SELECT
                p.product_id,
                p.description,
                COALESCE(SUM(i.sales_amount), 0) AS total_revenue,
                COALESCE(SUM(i.quantity), 0) AS total_quantity
            FROM order_items i
            JOIN orders o ON i.order_id = o.order_id
            JOIN products p ON i.product_id = p.product_id
            WHERE {' AND '.join(filters)}
            GROUP BY p.product_id, p.description
            ORDER BY total_revenue DESC
            LIMIT :limit
            """
        )
        params["limit"] = limit

        with self.engine.connect() as conn:
            rows = conn.execute(query, params).mappings().all()

        return [ProductTopItem(**row) for row in rows]

    def fetch_top_customers(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 10,
    ) -> list[CustomerTopItem]:
        filters = ["TRUE"]
        params: dict[str, Any] = {}

        if start_date is not None:
            filters.append("o.invoice_date::date >= :start_date")
            params["start_date"] = start_date
        if end_date is not None:
            filters.append("o.invoice_date::date <= :end_date")
            params["end_date"] = end_date

        query = text(
            f"""
            SELECT
                c.customer_id,
                c.country,
                COALESCE(SUM(o.total_amount), 0) AS total_revenue,
                COUNT(o.order_id) AS total_orders
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE {' AND '.join(filters)}
            GROUP BY c.customer_id, c.country
            ORDER BY total_revenue DESC
            LIMIT :limit
            """
        )
        params["limit"] = limit

        with self.engine.connect() as conn:
            rows = conn.execute(query, params).mappings().all()

        return [CustomerTopItem(**row) for row in rows]

    def fetch_segments(self, limit: int = 50) -> list[SegmentItem]:
        query = text(
            "SELECT customer_id, segment, recency, frequency, monetary, r_score, f_score, m_score FROM customer_segments ORDER BY customer_id LIMIT :limit"
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"limit": limit}).mappings().all()

        return [SegmentItem(**row) for row in rows]

    def fetch_forecast(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 30,
    ) -> list[ForecastItem]:
        filters = ["TRUE"]
        params: dict[str, Any] = {}

        if start_date is not None:
            filters.append("forecast_date >= :start_date")
            params["start_date"] = start_date
        if end_date is not None:
            filters.append("forecast_date <= :end_date")
            params["end_date"] = end_date

        query = text(
            f"SELECT forecast_date, forecast_value, model_name FROM forecasts WHERE {' AND '.join(filters)} ORDER BY forecast_date DESC LIMIT :limit"
        )
        params["limit"] = limit

        with self.engine.connect() as conn:
            rows = conn.execute(query, params).mappings().all()

        return [ForecastItem(**row) for row in rows]


def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def build_prediction(request_data: dict[str, Any], model: Any) -> dict[str, Any]:
    frame = pd.DataFrame([request_data])
    if hasattr(model, "feature_names_in_"):
        frame = frame.reindex(columns=list(model.feature_names_in_))
    prediction = float(model.predict(frame)[0])
    return {"predicted_revenue": prediction, "model_name": "best_forecast_model"}


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not configured. Please set it in your environment or .env file.")
    return database_url
