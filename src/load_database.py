from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


load_dotenv()


def get_engine() -> object:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set. Create a .env file or export the variable first.")

    print(f"Connecting to PostgreSQL database...")
    return create_engine(database_url)


def create_schema(engine: object) -> None:
    schema_path = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with engine.begin() as connection:
        sql = schema_path.read_text(encoding="utf-8")
        connection.execute(text(sql))
    print("Database schema created successfully.")


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def safe_upsert(engine: object, table_name: str, df: pd.DataFrame, key_columns: list[str]) -> None:
    if df.empty:
        print(f"No rows to insert for {table_name}.")
        return

    with engine.begin() as connection:
        df.to_sql(table_name, con=connection, if_exists="append", index=False, method="multi")
    print(f"Loaded {len(df)} rows into {table_name}.")


def load_processed_data(engine: object) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cleaned_path = repo_root / "data" / "processed" / "cleaned_retail_data.csv"
    segments_path = repo_root / "data" / "processed" / "customer_segments.csv"

    cleaned_df = load_csv(cleaned_path)
    segments_df = load_csv(segments_path)

    if "customerid" in cleaned_df.columns:
        cleaned_df["customerid"] = pd.to_numeric(cleaned_df["customerid"], errors="coerce")
    if "invoiceno" in cleaned_df.columns:
        cleaned_df["invoiceno"] = cleaned_df["invoiceno"].astype(str)
    if "invoicedate" in cleaned_df.columns:
        cleaned_df["invoicedate"] = pd.to_datetime(cleaned_df["invoicedate"], errors="coerce")

    customers_df = (
        cleaned_df[["customerid", "country"]]
        .dropna(subset=["customerid"])
        .drop_duplicates(subset=["customerid"])
        .rename(columns={"customerid": "customer_id", "country": "country"})
    )

    products_df = (
        cleaned_df[["stockcode", "description"]]
        .dropna(subset=["stockcode"])
        .drop_duplicates(subset=["stockcode"])
        .rename(columns={"stockcode": "product_id", "description": "description"})
    )

    orders_df = (
        cleaned_df[["invoiceno", "customerid", "invoicedate", "country", "salesamount"]]
        .dropna(subset=["invoiceno"])
        .drop_duplicates(subset=["invoiceno"])
        .rename(columns={"invoiceno": "order_id", "customerid": "customer_id", "invoicedate": "invoice_date", "salesamount": "total_amount"})
    )
    orders_df["invoice_date"] = pd.to_datetime(orders_df["invoice_date"], errors="coerce")

    order_items_df = cleaned_df[["invoiceno", "stockcode", "quantity", "unitprice", "salesamount"]].copy()
    order_items_df = order_items_df.rename(columns={"invoiceno": "order_id", "stockcode": "product_id", "quantity": "quantity", "unitprice": "unit_price", "salesamount": "sales_amount"})
    order_items_df = order_items_df.dropna(subset=["order_id", "product_id"]).drop_duplicates()

    segments_df = segments_df.rename(columns={"customerid": "customer_id"})
    segments_df["customer_id"] = pd.to_numeric(segments_df["customer_id"], errors="coerce")
    customer_segments_df = segments_df[["customer_id", "segment", "recency", "frequency", "monetary", "r_score", "f_score", "m_score"]].dropna(subset=["customer_id"]).drop_duplicates(subset=["customer_id"])

    daily_sales_df = (
        cleaned_df.assign(
            sale_date=pd.to_datetime(
                cleaned_df["invoicedate"],
                errors="coerce"
            ).dt.date
        )
        .dropna(subset=["sale_date"])
        .groupby("sale_date", as_index=False)
        .agg(
            revenue=("salesamount", "sum"),
            orders_count=("invoiceno", "nunique"),
        )
    )

    forecasts_df = pd.DataFrame(
        {
            "forecast_date": [pd.Timestamp.today().date()],
            "forecast_value": [cleaned_df["salesamount"].sum()],
            "model_name": ["rule_based_baseline"],
        }
    )

    safe_upsert(engine, "customers", customers_df, ["customer_id"])
    safe_upsert(engine, "products", products_df, ["product_id"])
    safe_upsert(engine, "orders", orders_df, ["order_id"])
    safe_upsert(engine, "order_items", order_items_df, ["order_id", "product_id"])
    safe_upsert(engine, "customer_segments", customer_segments_df, ["customer_id"])
    safe_upsert(engine, "daily_sales", daily_sales_df, ["sale_date"])
    safe_upsert(engine, "forecasts", forecasts_df, ["forecast_date", "model_name"])

    print("Processed data loaded successfully.")


def main() -> None:
    try:
        engine = get_engine()
        create_schema(engine)
        load_processed_data(engine)
        print("Database loading completed.")
    except (ValueError, FileNotFoundError, SQLAlchemyError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
