from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "best_forecast_model.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.json"


@st.cache_data(show_spinner=False)
def load_sales_data() -> pd.DataFrame:
    """Load cleaned retail sales data with the expected columns."""
    path = DATA_DIR / "cleaned_retail_data.csv"
    if not path.exists():
        st.error(f"Missing file: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    for column in ["invoicedate", "salesamount", "quantity", "unitprice"]:
        if column in df.columns:
            if column == "invoicedate":
                df[column] = pd.to_datetime(df[column], errors="coerce")
            else:
                df[column] = pd.to_numeric(df[column], errors="coerce")

    if "salesamount" not in df.columns:
        df["salesamount"] = df["quantity"] * df["unitprice"]

    df = df.dropna(subset=["invoicedate"]).copy()
    df["invoice_date"] = df["invoicedate"]
    df["date"] = df["invoice_date"].dt.date
    df["month"] = df["invoice_date"].dt.to_period("M").astype(str)
    df["weekday"] = df["invoice_date"].dt.day_name()
    df["hour"] = df["invoice_date"].dt.hour
    df["country"] = df.get("country", pd.Series(["Unknown"] * len(df), index=df.index))
    df["country"] = df["country"].fillna("Unknown")
    df["description"] = df.get("description", pd.Series(["Unknown"] * len(df), index=df.index))
    df["description"] = df["description"].fillna("Unknown")
    df["customerid"] = df.get("customerid", pd.Series(["Unknown"] * len(df), index=df.index))
    return df


@st.cache_data(show_spinner=False)
def load_customer_segments() -> pd.DataFrame:
    """Load the saved customer segmentation results."""
    path = DATA_DIR / "customer_segments.csv"
    if not path.exists():
        return pd.DataFrame(columns=["customerid", "segment", "monetary"])

    df = pd.read_csv(path)
    if "customerid" in df.columns:
        df["customerid"] = df["customerid"].astype(str)
    if "segment" in df.columns:
        df["segment"] = df["segment"].fillna("Unknown")
    return df


@st.cache_data(show_spinner=False)
def load_future_forecast() -> pd.DataFrame:
    """Load the generated future 30-day forecast output."""
    path = DATA_DIR / "future_30_day_forecast.csv"
    if not path.exists():
        return pd.DataFrame(columns=["forecast_date", "predicted_revenue", "day_of_week", "weekend"])

    df = pd.read_csv(path)
    df["forecast_date"] = pd.to_datetime(df["forecast_date"], errors="coerce")
    return df.dropna(subset=["forecast_date"]).copy()


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any]:
    """Load model evaluation metrics if available."""
    path = METRICS_PATH
    if not path.exists():
        return {"mae": None, "rmse": None, "mape": None}

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = pd.read_json(handle, typ="series")
        return data.to_dict()
    except Exception:
        return {"mae": None, "rmse": None, "mape": None}


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    """Load validation comparison rows from the training summary if available."""
    path = ROOT / "models" / "comparison_results.csv"
    if not path.exists():
        return pd.DataFrame(columns=["model", "mae", "rmse", "mape"])

    return pd.read_csv(path)


def apply_filters(df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared sidebar filters to the selected dataframe."""
    if df.empty:
        return df

    if "date" in df.columns:
        start_date = st.sidebar.date_input("Start date", value=pd.Timestamp(df["date"].min()).date(), key="start_date")
        end_date = st.sidebar.date_input("End date", value=pd.Timestamp(df["date"].max()).date(), key="end_date")
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        mask = (df["date"] >= start_ts) & (df["date"] < end_ts)
        df = df.loc[mask].copy()

    if "country" in df.columns:
        countries = sorted(df["country"].dropna().astype(str).unique())
        selected_countries = st.sidebar.multiselect("Country", countries, default=countries)
        if selected_countries:
            df = df[df["country"].astype(str).isin(selected_countries)].copy()

    if "description" in df.columns:
        products = sorted(df["description"].dropna().astype(str).unique())
        selected_products = st.sidebar.multiselect("Product", products, default=products)
        if selected_products:
            df = df[df["description"].astype(str).isin(selected_products)].copy()

    if "segment" in df.columns:
        segments = sorted(df["segment"].dropna().astype(str).unique())
        selected_segments = st.sidebar.multiselect("Customer segment", segments, default=segments)
        if selected_segments:
            df = df[df["segment"].astype(str).isin(selected_segments)].copy()

    return df


def show_empty_state(message: str) -> None:
    """Display a simple empty-state message when a filtered view has no rows."""
    st.info(message)


def metric_card(title: str, value: Any, help_text: str | None = None) -> None:
    """Render a simple metric card."""
    with st.container():
        st.metric(label=title, value=value, help=help_text)


def render_overview(sales_df: pd.DataFrame) -> None:
    """Render the Overview page."""
    st.subheader("Overview")
    if sales_df.empty:
        show_empty_state("No sales data is available for the current filters.")
        return

    total_revenue = sales_df["salesamount"].sum()
    unique_orders = sales_df["invoiceno"].nunique() if "invoiceno" in sales_df.columns else 0
    unique_customers = sales_df["customerid"].nunique() if "customerid" in sales_df.columns else 0
    products_sold = sales_df["description"].nunique() if "description" in sales_df.columns else 0
    avg_order_value = total_revenue / unique_orders if unique_orders else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total revenue", f"${total_revenue:,.0f}")
    col2.metric("Unique orders", f"{unique_orders:,}")
    col3.metric("Customers", f"{unique_customers:,}")
    col4.metric("Products sold", f"{products_sold:,}")
    col5.metric("Average order value", f"${avg_order_value:,.2f}")

    cancellation_rate = 0
    if "is_cancelled" in sales_df.columns:
        cancellation_rate = sales_df["is_cancelled"].mean() * 100
    st.metric("Cancellation rate", f"{cancellation_rate:.2f}%")


def render_sales_analysis(sales_df: pd.DataFrame) -> None:
    """Render the Sales Analysis page."""
    st.subheader("Sales Analysis")
    if sales_df.empty:
        show_empty_state("No sales data is available for the current filters.")
        return

    daily = sales_df.groupby("date", as_index=False)["salesamount"].sum()
    monthly = sales_df.groupby("month", as_index=False)["salesamount"].sum()
    by_country = sales_df.groupby("country", as_index=False)["salesamount"].sum().sort_values("salesamount", ascending=False)
    by_weekday = sales_df.groupby("weekday", as_index=False)["salesamount"].sum()
    by_hour = sales_df.groupby("hour", as_index=False)["salesamount"].sum()

    st.plotly_chart(px.line(daily, x="date", y="salesamount", title="Daily revenue"), use_container_width=True)
    st.plotly_chart(px.bar(monthly, x="month", y="salesamount", title="Monthly revenue"), use_container_width=True)
    st.plotly_chart(px.bar(by_country.head(15), x="country", y="salesamount", title="Revenue by country"), use_container_width=True)
    st.plotly_chart(px.bar(by_weekday, x="weekday", y="salesamount", title="Revenue by weekday"), use_container_width=True)
    st.plotly_chart(px.bar(by_hour, x="hour", y="salesamount", title="Revenue by hour"), use_container_width=True)


def render_product_analysis(sales_df: pd.DataFrame) -> None:
    """Render the Product Analysis page."""
    st.subheader("Product Analysis")
    if sales_df.empty:
        show_empty_state("No sales data is available for the current filters.")
        return

    revenue_by_product = sales_df.groupby("description", as_index=False).agg(revenue=("salesamount", "sum"), quantity=("quantity", "sum"))
    top_revenue = revenue_by_product.sort_values("revenue", ascending=False).head(10)
    top_quantity = revenue_by_product.sort_values("quantity", ascending=False).head(10)

    cancelled_df = sales_df[sales_df.get("is_cancelled", False).astype(bool)] if "is_cancelled" in sales_df.columns else pd.DataFrame(columns=sales_df.columns)
    cancelled_products = cancelled_df.groupby("description", as_index=False).size().sort_values("size", ascending=False).head(10) if not cancelled_df.empty else pd.DataFrame(columns=["description", "size"])

    low_performing = revenue_by_product.sort_values("revenue", ascending=True).head(10)

    st.plotly_chart(px.bar(top_revenue, x="description", y="revenue", title="Top products by revenue"), use_container_width=True)
    st.plotly_chart(px.bar(top_quantity, x="description", y="quantity", title="Top products by quantity"), use_container_width=True)
    st.plotly_chart(px.bar(cancelled_products, x="description", y="size", title="Frequently cancelled products"), use_container_width=True)
    st.plotly_chart(px.bar(low_performing, x="description", y="revenue", title="Low-performing products"), use_container_width=True)


def render_customer_analysis(sales_df: pd.DataFrame, segments_df: pd.DataFrame) -> None:
    """Render the Customer Analysis page."""
    st.subheader("Customer Analysis")
    if sales_df.empty:
        show_empty_state("No sales data is available for the current filters.")
        return

    if segments_df.empty:
        show_empty_state("Customer segment data is not available yet.")
        return

    segment_counts = segments_df["segment"].value_counts().reset_index()
    segment_counts.columns = ["segment", "customers"]
    segment_revenue = segments_df.groupby("segment", as_index=False)["monetary"].sum() if "monetary" in segments_df.columns else pd.DataFrame(columns=["segment", "monetary"])

    st.plotly_chart(px.bar(segment_counts, x="segment", y="customers", title="Customers per segment"), use_container_width=True)
    st.plotly_chart(px.bar(segment_revenue, x="segment", y="monetary", title="Revenue per segment"), use_container_width=True)

    top_customers = segments_df.sort_values("monetary", ascending=False).head(10)
    st.dataframe(top_customers[["customerid", "segment", "monetary"]].rename(columns={"monetary": "revenue"}), use_container_width=True)


def render_forecasting(forecast_df: pd.DataFrame, metrics: dict[str, Any], comparison_df: pd.DataFrame) -> None:
    """Render the Forecasting page."""
    st.subheader("Forecasting")

    if forecast_df.empty:
        show_empty_state("Future forecast data is not available yet.")
        return

    st.plotly_chart(px.line(forecast_df, x="forecast_date", y="predicted_revenue", title="Future 30-day forecast"), use_container_width=True)

    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{metrics.get('mae', None) if metrics.get('mae') is not None else 'n/a'}")
        col2.metric("RMSE", f"{metrics.get('rmse', None) if metrics.get('rmse') is not None else 'n/a'}")
        col3.metric("MAPE", f"{metrics.get('mape', None) if metrics.get('mape') is not None else 'n/a'}")

    if not comparison_df.empty:
        st.dataframe(comparison_df, use_container_width=True)


def main() -> None:
    """Create the Streamlit app layout and navigation."""
    st.set_page_config(page_title="InsightRetail Dashboard", page_icon="📈", layout="wide")
    st.title("InsightRetail: Retail Sales Analytics and Demand Forecasting Platform")
    st.caption("Interactive dashboards built from the processed retail data and forecasting outputs.")

    sales_df = load_sales_data()
    segments_df = load_customer_segments()
    forecast_df = load_future_forecast()
    metrics = load_metrics()
    comparison_df = load_model_comparison()

    if sales_df.empty and segments_df.empty and forecast_df.empty:
        st.error("The expected processed files are not available. Please run the preprocessing scripts first.")
        return

    with st.sidebar:
        st.header("Filters")
        sales_df_filtered = apply_filters(sales_df, sales_df)
        segments_df_filtered = segments_df.copy()
        if "segment" in segments_df_filtered.columns:
            segments_df_filtered = segments_df_filtered[segments_df_filtered["segment"].astype(str).isin([s for s in st.session_state.get("segment", []) if False])]
        if not sales_df_filtered.empty and "customerid" in sales_df_filtered.columns and "customerid" in segments_df_filtered.columns:
            segments_df_filtered = segments_df_filtered[segments_df_filtered["customerid"].astype(str).isin(sales_df_filtered["customerid"].astype(str).unique())]

    page = st.sidebar.radio("Navigate", ["Overview", "Sales Analysis", "Product Analysis", "Customer Analysis", "Forecasting"])

    if page == "Overview":
        render_overview(sales_df_filtered)
    elif page == "Sales Analysis":
        render_sales_analysis(sales_df_filtered)
    elif page == "Product Analysis":
        render_product_analysis(sales_df_filtered)
    elif page == "Customer Analysis":
        render_customer_analysis(sales_df_filtered, segments_df_filtered)
    else:
        render_forecasting(forecast_df, metrics, comparison_df)


if __name__ == "__main__":
    main()
