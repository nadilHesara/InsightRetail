from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_or_build_cleaned_dataset(
    raw_path: str | Path | None = None,
    cleaned_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the cleaned retail data or build it from the raw Excel file if missing."""
    if cleaned_path is None:
        cleaned_path = Path("data/processed/cleaned_retail_data.csv")
    cleaned_path = Path(cleaned_path)

    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
        if "salesamount" not in df.columns:
            df["salesamount"] = df["quantity"] * df["unitprice"]
        return df

    if raw_path is None:
        raw_path = Path("data/raw/Online_Retail.xlsx")
    raw_path = Path(raw_path)

    df = pd.read_excel(raw_path)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
    df["is_cancelled"] = df["invoiceno"].astype(str).str.startswith("C", na=False)

    # Keep only valid, non-cancelled transactions for segment analysis
    df = df[~df["is_cancelled"]].copy()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unitprice"] = pd.to_numeric(df["unitprice"], errors="coerce")
    df = df[(df["quantity"] > 0) & (df["unitprice"] > 0) & df["invoicedate"].notna()].copy()
    df["salesamount"] = df["quantity"] * df["unitprice"]

    # Remove rows without a customer ID before building RFM features
    df = df.dropna(subset=["customerid"]).copy()

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_path, index=False)
    return df


def compute_rfm_metrics(df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Create customer-level RFM metrics: Recency, Frequency, Monetary."""
    if reference_date is None:
        reference_date = df["invoicedate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("customerid")
        .agg(
            last_purchase_date=("invoicedate", "max"),
            frequency=("invoiceno", "nunique"),
            monetary=("salesamount", "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (reference_date - rfm["last_purchase_date"]).dt.days
    rfm = rfm.sort_values("customerid").reset_index(drop=True)
    return rfm


def _score_series(values: pd.Series, descending: bool) -> pd.Series:
    ranked = values.rank(method="first", ascending=not descending)
    bins = pd.qcut(ranked, q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    return bins.astype(int)


def score_rfm(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Assign simple 1-to-5 scores for recency, frequency, and monetary values."""
    rfm_df = rfm_df.copy()
    rfm_df["r_score"] = _score_series(-rfm_df["recency"], descending=True)
    rfm_df["f_score"] = _score_series(rfm_df["frequency"], descending=True)
    rfm_df["m_score"] = _score_series(rfm_df["monetary"], descending=True)
    return rfm_df


def assign_segments(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Create simple, easy-to-understand customer segment labels."""
    rfm_df = rfm_df.copy()

    def segment(row: pd.Series) -> str:
        if row["r_score"] >= 4 and row["f_score"] >= 4 and row["m_score"] >= 4:
            return "High-value customers"
        if row["f_score"] >= 4 and row["r_score"] >= 4:
            return "Loyal customers"
        if row["r_score"] >= 4 and row["f_score"] <= 2:
            return "New customers"
        if row["f_score"] >= 3 and row["m_score"] >= 3:
            return "Regular customers"
        if row["r_score"] <= 2 and row["f_score"] >= 2:
            return "At-risk customers"
        return "Inactive customers"

    rfm_df["segment"] = rfm_df.apply(segment, axis=1)
    return rfm_df


def describe_segment_rules() -> Dict[str, str]:
    """Return plain-language rules for the segments."""
    return {
        "High-value customers": "Recent buyers with high frequency and high spending.",
        "Loyal customers": "Customers who buy often and have made recent purchases.",
        "New customers": "Recent customers who have not bought very often yet.",
        "Regular customers": "Customers with steady buying habits and moderate spending.",
        "At-risk customers": "Customers who have not bought recently but still show some activity.",
        "Inactive customers": "Customers with low recency, low frequency, and low spending.",
    }


def plot_segment_counts(rfm_df: pd.DataFrame, output_path: str | Path | None = None) -> None:
    counts = rfm_df["segment"].value_counts().sort_index()
    counts.plot(kind="bar", figsize=(10, 4), title="Customers by segment")
    plt.xlabel("Segment")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150)
    plt.show()


def plot_segment_revenue(rfm_df: pd.DataFrame, output_path: str | Path | None = None) -> None:
    revenue = rfm_df.groupby("segment")["monetary"].sum().sort_values(ascending=False)
    revenue.plot(kind="bar", figsize=(10, 4), title="Revenue by segment")
    plt.xlabel("Segment")
    plt.ylabel("Revenue")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150)
    plt.show()


def plot_recency_vs_frequency(rfm_df: pd.DataFrame, output_path: str | Path | None = None) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(rfm_df["recency"], rfm_df["frequency"], alpha=0.6)
    plt.title("Recency vs Frequency")
    plt.xlabel("Recency (days)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150)
    plt.show()


def plot_frequency_vs_monetary(rfm_df: pd.DataFrame, output_path: str | Path | None = None) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(rfm_df["frequency"], rfm_df["monetary"], alpha=0.6)
    plt.title("Frequency vs Monetary")
    plt.xlabel("Frequency")
    plt.ylabel("Monetary value")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150)
    plt.show()


def build_customer_segments(
    df: pd.DataFrame,
    cleaned_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build and save the customer segmentation output."""
    if cleaned_path is None:
        cleaned_path = Path("data/processed/cleaned_retail_data.csv")
    cleaned_path = Path(cleaned_path)

    if not cleaned_path.exists():
        df = load_or_build_cleaned_dataset(raw_path=None, cleaned_path=cleaned_path)
    else:
        df = pd.read_csv(cleaned_path)

    rfm_df = compute_rfm_metrics(df)
    rfm_df = score_rfm(rfm_df)
    rfm_df = assign_segments(rfm_df)

    if output_path is None:
        output_path = Path("data/processed/customer_segments.csv")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rfm_df.to_csv(output_path, index=False)
    return rfm_df
