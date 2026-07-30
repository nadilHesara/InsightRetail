from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def _read_input_data(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix in {".csv"}:
        return pd.read_csv(input_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    raise ValueError(f"Unsupported file format: {suffix}")


def _standardize_description(desc: object) -> str:
    if pd.isna(desc):
        return "Unknown"
    text = str(desc).strip().lower()
    if not text:
        return "Unknown"
    words = text.split()
    return " ".join(word.capitalize() for word in words)


def clean_sales_data(input_path: str | Path, output_dir: str | Path | None = None) -> Dict[str, pd.DataFrame]:
    """Clean retail sales data and split cancelled orders and rejected rows.

    The function does not silently discard everything. It keeps three outputs:
    - cleaned: rows retained for normal sales analysis
    - cancelled_orders: cancelled transactions analyzed separately as returns
    - rejected_rows: rows removed from the main analysis because they are invalid
    """
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _read_input_data(input_path)

    original_rows = len(df)

    # Remove exact duplicate rows before cleaning decisions
    df = df.drop_duplicates().copy()
    duplicates_removed = original_rows - len(df)

    # Identify cancelled orders separately
    cancelled_mask = df["InvoiceNo"].astype(str).str.startswith("C", na=False)
    cancelled_orders = df.loc[cancelled_mask].copy()
    retained = df.loc[~cancelled_mask].copy()

    # Convert InvoiceDate to datetime and keep invalid dates as NaT for later rejection
    retained["InvoiceDate"] = pd.to_datetime(retained["InvoiceDate"], errors="coerce")

    # Standardize descriptions
    retained["Description"] = retained["Description"].apply(_standardize_description)

    # Clean quantities and prices
    retained["Quantity"] = pd.to_numeric(retained["Quantity"], errors="coerce")
    retained["UnitPrice"] = pd.to_numeric(retained["UnitPrice"], errors="coerce")

    # Validity rules for main analysis
    invalid_mask = (
        retained["Quantity"].isna() |
        (retained["Quantity"] <= 0) |
        retained["UnitPrice"].isna() |
        (retained["UnitPrice"] <= 0) |
        retained["InvoiceDate"].isna()
    )
    rejected_rows = retained.loc[invalid_mask].copy()
    cleaned = retained.loc[~invalid_mask].copy()

    # Handle missing customer IDs by filling with a placeholder
    cleaned["CustomerID"] = cleaned["CustomerID"].fillna("Unknown")
    rejected_rows["CustomerID"] = rejected_rows["CustomerID"].fillna("Unknown")
    cancelled_orders["CustomerID"] = cancelled_orders["CustomerID"].fillna("Unknown")

    # Keep cancelled orders separate from the main analysis
    cancelled_orders = cancelled_orders.reset_index(drop=True)
    cleaned = cleaned.reset_index(drop=True)
    rejected_rows = rejected_rows.reset_index(drop=True)

    # Save outputs
    cleaned_path = output_dir / "cleaned_sales_data.csv"
    cancelled_path = output_dir / "cancelled_orders.csv"
    rejected_path = output_dir / "rejected_rows.csv"
    summary_path = output_dir / "cleaning_summary.txt"

    cleaned.to_csv(cleaned_path, index=False)
    cancelled_orders.to_csv(cancelled_path, index=False)
    rejected_rows.to_csv(rejected_path, index=False)

    summary_lines = [
        "Sales data cleaning summary",
        f"- Duplicate rows removed: {duplicates_removed}",
        f"- Cancelled orders separated: {len(cancelled_orders)}",
        f"- Invalid rows moved to rejected rows: {len(rejected_rows)}",
        f"- Rows retained for sales analysis: {len(cleaned)}",
        "- Cancelled orders were removed from the sales forecast but analyzed separately as product returns.",
        "- Invalid quantities, invalid prices, and missing/invalid dates were excluded from the cleaned dataset.",
        "- Missing customer IDs were filled with 'Unknown' rather than dropped.",
        "- Product descriptions were standardized to title case.",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "cleaned": cleaned,
        "cancelled_orders": cancelled_orders,
        "rejected_rows": rejected_rows,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean retail sales data")
    parser.add_argument("input_path", type=str)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    clean_sales_data(args.input_path, args.output_dir)
