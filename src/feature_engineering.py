from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_cleaned_sales_data(cleaned_path: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned retail data and keep only valid, non-cancelled transactions."""
    if cleaned_path is None:
        cleaned_path = Path("data/processed/cleaned_retail_data.csv")
    cleaned_path = Path(cleaned_path)

    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {cleaned_path}")

    df = pd.read_csv(cleaned_path)

    # Keep only non-cancelled and valid rows
    if "is_cancelled" in df.columns:
        df = df[~df["is_cancelled"]].copy()
    if "invoicedate" in df.columns:
        df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    if "unitprice" in df.columns:
        df["unitprice"] = pd.to_numeric(df["unitprice"], errors="coerce")
    if "salesamount" in df.columns:
        df["salesamount"] = pd.to_numeric(df["salesamount"], errors="coerce")

    valid_mask = (
        df["invoicedate"].notna()
        & df["quantity"].gt(0)
        & df["unitprice"].gt(0)
        & df["salesamount"].notna()
    )
    df = df.loc[valid_mask].copy()

    return df


def build_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into one row per day with total revenue."""
    if "invoicedate" not in df.columns or "salesamount" not in df.columns:
        raise KeyError("Expected invoicedate and salesamount columns in the input data.")

    daily = (
        df.assign(sale_date=df["invoicedate"].dt.date)
        .groupby("sale_date", as_index=False)["salesamount"]
        .sum()
        .rename(columns={"salesamount": "revenue"})
    )
    daily["sale_date"] = pd.to_datetime(daily["sale_date"])
    return daily


def add_missing_calendar_dates(daily_df: pd.DataFrame, start_date: pd.Timestamp | None = None, end_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Fill missing calendar dates with zero revenue."""
    if daily_df.empty:
        raise ValueError("Daily sales dataframe is empty.")

    if start_date is None:
        start_date = daily_df["sale_date"].min()
    if end_date is None:
        end_date = daily_df["sale_date"].max()

    full_date_index = pd.date_range(start=start_date, end=end_date, freq="D")
    full_df = pd.DataFrame({"sale_date": full_date_index})
    daily_df = full_df.merge(daily_df, on="sale_date", how="left")
    daily_df["revenue"] = daily_df["revenue"].fillna(0.0)
    return daily_df.sort_values("sale_date").reset_index(drop=True)


def add_forecast_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged, rolling, and calendar-based features for forecasting."""
    features_df = daily_df.copy()
    features_df = features_df.sort_values("sale_date").reset_index(drop=True)

    features_df["prev_day_revenue"] = features_df["revenue"].shift(1)
    features_df["revenue_7_days_ago"] = features_df["revenue"].shift(7)
    features_df["revenue_14_days_ago"] = features_df["revenue"].shift(14)
    features_df["rolling_7d_avg"] = features_df["revenue"].shift(1).rolling(window=7, min_periods=7).mean()
    features_df["rolling_30d_avg"] = features_df["revenue"].shift(1).rolling(window=30, min_periods=30).mean()
    features_df["rolling_7d_std"] = features_df["revenue"].shift(1).rolling(window=7, min_periods=7).std()

    features_df["day_of_week"] = features_df["sale_date"].dt.day_name()
    features_df["month"] = features_df["sale_date"].dt.month_name()
    features_df["weekend"] = features_df["sale_date"].dt.dayofweek.isin([5, 6]).astype(int)

    features_df["target_next_day_revenue"] = features_df["revenue"].shift(-1)

    return features_df


def validate_dataset(features_df: pd.DataFrame) -> None:
    """Run simple validation checks for the forecasting dataset."""
    if features_df["sale_date"].duplicated().any():
        raise ValueError("Duplicate dates were found in the forecasting dataset.")

    if features_df["sale_date"].isna().any():
        raise ValueError("Missing dates were found in the forecasting dataset.")

    if (features_df["revenue"] < 0).any():
        raise ValueError("Negative revenue values were found.")

    lag_columns = [
        "prev_day_revenue",
        "revenue_7_days_ago",
        "revenue_14_days_ago",
        "rolling_7d_avg",
        "rolling_30d_avg",
        "rolling_7d_std",
    ]

    for col in lag_columns:
        if features_df[col].isna().any():
            raise ValueError(f"Missing lag feature values found in {col}.")

    # Rolling features should be based on earlier days, not the next day's values.
    future_based_7d_avg = features_df["revenue"].shift(-1).rolling(window=7, min_periods=7).mean()
    if features_df["rolling_7d_avg"].equals(future_based_7d_avg):
        raise ValueError("Possible data leakage detected in rolling features.")


def prepare_forecast_dataset(cleaned_path: str | Path | None = None, output_path: str | Path | None = None) -> pd.DataFrame:
    """Create the final feature dataset and save it to disk."""
    cleaned_df = load_cleaned_sales_data(cleaned_path)
    daily_df = build_daily_sales(cleaned_df)
    daily_df = add_missing_calendar_dates(daily_df)
    features_df = add_forecast_features(daily_df)

    # Remove rows with missing lag features created by shifting and rolling windows
    features_df = features_df.dropna(subset=["prev_day_revenue", "revenue_7_days_ago", "revenue_14_days_ago", "rolling_7d_avg", "rolling_30d_avg", "rolling_7d_std"]).copy()

    validate_dataset(features_df)

    if output_path is None:
        output_path = Path("data/processed/daily_sales_features.csv")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print(f"Forecast dataset saved to: {output_path}")
    return features_df


if __name__ == "__main__":
    prepare_forecast_dataset()
