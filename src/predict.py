from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_forecast_model.joblib"
HISTORY_PATH = ROOT_DIR / "data" / "processed" / "daily_sales_features.csv"
OUTPUT_FORECAST_PATH = ROOT_DIR / "data" / "processed" / "future_30_day_forecast.csv"
OUTPUT_FIGURE_PATH = ROOT_DIR / "reports" / "figures" / "future_forecast.png"

FEATURE_COLUMNS = [
    "prev_day_revenue",
    "revenue_7_days_ago",
    "revenue_14_days_ago",
    "rolling_7d_avg",
    "rolling_30d_avg",
    "rolling_7d_std",
    "day_of_week",
    "month",
    "weekend",
]


def load_history_data(path: Path | str = HISTORY_PATH) -> pd.DataFrame:
    """Load the historical daily sales feature table used during training."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Historical feature data not found: {path}")

    df = pd.read_csv(path)
    if "sale_date" not in df.columns or "revenue" not in df.columns:
        raise KeyError("Expected sale_date and revenue columns in the historical data.")

    df["sale_date"] = pd.to_datetime(df["sale_date"])
    return df.sort_values("sale_date").reset_index(drop=True)


def build_feature_row(history_df: pd.DataFrame, next_date: pd.Timestamp, day_of_week_categories: list[str], month_categories: list[str]) -> dict[str, Any]:
    """Build one feature row for a future day using the most recent known revenue history."""
    revenue_series = history_df["revenue"].astype(float)

    if revenue_series.empty:
        raise ValueError("Historical revenue data is empty.")

    prev_day_revenue = float(revenue_series.iloc[-1])
    revenue_7_days_ago = float(revenue_series.iloc[-7]) if len(revenue_series) >= 7 else float(revenue_series.iloc[-1])
    revenue_14_days_ago = float(revenue_series.iloc[-14]) if len(revenue_series) >= 14 else float(revenue_series.iloc[-1])

    recent_values = revenue_series.tail(7)
    rolling_7d_avg = float(recent_values.mean()) if not recent_values.empty else 0.0
    rolling_30d_avg = float(revenue_series.tail(30).mean()) if len(revenue_series) >= 30 else float(revenue_series.mean())
    rolling_7d_std = float(revenue_series.tail(7).std(ddof=0)) if len(revenue_series) >= 7 else 0.0

    day_name = next_date.day_name()
    month_name = next_date.month_name()
    weekend = int(next_date.dayofweek in [5, 6])

    day_code = pd.Categorical([day_name], categories=day_of_week_categories).codes[0]
    month_code = pd.Categorical([month_name], categories=month_categories).codes[0]

    return {
        "prev_day_revenue": prev_day_revenue,
        "revenue_7_days_ago": revenue_7_days_ago,
        "revenue_14_days_ago": revenue_14_days_ago,
        "rolling_7d_avg": rolling_7d_avg,
        "rolling_30d_avg": rolling_30d_avg,
        "rolling_7d_std": rolling_7d_std,
        "day_of_week": day_code,
        "month": month_code,
        "weekend": weekend,
    }


def build_feature_frame(history_df: pd.DataFrame, future_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Create a feature matrix for the next 30 forecast days using the same feature names and order as training."""
    day_of_week_categories = sorted(history_df["day_of_week"].dropna().astype(str).unique())
    month_categories = sorted(history_df["month"].dropna().astype(str).unique())

    rows = []
    for forecast_date in future_dates:
        row = build_feature_row(history_df, forecast_date, day_of_week_categories, month_categories)
        rows.append(row)

    feature_frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return feature_frame


def recursive_forecast(model: Any, history_df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    """Forecast one day at a time and feed each prediction into the next day’s feature construction.

    This is a simple recursive approach: after predicting day t, that prediction becomes the latest
    revenue value for the next step. It is useful for a lightweight baseline, but it can accumulate
    error over time because later predictions depend on earlier ones.
    """
    history = history_df[["sale_date", "revenue"]].copy()
    history = history.sort_values("sale_date").reset_index(drop=True)

    last_date = history["sale_date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    results = []
    for forecast_date in future_dates:
        feature_row = build_feature_row(history, forecast_date, sorted(history_df["day_of_week"].dropna().astype(str).unique()), sorted(history_df["month"].dropna().astype(str).unique()))
        feature_frame = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)

        # Use the model's fitted feature names when available and keep the column order aligned.
        if hasattr(model, "feature_names_in_"):
            feature_frame = feature_frame.reindex(columns=list(model.feature_names_in_))

        prediction = float(model.predict(feature_frame)[0])
        results.append({
            "forecast_date": forecast_date,
            "predicted_revenue": prediction,
            "day_of_week": forecast_date.day_name(),
            "weekend": int(forecast_date.dayofweek in [5, 6]),
        })

        history.loc[len(history)] = [forecast_date, prediction]

    return pd.DataFrame(results)


def save_forecast_output(forecast_df: pd.DataFrame, output_path: Path | str = OUTPUT_FORECAST_PATH) -> None:
    """Save the forecast table to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_path, index=False)


def plot_forecast(history_df: pd.DataFrame, forecast_df: pd.DataFrame, output_path: Path | str = OUTPUT_FIGURE_PATH) -> None:
    """Create a simple line chart with historical and future revenue."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    historical_plot = history_df[["sale_date", "revenue"]].copy()
    historical_plot = historical_plot.rename(columns={"sale_date": "date", "revenue": "revenue"})

    forecast_plot = forecast_df[["forecast_date", "predicted_revenue"]].copy()
    forecast_plot = forecast_plot.rename(columns={"forecast_date": "date", "predicted_revenue": "revenue"})

    combined = pd.concat([historical_plot, forecast_plot], ignore_index=True)

    plt.figure(figsize=(10, 4))
    plt.plot(combined["date"], combined["revenue"], color="tab:blue", linewidth=1.5)
    plt.axvline(forecast_plot["date"].iloc[0], color="tab:red", linestyle="--", linewidth=1)
    plt.title("Historical and future sales revenue")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    """Load the trained model and generate a 30-day revenue forecast."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    history_df = load_history_data()
    model = joblib.load(MODEL_PATH)

    forecast_df = recursive_forecast(model, history_df, horizon=30)
    save_forecast_output(forecast_df)
    plot_forecast(history_df, forecast_df)

    print(f"Saved forecasts to: {OUTPUT_FORECAST_PATH}")
    print(f"Saved figure to: {OUTPUT_FIGURE_PATH}")


if __name__ == "__main__":
    main()
