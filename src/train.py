from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover - depends on optional install
    Prophet = None

warnings.filterwarnings("ignore")


def load_feature_data(path: str | Path) -> pd.DataFrame:
    """Load the prepared forecasting dataset and sort it by date."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_csv(path)
    if "sale_date" not in df.columns:
        raise KeyError("The forecasting file must contain a sale_date column.")

    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = df.sort_values("sale_date").reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data chronologically into train, validation, and test sets."""
    n_rows = len(df)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Create features and target for modeling."""
    feature_columns = [
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

    X = df[feature_columns].copy()
    X["day_of_week"] = X["day_of_week"].astype("category").cat.codes
    X["month"] = X["month"].astype("category").cat.codes

    y = df["target_next_day_revenue"].copy()

    complete_rows = X.notna().all(axis=1) & y.notna()
    X = X.loc[complete_rows].copy()
    y = y.loc[complete_rows].copy()
    return X, y


def make_baseline_predictions(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Create simple baseline forecasts."""
    predictions = {}
    predictions["today"] = df["revenue"].shift(1).fillna(0)
    predictions["7_days_ago"] = df["revenue_7_days_ago"].fillna(0)
    predictions["rolling_7d_avg"] = df["rolling_7d_avg"].fillna(0)
    return predictions


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """Calculate MAE, RMSE, and MAPE."""
    actual = pd.Series(actual).astype(float)
    predicted = pd.Series(predicted).astype(float)
    predicted = predicted.set_axis(actual.index)

    valid_mask = actual.abs() > 1e-8
    actual_valid = actual.loc[valid_mask]
    predicted_valid = predicted.loc[valid_mask]

    mae = mean_absolute_error(actual, predicted)
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    mape = float("nan")
    if not actual_valid.empty:
        mape = (abs((actual_valid - predicted_valid) / actual_valid).replace([float("inf"), -float("inf")], float("nan")).mean() * 100)

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def forecast_with_sarima(history: pd.Series) -> float:
    """Forecast the next value using a simple SARIMA model."""
    series = history.astype(float).dropna()
    if len(series) < 10:
        return float(series.iloc[-1]) if not series.empty else 0.0

    sarima_model = ARIMA(series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7))
    sarima_result = sarima_model.fit(method_kwargs={"maxiter": 50})
    forecast = sarima_result.forecast(steps=1)
    return float(forecast.iloc[0])


def forecast_with_prophet(history: pd.Series) -> float:
    """Forecast the next value using Prophet on the recent history."""
    if Prophet is None:
        raise ImportError("Prophet is not installed.")

    history_df = history.reset_index()
    history_df.columns = ["ds", "y"]
    history_df["ds"] = pd.to_datetime(history_df["ds"])
    history_df = history_df.sort_values("ds")
    history_df = history_df.dropna()

    if len(history_df) < 10:
        return float(history_df["y"].iloc[-1]) if not history_df.empty else 0.0

    prophet_model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False)
    prophet_model.fit(history_df)
    future = prophet_model.make_future_dataframe(periods=1, freq="D")
    forecast = prophet_model.predict(future)
    return float(forecast["yhat"].iloc[-1])


def evaluate_time_series_model(model_name: str, train_df: pd.DataFrame, eval_df: pd.DataFrame, actuals: pd.Series) -> pd.Series:
    """Evaluate a time-series model with a rolling one-step forecast on the validation window."""
    predictions = []
    history_df = train_df[["sale_date", "revenue"]].copy()
    history_df = history_df.rename(columns={"sale_date": "ds", "revenue": "y"})

    for _, row in eval_df.iterrows():
        history_series = history_df.set_index("ds")["y"].astype(float).sort_index()
        if model_name == "sarima":
            prediction = forecast_with_sarima(history_series)
        elif model_name == "prophet":
            prediction = forecast_with_prophet(history_series)
        else:
            raise ValueError(f"Unsupported time-series model: {model_name}")

        predictions.append(prediction)
        history_df = pd.concat(
            [history_df, pd.DataFrame({"ds": [row["sale_date"]], "y": [row["revenue"]]})],
            ignore_index=True,
        )

    return pd.Series(predictions, index=actuals.index)


def get_available_model_names() -> list[str]:
    """Return the list of supported model names for the training workflow."""
    names = ["linear_regression", "random_forest", "sarima", "prophet"]
    try:
        from xgboost import XGBRegressor  # noqa: F401

        names.append("xgboost")
    except ImportError:
        pass
    return names


def evaluate_baselines(val_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the simple baseline models on the validation set."""
    actual = val_df["target_next_day_revenue"]
    baseline_preds = make_baseline_predictions(val_df)

    rows = []
    for name, preds in baseline_preds.items():
        metrics = calculate_metrics(actual, preds)
        rows.append({"model": name, "split": "validation", **metrics})

    return pd.DataFrame(rows)


def train_models(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Train baseline and ML models on the training data and evaluate on validation data."""
    X_train, y_train = build_feature_matrix(train_df)
    X_val, y_val = build_feature_matrix(val_df)

    models = {}
    models["linear_regression"] = LinearRegression()
    models["random_forest"] = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    except ImportError:
        print("XGBoost is not installed; skipping XGBoost model.")

    if Prophet is not None:
        models["prophet"] = "prophet"
    else:
        print("Prophet is not installed; skipping Prophet model.")

    models["sarima"] = "sarima"

    rows = []
    for name, model in models.items():
        if name in {"linear_regression", "random_forest", "xgboost"}:
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
        elif name == "sarima":
            preds = evaluate_time_series_model("sarima", train_df, val_df, y_val)
        elif name == "prophet":
            preds = evaluate_time_series_model("prophet", train_df, val_df, y_val)
        else:
            continue

        metrics = calculate_metrics(y_val, preds)
        rows.append({"model": name, "split": "validation", **metrics})

    return models, pd.DataFrame(rows)


def select_best_model(results_df: pd.DataFrame, models: Dict[str, object], val_df: pd.DataFrame) -> Tuple[str, object]:
    """Select the best model by validation MAE."""
    results_df = results_df.copy()
    best_row = results_df.sort_values("mae").iloc[0]
    best_model_name = best_row["model"]
    best_model = models[best_model_name]
    return best_model_name, best_model


def evaluate_on_test(best_model: object, best_model_name: str, test_df: pd.DataFrame, train_df: pd.DataFrame) -> Dict[str, float]:
    """Evaluate the selected model on the test set."""
    if best_model_name in {"linear_regression", "random_forest", "xgboost"}:
        X_test, y_test = build_feature_matrix(test_df)
        preds = best_model.predict(X_test)
    elif best_model_name == "sarima":
        train_series = train_df.set_index("sale_date")["revenue"].astype(float)
        test_series = test_df.set_index("sale_date")["revenue"].astype(float)
        sarima_model = ARIMA(train_series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7))
        sarima_result = sarima_model.fit()
        preds = sarima_result.forecast(steps=len(test_series))
        preds.index = test_series.index
        y_test = test_series.astype(float)
    elif best_model_name == "prophet":
        train_frame = train_df[["sale_date", "revenue"]].copy()
        train_frame = train_frame.rename(columns={"sale_date": "ds", "revenue": "y"})
        test_frame = test_df[["sale_date", "revenue"]].copy()
        test_frame = test_frame.rename(columns={"sale_date": "ds", "revenue": "y"})
        prophet_model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False)
        prophet_model.fit(train_frame)
        future = prophet_model.make_future_dataframe(periods=len(test_frame), freq="D")
        forecast = prophet_model.predict(future)
        preds = forecast["yhat"].tail(len(test_frame)).reset_index(drop=True)
        y_test = test_frame["y"].astype(float)
    else:
        raise ValueError(f"Unsupported model: {best_model_name}")

    return calculate_metrics(y_test, preds)


def save_outputs(best_model: object, best_model_name: str, results_df: pd.DataFrame, test_metrics: Dict[str, float], output_dir: str | Path) -> None:
    """Save the trained model and evaluation metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "best_forecast_model.joblib"
    metrics_path = output_dir / "model_metrics.json"

    payload = {"model_name": best_model_name, "model": best_model}
    if best_model_name in {"linear_regression", "random_forest", "xgboost"}:
        joblib.dump(best_model, model_path)
    else:
        joblib.dump(payload, model_path)

    metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


def plot_results(actual: pd.Series, predicted: pd.Series, results_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Create simple charts for actual vs predicted, metrics, and residuals."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(actual.index, actual, label="Actual", color="black")
    plt.plot(actual.index, predicted, label="Predicted", color="tab:blue")
    plt.legend()
    plt.title("Actual vs predicted revenue")
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    results_df.plot(kind="bar", x="model", y="mae", legend=False, title="Validation MAE by model")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(output_dir / "model_metrics.png", dpi=150)
    plt.close()

    residuals = actual - predicted
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=20, color="tab:orange")
    plt.title("Residual errors")
    plt.xlabel("Residual")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals.png", dpi=150)
    plt.close()


def run_training(feature_path: str | Path = "data/processed/daily_sales_features.csv") -> Dict[str, object]:
    """Run the end-to-end training and evaluation workflow."""
    df = load_feature_data(feature_path)
    train_df, val_df, test_df = split_data(df)

    baseline_results = evaluate_baselines(val_df)
    model_dict, model_results = train_models(train_df, val_df)

    results_df = pd.concat([baseline_results, model_results], ignore_index=True)
    best_model_name, best_model = select_best_model(results_df, model_dict, val_df)
    test_metrics = evaluate_on_test(best_model, best_model_name, test_df, train_df)

    # Use the best model to create final predictions on the test set
    if best_model_name in {"linear_regression", "random_forest", "xgboost"}:
        X_test, y_test = build_feature_matrix(test_df)
        predictions = best_model.predict(X_test)
        prediction_series = pd.Series(predictions, index=y_test.index)
    elif best_model_name == "sarima":
        test_series = test_df.set_index("sale_date")["revenue"].astype(float)
        prediction_series = pd.Series(
            [forecast_with_sarima(pd.concat([train_df[["sale_date", "revenue"]].copy(), test_df[["sale_date", "revenue"]].copy()], ignore_index=True).set_index("sale_date")["revenue"].astype(float).iloc[: i + 1]) for i in range(len(test_series))],
            index=test_series.index,
        )
        y_test = test_series.astype(float)
    elif best_model_name == "prophet":
        test_series = test_df.set_index("sale_date")["revenue"].astype(float)
        prediction_series = pd.Series(
            [forecast_with_prophet(pd.concat([train_df[["sale_date", "revenue"]].copy(), test_df[["sale_date", "revenue"]].copy()], ignore_index=True).set_index("sale_date")["revenue"].astype(float).iloc[: i + 1]) for i in range(len(test_series))],
            index=test_series.index,
        )
        y_test = test_series.astype(float)
    else:
        raise ValueError(f"Unsupported model: {best_model_name}")

    save_outputs(best_model, best_model_name, results_df, test_metrics, "models")
    plot_results(y_test, prediction_series, results_df, "reports")

    summary = {
        "best_model": best_model_name,
        "validation_results": results_df.to_dict(orient="records"),
        "test_metrics": test_metrics,
    }

    return summary


if __name__ == "__main__":
    run_training()
