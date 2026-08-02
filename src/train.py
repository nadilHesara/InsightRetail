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
    mae = mean_absolute_error(actual, predicted)
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    mape = (abs((actual - predicted) / actual).replace([float("inf"), -float("inf")], float("nan")).mean() * 100)
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


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
    models["random_forest"] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.1, random_state=42)
    except ImportError:
        print("XGBoost is not installed; skipping XGBoost model.")

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
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


def evaluate_on_test(best_model: object, test_df: pd.DataFrame) -> Dict[str, float]:
    """Evaluate the selected model on the test set."""
    X_test, y_test = build_feature_matrix(test_df)
    preds = best_model.predict(X_test)
    return calculate_metrics(y_test, preds)


def save_outputs(best_model: object, results_df: pd.DataFrame, test_metrics: Dict[str, float], output_dir: str | Path) -> None:
    """Save the trained model and evaluation metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "best_forecast_model.joblib"
    metrics_path = output_dir / "model_metrics.json"

    joblib.dump(best_model, model_path)
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
    test_metrics = evaluate_on_test(best_model, test_df)

    # Use the best model to create final predictions on the test set
    X_test, y_test = build_feature_matrix(test_df)
    predictions = best_model.predict(X_test)

    save_outputs(best_model, results_df, test_metrics, "models")
    plot_results(y_test, pd.Series(predictions, index=y_test.index), results_df, "reports")

    summary = {
        "best_model": best_model_name,
        "validation_results": results_df.to_dict(orient="records"),
        "test_metrics": test_metrics,
    }

    return summary


if __name__ == "__main__":
    run_training()
