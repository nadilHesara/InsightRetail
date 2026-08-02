import pandas as pd

from src.train import calculate_metrics, get_available_model_names


def test_available_model_names_include_expanded_forecasters():
    names = get_available_model_names()

    assert "linear_regression" in names
    assert "random_forest" in names
    assert "xgboost" in names
    assert "sarima" in names
    assert "prophet" in names


def test_calculate_metrics_handles_zero_targets_without_nan():
    actual = pd.Series([0.0, 10.0, 20.0])
    predicted = pd.Series([0.0, 11.0, 18.0])

    metrics = calculate_metrics(actual, predicted)

    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert pd.notna(metrics["mape"])


def test_calculate_metrics_accepts_numpy_predictions_without_nan():
    actual = pd.Series([0.0, 10.0, 20.0], index=["a", "b", "c"])
    predicted = [0.0, 11.0, 18.0]

    metrics = calculate_metrics(actual, predicted)

    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert pd.notna(metrics["mape"])
