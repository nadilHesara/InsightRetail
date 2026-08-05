from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "InsightRetail API is running"}


def test_swagger_documentation_is_available():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "Swagger UI" in response.text


def test_predict_endpoint_returns_validation_error():
    response = client.post("/predict", json={"prev_day_revenue": -1})
    assert response.status_code == 422


def test_summary_endpoint_requires_database_url():
    backup = os.environ.pop("DATABASE_URL", None)
    response = client.get("/summary")
    assert response.status_code == 500
    assert "DATABASE_URL is not configured" in response.json()["detail"]
    if backup is not None:
        os.environ["DATABASE_URL"] = backup
