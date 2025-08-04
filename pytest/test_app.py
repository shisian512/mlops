# tests/test_app.py

import pytest
from fastapi.testclient import TestClient
from app import app, FEATURE_COUNT
from unittest.mock import patch
import numpy as np

client = TestClient(app)

# ─── Test: Predict with Valid Input ────────────────────────────────────────────
def test_predict_valid_input():
    test_data = [[1.0] * FEATURE_COUNT]
    expected_prediction = [42.0]

    with patch("app.model") as mock_model:
        mock_model.predict.return_value = np.array(expected_prediction)
        response = client.post("/predict", json={"data": test_data})

        assert response.status_code == 200
        assert response.json() == {"predictions": expected_prediction}

# ─── Test: Model is None ───────────────────────────────────────────────────────
def test_predict_model_none():
    with patch("app.model", None):
        response = client.post("/predict", json={"data": [[1.0] * FEATURE_COUNT]})
        assert response.status_code == 503
        assert "No model is currently available" in response.json()["detail"]

# ─── Test: Input with Wrong Feature Count ──────────────────────────────────────
def test_predict_wrong_feature_count():
    wrong_data = [[1.0] * (FEATURE_COUNT + 1)]
    with patch("app.model") as mock_model:
        mock_model.predict.return_value = [0.0]
        response = client.post("/predict", json={"data": wrong_data})
        assert response.status_code == 422
        assert f"Each record must have exactly {FEATURE_COUNT} features" in response.json()["detail"]

# ─── Test: Predict Raises Exception ────────────────────────────────────────────
def test_predict_model_exception():
    test_data = [[1.0] * FEATURE_COUNT]

    with patch("app.model") as mock_model:
        mock_model.predict.side_effect = Exception("Mock prediction error")

        response = client.post("/predict", json={"data": test_data})
        assert response.status_code == 500
        assert "Prediction error: Mock prediction error" in response.json()["detail"]
