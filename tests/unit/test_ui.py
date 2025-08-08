# tests/test_ui.py

import pytest
from unittest.mock import patch, MagicMock
from ui import get_prediction, render_input_fields

# ─── Test: get_prediction returns prediction from API ──────────────────────────
@patch("ui.requests.post")
def test_get_prediction_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"predictions": [42.0]}
    mock_post.return_value = mock_response

    result = get_prediction([1.0, 2.0, 3.0, 4.0])
    assert result == 42.0

# ─── Test: get_prediction handles 503 model unavailable ────────────────────────
@patch("ui.requests.post")
def test_get_prediction_model_unavailable(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.json.return_value = {"detail": "No model available"}
    mock_post.return_value = mock_response

    with pytest.raises(Exception, match="No model is currently available"):
        get_prediction([1.0, 2.0, 3.0, 4.0])

# ─── Test: get_prediction handles unexpected error ─────────────────────────────
@patch("ui.requests.post")
def test_get_prediction_failure(mock_post):
    mock_post.side_effect = Exception("connection failed")

    with pytest.raises(Exception, match="connection failed"):
        get_prediction([1.0, 2.0, 3.0, 4.0])

# ─── Optional: Smoke test for Streamlit input renderer ─────────────────────────
def test_render_input_fields(monkeypatch):
    import streamlit as st

    # Monkeypatch streamlit.number_input to return mock values
    monkeypatch.setattr(st, "number_input", lambda label, value, format: 1.0)

    values = render_input_fields()
    assert isinstance(values, list)
    assert len(values) == 4
    assert all(isinstance(v, float) for v in values)
