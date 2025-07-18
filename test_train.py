# tests/test_train.py

import pytest
import yaml
import tempfile
import os
import pandas as pd
from unittest.mock import patch, MagicMock

from train import (
    load_config,
    load_data,
    get_latest_version,
    get_metric_for_alias,
    champion_challenger_test,
)

# ─── Test: Config loading ──────────────────────────────────────────────────────
def test_load_config():
    config_data = {
        "data": {"test_size": 0.2, "random_state": 42},
        "model": {"hyperparams": {"n_estimators": 10}}
    }
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        f.flush()
        path = f.name

    loaded = load_config(path)
    assert loaded == config_data
    os.remove(path)

# ─── Test: Data splitting ──────────────────────────────────────────────────────
def test_load_data():
    df = pd.DataFrame({
        "feature_0": [1, 2, 3, 4],
        "feature_1": [1, 2, 3, 4],
        "feature_2": [1, 2, 3, 4],
        "feature_3": [1, 2, 3, 4],
        "y": [10, 20, 30, 40]
    })
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        f.flush()
        path = f.name

    cfg = {"data": {"test_size": 0.25, "random_state": 1}}
    X_train, X_test, y_train, y_test = load_data(path, cfg)

    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 1
    assert y_train.shape[0] == 3
    assert y_test.shape[0] == 1

    os.remove(path)

# ─── Test: Get latest model version ────────────────────────────────────────────
@patch("train.client")
def test_get_latest_version(mock_client):
    mock_client.search_model_versions.return_value = [
        MagicMock(version="1"), MagicMock(version="3"), MagicMock(version="2")
    ]
    version = get_latest_version("regression_model")
    assert version == "3"

# ─── Test: Get metric for alias ────────────────────────────────────────────────
@patch("train.client")
def test_get_metric_for_alias(mock_client):
    mock_run = MagicMock()
    mock_run.data.metrics = {"mse_custom": 0.123}
    
    version = MagicMock()
    version.aliases = ["champion"]
    version.run_id = "abc123"

    mock_client.search_model_versions.return_value = [version]
    mock_client.get_run.return_value = mock_run

    metric = get_metric_for_alias("regression_model", "champion", "mse_custom")
    assert metric == 0.123

# ─── Test: Champion vs Challenger ──────────────────────────────────────────────
@patch("train.get_metric_for_alias")
def test_champion_challenger_test(mock_get_metric):
    mock_get_metric.side_effect = [0.2, 0.1]  # champ, chall
    assert champion_challenger_test("regression_model", "mse_custom") == True

    mock_get_metric.side_effect = [0.1, 0.3]
    assert champion_challenger_test("regression_model", "mse_custom") == False
