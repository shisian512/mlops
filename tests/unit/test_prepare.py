# tests/test_prepare.py

import pandas as pd
import pytest
import tempfile
import yaml
import os
from prepare import (
    load_config,
    load_data,
    impute_missing,
    save_data,
)

# ─── Test: load_config loads YAML correctly ────────────────────────────────────
def test_load_config():
    mock_config = {
        "raw_data": "data/raw.csv",
        "clean_data": "data/clean.csv"
    }
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as f:
        yaml.dump(mock_config, f)
        f.flush()
        path = f.name
    
    loaded = load_config(path)
    assert loaded == mock_config

    os.remove(path)

# ─── Test: load_data reads CSV correctly ───────────────────────────────────────
def test_load_data():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name

    loaded_df = load_data(path)
    pd.testing.assert_frame_equal(df, loaded_df)

    os.remove(path)

# ─── Test: impute_missing fills NaNs correctly ─────────────────────────────────
def test_impute_missing():
    df = pd.DataFrame({
        "x": [1, None, 3],
        "y": [4, 5, None]
    })
    imputed_df = impute_missing(df.copy())
    assert not imputed_df.isnull().values.any()
    assert imputed_df.shape == df.shape

# ─── Test: save_data writes correct content ────────────────────────────────────
def test_save_data():
    df = pd.DataFrame({
        "col1": [1, 2, 3]
    })
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        path = f.name

    save_data(df, path)
    loaded = pd.read_csv(path)
    pd.testing.assert_frame_equal(df, loaded)

    os.remove(path)
