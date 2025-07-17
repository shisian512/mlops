"""
data_prep.py

Simple data preparation script for filling missing values
and exporting a cleaned dataset for model training.

Steps:
  1. Load raw CSV data.
  2. Impute missing values with column means.
  3. Save the cleaned dataset to a new CSV file.

Usage:
    python data_prep.py
"""

import pandas as pd
import yaml
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ─── Configuration ─────────────────────────────────────────────────────────────
CONFIG_FILE = "data_prep.yaml"  # Path to YAML config file
RAW_DATA_PATH = "data/train.csv"
CLEAN_DATA_PATH = "data/prepared.csv"

# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """
    Load configuration parameters from a YAML file.

    Expects keys:
      raw_data: str
      clean_data: str
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    """
    Read raw data from a CSV file into a DataFrame.

    Args:
        path: Path to raw CSV file.

    Returns:
        Raw pandas DataFrame.
    """
    return pd.read_csv(path)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in numeric columns with the column mean.

    Args:
        df: Input DataFrame with potential NaNs.

    Returns:
        DataFrame with NaNs replaced by column means.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df


def save_data(df: pd.DataFrame, path: str):
    """
    Save the DataFrame to a CSV file without the index.

    Args:
        df: DataFrame to save.
        path: Destination CSV file path.
    """
    df.to_csv(path, index=False)


# ─── Main Execution ────────────────────────────────────────────────────────────

def main():
    # 1. (Optional) Load config to override default paths
    try:
        cfg = load_config(CONFIG_FILE)
        raw_path = cfg.get('raw_data', RAW_DATA_PATH)
        clean_path = cfg.get('clean_data', CLEAN_DATA_PATH)
    except FileNotFoundError:
        raw_path, clean_path = RAW_DATA_PATH, CLEAN_DATA_PATH

    # 2. Load raw data
    df = load_data(raw_path)
    print(f"Loaded raw data with shape {df.shape}")

    # 3. Impute missing values
    df_clean = impute_missing(df)
    print(f"After imputation, any NaNs left? {df_clean.isnull().any().any()}")

    # 4. Save cleaned data
    save_data(df_clean, clean_path)
    print(f"Clean data saved to '{clean_path}' with shape {df_clean.shape}")


if __name__ == "__main__":
    main()
