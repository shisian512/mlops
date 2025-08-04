"""
Data preparation script for filling missing values and exporting a cleaned dataset for model training.
"""
import pandas as pd
import yaml
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

CONFIG_FILE = "data_prep.yaml"  # Path to YAML config file
RAW_DATA_PATH = "data/train.csv"
CLEAN_DATA_PATH = "data/prepared.csv"

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df

def save_data(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def main():
    try:
        cfg = load_config(CONFIG_FILE)
        raw_path = cfg.get('raw_data', RAW_DATA_PATH)
        clean_path = cfg.get('clean_data', CLEAN_DATA_PATH)
    except FileNotFoundError:
        raw_path, clean_path = RAW_DATA_PATH, CLEAN_DATA_PATH
    df = load_data(raw_path)
    print(f"Loaded raw data with shape {df.shape}")
    df_clean = impute_missing(df)
    print(f"After imputation, any NaNs left? {df_clean.isnull().any().any()}")
    save_data(df_clean, clean_path)
    print(f"Clean data saved to '{clean_path}' with shape {df_clean.shape}")

if __name__ == "__main__":
    main()
