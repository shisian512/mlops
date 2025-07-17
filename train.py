"""
train.py

End-to-end training, registration, and champion/challenger promotion
for a RandomForest regression model using MLflow.

Steps:
  1. Load configuration parameters and data.
  2. Train a RandomForestRegressor.
  3. Compute and log a custom MSE metric.
  4. Persist the model locally (pickle) and in MLflow Model Registry.
  5. Tag the new version as 'challenger'.
  6. Compare challenger vs. existing champion by metric.
  7. Promote challenger to champion if performance improves.

Usage:
    python train.py
"""

import json
import yaml
import joblib
import warnings

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)

# ─── Configuration ─────────────────────────────────────────────────────────────

EXPERIMENT_NAME  = "regression_experiment"
MODEL_NAME       = "regression_model"
CHALLENGER_ALIAS = "challenger"
CHAMPION_ALIAS   = "champion"
PARAMS_FILE      = "params.yaml"
DATA_PATH        = "data/prepared.csv"
METRICS_PATH     = "metrics.json"
MODEL_PICKLE     = "model.pkl"
FEATURE_COLS     = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
TARGET_COL       = 'y'
METRIC_KEY       = "mse_custom"

# ─── Setup MLflow ───────────────────────────────────────────────────────────────

# Ensure the experiment exists (creates if missing)
mlflow.set_experiment(EXPERIMENT_NAME)
# Enable automatic logging of parameters, metrics, and models
mlflow.sklearn.autolog()
# Single shared client instance for registry operations
client = MlflowClient()

# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """
    Load full configuration from a YAML file.

    Expects structure:
      data:
        test_size: float
        random_state: int
      model:
        type: string
        hyperparams: dict
      validation:
        ...
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_data(path: str, cfg: dict):
    """
    Read CSV data, split into train/test using configuration parameters.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(path)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    ds_cfg = cfg['data']
    return train_test_split(
        X, y,
        test_size=ds_cfg['test_size'],
        random_state=ds_cfg['random_state']
    )


def get_latest_version(model_name: str) -> str:
    """
    Query the MLflow Model Registry for all versions
    of the given model, and return the highest version number.
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    latest = max(int(v.version) for v in versions)
    return str(latest)


def get_metric_for_alias(model_name: str, alias: str, metric_key: str) -> float:
    """
    Retrieve the most recent value of `metric_key` for the model
    version tagged with `alias`. If alias not found, returns +inf.
    """
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        history = client.get_metric_history(mv.run_id, metric_key)
        return history[-1].value if history else float("inf")
    except Exception:
        return float("inf")


def champion_challenger_test(model_name: str, metric_key: str) -> bool:
    """
    Compare challenger vs. champion metrics.
    Returns True if challenger metric is strictly lower.
    """
    champ_val = get_metric_for_alias(model_name, CHAMPION_ALIAS, metric_key)
    chall_val = get_metric_for_alias(model_name, CHALLENGER_ALIAS, metric_key)

    print(f"Current Champion {metric_key}: {champ_val:.6f}")
    print(f"Current Challenger {metric_key}: {chall_val:.6f}")

    return chall_val < champ_val


def promote_alias(model_name: str, alias: str, version: str):
    """
    Point the given alias to a specific model version in the registry.
    """
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version
    )
    print(f"Alias '{alias}' → version {version}")

# ─── Main Training & Registration ──────────────────────────────────────────────

def main():
    # 1. Load configuration and data
    cfg = load_config(PARAMS_FILE)
    X_train, X_test, y_train, y_test = load_data(DATA_PATH, cfg)

    # 2. Prepare model hyperparameters
    model_cfg = cfg['model']['hyperparams']
    # Merge validation settings into hyperparameters if needed
    model_cfg.update(cfg.get('validation', {}))

    # 3. Train & log in MLflow
    with mlflow.start_run() as run:
        # Instantiate and train the model
        model = RandomForestRegressor(**model_cfg)
        model.fit(X_train, y_train)

        # Persist a local copy for DVC or other uses
        joblib.dump(model, MODEL_PICKLE)

        # Compute custom MSE metric on hold-out set
        mse = mean_squared_error(y_test, model.predict(X_test))
        mlflow.log_metric(METRIC_KEY, mse)

        # Also write metrics to JSON (for DVC or CI pipelines)
        with open(METRICS_PATH, 'w') as f:
            json.dump({"mse": mse}, f)

        # Register model in MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            registered_model_name=MODEL_NAME,
            input_example=X_train.iloc[:5]
        )

    # 4. Alias management outside of the run
    latest_v = get_latest_version(MODEL_NAME)

    # Tag the new version as 'challenger'
    promote_alias(MODEL_NAME, CHALLENGER_ALIAS, latest_v)

    # 5. Perform champion/challenger comparison & promote if needed
    if champion_challenger_test(MODEL_NAME, METRIC_KEY):
        promote_alias(MODEL_NAME, CHAMPION_ALIAS, latest_v)
    else:
        print("No promotion: existing champion remains.")


if __name__ == "__main__":
    main()
