# src/train.py
"""
End-to-end training, registration, and champion/challenger promotion
for a RandomForest regression model using MLflow.
"""

from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml
import warnings
from dotenv import load_dotenv
import os
import mlflow
import sys

warnings.simplefilter(action="ignore", category=FutureWarning)

# We will now get all configurations from the params.yaml file
PARAMS_FILE = "params.yaml"
# The data path will be passed as a command-line argument
METRICS_PATH = "./metrics.json"
MODEL_PICKLE = "./models/model.pkl"
FEATURE_COLS = ["feature_0", "feature_1", "feature_2", "feature_3"]
TARGET_COL = "y"
METRIC_KEY = "mse"  # Renamed for clarity in MLflow

load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not AWS_ENDPOINT_URL:
    raise ValueError("AWS credentials or endpoint not set in environment or .env")

# --- MLflow Helper Functions ---


def load_config(path: str) -> dict:
    """Loads YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_metric_for_alias(model_name: str, alias: str, metric_key: str) -> float:
    """Gets the metric value for a given model alias."""
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            if alias in v.aliases:
                run_id = v.run_id
                metrics = client.get_run(run_id).data.metrics
                return metrics.get(metric_key, float("inf"))
        return float("inf")
    except Exception as e:
        print(f"Could not get metric for alias '{alias}': {e}")
        return float("inf")


def champion_challenger_test(model_name: str, metric_key: str) -> bool:
    """Compares challenger and champion metrics."""
    champ_val = get_metric_for_alias(model_name, CHAMPION_ALIAS, metric_key)
    chall_val = get_metric_for_alias(model_name, CHALLENGER_ALIAS, metric_key)
    print(f"Current Champion {metric_key}: {champ_val:.6f}")
    print(f"Current Challenger {metric_key}: {chall_val:.6f}")
    return chall_val < champ_val


def promote_alias(model_name: str, alias: str, version: str):
    """Promotes a model version to a specific alias."""
    client = MlflowClient()
    client.set_registered_model_alias(name=model_name, alias=alias, version=version)
    print(f"Alias '{alias}' → version {version}")


# --- Main Training Function ---


def run_training_job(data_path: str, params_file: str = PARAMS_FILE):
    """
    Main function to load data, train a model, and log with MLflow.
    """
    cfg = load_config(params_file)
    mlflow_cfg = cfg["mlflow"]
    print("mlflow_cfg", mlflow_cfg)
    model_cfg = cfg["model"]["hyperparams"]
    print("model_cfg", model_cfg)
    data_cfg = cfg["data"]
    print("data_cfg", data_cfg)

    # MLflow configuration
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    print("tracking_uri", mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    print("experiment", mlflow_cfg["experiment_name"])
    mlflow.sklearn.autolog()
    client = MlflowClient()

    MODEL_NAME = mlflow_cfg["model_name"]
    CHALLENGER_ALIAS = "challenger"
    CHAMPION_ALIAS = "champion"

    # Load data
    print("Loading data......")
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_cfg["test_size"], random_state=data_cfg["random_state"]
    )

    # Train the model
    print("Training......")
    model = RandomForestRegressor(**model_cfg)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.6f}")

    # Log with MLflow and register the model
    with mlflow.start_run() as run:
        mlflow.log_metric(METRIC_KEY, mse)
        # mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        # Register the model in MLflow Model Registry
        mv = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model", name=MODEL_NAME
        )

        # Assign the 'challenger' alias to the new model version
        client.set_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS, mv.version)
        print(
            f"New model registered as version {mv.version} and assigned '{CHALLENGER_ALIAS}' alias."
        )

        # This part of the logic is better handled by a separate Airflow task
        # to ensure the training job finishes before checking and promoting.
        # if champion_challenger_test(MODEL_NAME, METRIC_KEY):
        #     promote_alias(MODEL_NAME, CHAMPION_ALIAS, mv.version)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <path_to_data> [path_to_params.yaml]")
        sys.exit(1)

    data_path = sys.argv[1]
    params_path = sys.argv[2] if len(sys.argv) > 2 else PARAMS_FILE
    run_training_job(data_path, params_path)
