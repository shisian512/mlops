# src/train.py
"""
End-to-end training, registration, and champion/challenger promotion
for a RandomForest regression model using MLflow.

This script handles the complete ML model lifecycle:
1. Loads and splits data into training and testing sets
2. Trains a RandomForestRegressor model with configurable hyperparameters
3. Evaluates model performance using mean squared error (MSE)
4. Logs the model and metrics to MLflow
5. Registers the model in MLflow Model Registry with a 'challenger' alias
6. Provides functionality to compare and promote models using a champion/challenger approach

Configuration is loaded from a params.yaml file, and environment variables
are used for AWS and MLflow connection settings.
"""

# Standard library imports
import os
import sys
import warnings

# Third-party imports
import mlflow
from mlflow import MlflowClient
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import yaml

warnings.simplefilter(action="ignore", category=FutureWarning)

# Configuration constants
PARAMS_FILE = "params.yaml"  # Default parameters file path
METRICS_PATH = "./metrics.json"  # Path for saving metrics
MODEL_PICKLE = "./models/model.pkl"  # Path for saving model pickle
FEATURE_COLS = ["feature_0", "feature_1", "feature_2", "feature_3"]  # Feature column names
TARGET_COL = "y"  # Target column name
METRIC_KEY = "mse"  # Key for tracking MSE in MLflow

# Load environment variables for AWS configuration
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")

# Validate required environment variables
if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not AWS_ENDPOINT_URL:
    raise ValueError("AWS credentials or endpoint not set in environment or .env")

# --- MLflow Helper Functions ---

def load_config(path: str) -> dict:
    """
    Loads YAML configuration file.
    
    Args:
        path (str): Path to the YAML configuration file
        
    Returns:
        dict: Parsed configuration as a dictionary
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_metric_for_alias(model_name: str, alias: str, metric_key: str) -> float:
    """
    Gets the metric value for a given model alias from MLflow.
    
    Args:
        model_name (str): Name of the registered model in MLflow
        alias (str): Model alias to retrieve metrics for (e.g., 'champion', 'challenger')
        metric_key (str): Name of the metric to retrieve
        
    Returns:
        float: The metric value or infinity if not found/error occurs
    """
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
    """
    Compares challenger and champion metrics to determine if challenger is better.
    
    For regression metrics like MSE, lower values are better.
    
    Args:
        model_name (str): Name of the registered model in MLflow
        metric_key (str): Name of the metric to compare
        
    Returns:
        bool: True if challenger performs better than champion, False otherwise
    """
    champ_val = get_metric_for_alias(model_name, CHAMPION_ALIAS, metric_key)
    chall_val = get_metric_for_alias(model_name, CHALLENGER_ALIAS, metric_key)
    print(f"Current Champion {metric_key}: {champ_val:.6f}")
    print(f"Current Challenger {metric_key}: {chall_val:.6f}")
    return chall_val < champ_val

def promote_alias(model_name: str, alias: str, version: str):
    """
    Promotes a model version to a specific alias in MLflow Model Registry.
    
    Args:
        model_name (str): Name of the registered model in MLflow
        alias (str): Alias to assign (e.g., 'champion', 'challenger')
        version (str): Model version to promote
    """
    client = MlflowClient()
    client.set_registered_model_alias(name=model_name, alias=alias, version=version)
    print(f"Alias '{alias}' → version {version}")

# --- Main Training Function ---

def run_training_job(data_path: str, params_file: str = PARAMS_FILE):
    """
    Main function to load data, train a model, and log with MLflow.
    
    This function performs the following steps:
    1. Load configuration from params.yaml
    2. Configure MLflow tracking
    3. Load and split the dataset
    4. Train a RandomForestRegressor model
    5. Evaluate model performance
    6. Log model and metrics to MLflow
    7. Register the model in MLflow Model Registry
    8. Assign the 'challenger' alias to the new model
    
    Args:
        data_path (str): Path to the CSV data file
        params_file (str): Path to the parameters YAML file
    """
    # Load configuration
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
    mlflow.sklearn.autolog()  # Enable automatic logging of sklearn models
    client = MlflowClient()
    
    MODEL_NAME = mlflow_cfg["model_name"]
    CHALLENGER_ALIAS = "challenger"
    CHAMPION_ALIAS = "champion"

    # Load data
    print("Loading data......")
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLS]  # Extract feature columns
    y = df[TARGET_COL]    # Extract target column
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_cfg["test_size"], random_state=data_cfg["random_state"]
    )
    
    # Train the model with hyperparameters from config
    print("Training......")
    model = RandomForestRegressor(**model_cfg)
    model.fit(X_train, y_train)
    
    # Evaluate model performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.6f}")

    # Log with MLflow and register the model
    with mlflow.start_run() as run:
        # Log the MSE metric
        mlflow.log_metric(METRIC_KEY, mse)
        
        # Log the trained model
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        
        # Register the model in MLflow Model Registry
        mv = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model", name=MODEL_NAME
        )
        
        # Assign the 'challenger' alias to the new model version
        client.set_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS, mv.version)
        print(f"New model registered as version {mv.version} and assigned '{CHALLENGER_ALIAS}' alias.")

        # Note: Champion/challenger promotion logic is commented out
        # This is better handled by a separate Airflow task to ensure
        # the training job finishes before checking and promoting
        # if champion_challenger_test(MODEL_NAME, METRIC_KEY):
        #     promote_alias(MODEL_NAME, CHAMPION_ALIAS, mv.version)

if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python train.py <path_to_data> [path_to_params.yaml]")
        sys.exit(1)
    
    data_path = sys.argv[1]  # Required: path to data file
    params_path = sys.argv[2] if len(sys.argv) > 2 else PARAMS_FILE  # Optional: path to params file
    
    # Execute the training job
    run_training_job(data_path, params_path)

