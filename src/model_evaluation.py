# src/model_evaluation.py
"""
Model evaluation and explainability module for ML models.

This script performs comprehensive evaluation of trained ML models by:
1. Loading a model from MLflow using its run ID
2. Loading test data from a specified path
3. Calculating performance metrics (MSE)
4. Generating SHAP explainability reports
5. Logging all results back to MLflow for tracking and comparison

The evaluation results include both numerical metrics and visual explanations
of feature importance to support model interpretability requirements.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import shap
import json
import sys
import os
import yaml
from sklearn.metrics import mean_squared_error
from mlflow import MlflowClient

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

def run_model_evaluation(model_name: str, test_data_path: str, run_id: str, params_file: str):
    """
    Loads a model and test data to perform evaluation and explainability analysis,
    logging results back to a new MLflow run.
    
    This function performs the following steps:
    1. Loads configuration from the specified params file
    2. Sets up MLflow tracking
    3. Loads the model from MLflow using the provided run ID
    4. Loads and prepares test data
    5. Calculates mean squared error (MSE) on test data
    6. Generates SHAP explainability reports (summary plot and feature importance)
    7. Logs all metrics and artifacts to MLflow
    
    Args:
        model_name (str): Name of the model to evaluate
        test_data_path (str): Path to the CSV file containing test data
        run_id (str): MLflow run ID where the model is stored
        params_file (str): Path to the parameters YAML file
        
    Raises:
        SystemExit: If model loading or data loading fails
    """
    # Load configuration
    cfg = load_config(params_file)
    mlflow_cfg = cfg["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    client = MlflowClient()

    print(f"Starting evaluation for model {model_name} from run ID {run_id}...")
    
    # 1. Load the model from MLflow
    try:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Successfully loaded model from {model_uri}")
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        sys.exit(1)

    # 2. Load the test data
    try:
        test_df = pd.read_csv(test_data_path)
        feature_cols = cfg["feature_cols"]  # Extract feature column names from config
        target_col = cfg["target_col"]      # Extract target column name from config
        X_test = test_df[feature_cols]     # Features for testing
        y_test = test_df[target_col]       # Target values for testing
        print(f"Successfully loaded test data from {test_data_path}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        sys.exit(1)

    # 3. Perform evaluation and log metrics
    # Use the same run ID to append evaluation results to the original training run
    with mlflow.start_run(run_id=run_id, experiment_id=client.get_experiment_by_name(mlflow_cfg["experiment_name"]).experiment_id):
        # Generate predictions and calculate MSE
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Calculated MSE: {mse}")
        mlflow.log_metric("evaluation_mse", mse)  # Log MSE as a metric in MLflow

        # 4. Generate SHAP explainability report
        try:
            print("Generating SHAP explainability report...")
            # Create a SHAP TreeExplainer for the model
            explainer = shap.TreeExplainer(model)
            # Calculate SHAP values for test data
            shap_values = explainer.shap_values(X_test)
            
            # Create a summary plot and save it as an image
            shap_output_path = "shap_summary_plot.png"
            shap.summary_plot(shap_values, X_test, show=False)
            plt.savefig(shap_output_path)
            plt.close()
            mlflow.log_artifact(shap_output_path)  # Log the plot as an artifact
            print("SHAP summary plot logged as artifact.")

            # Extract and log feature importance as a JSON file
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': explainer.feature_importances_
            }).sort_values('importance', ascending=False).to_dict('records')
            
            # Save and log feature importance as a JSON file
            with open("feature_importance.json", "w") as f:
                json.dump(feature_importance, f, indent=4)
            mlflow.log_artifact("feature_importance.json")
            print("Feature importance JSON logged as artifact.")

        except Exception as e:
            print(f"Error generating or logging SHAP report: {e}")

    print("Model evaluation complete.")

if __name__ == "__main__":
    # Import matplotlib here to avoid issues with non-GUI backends
    import matplotlib.pyplot as plt 
    
    # Validate command-line arguments
    if len(sys.argv) != 5:
        print("Usage: python model_evaluation.py <model_name> <test_data_path> <run_id> <params_file>")
        sys.exit(1)
    
    # Execute the model evaluation with command-line arguments
    run_model_evaluation(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
