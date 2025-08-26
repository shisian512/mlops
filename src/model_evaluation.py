# src/model_evaluation.py
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
    """Loads YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_model_evaluation(
    model_name: str, test_data_path: str, run_id: str, params_file: str
):
    """
    Loads a model and test data to perform evaluation and explainability analysis,
    logging results back to a new MLflow run.
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
        feature_cols = cfg["feature_cols"]
        target_col = cfg["target_col"]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        print(f"Successfully loaded test data from {test_data_path}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        sys.exit(1)

    # 3. Perform evaluation and log metrics
    with mlflow.start_run(
        run_id=run_id,
        experiment_id=client.get_experiment_by_name(
            mlflow_cfg["experiment_name"]
        ).experiment_id,
    ):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Calculated MSE: {mse}")
        mlflow.log_metric("evaluation_mse", mse)

        # 4. Generate SHAP explainability report
        try:
            print("Generating SHAP explainability report...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            # Create a summary plot and save it as an image
            shap_output_path = "shap_summary_plot.png"
            shap.summary_plot(shap_values, X_test, show=False)
            plt.savefig(shap_output_path)
            plt.close()
            mlflow.log_artifact(shap_output_path)
            print("SHAP summary plot logged as artifact.")

            # You can also log a dictionary of feature importance
            feature_importance = (
                pd.DataFrame(
                    {
                        "feature": X_test.columns,
                        "importance": explainer.feature_importances_,
                    }
                )
                .sort_values("importance", ascending=False)
                .to_dict("records")
            )

            with open("feature_importance.json", "w") as f:
                json.dump(feature_importance, f, indent=4)
            mlflow.log_artifact("feature_importance.json")
            print("Feature importance JSON logged as artifact.")

        except Exception as e:
            print(f"Error generating or logging SHAP report: {e}")

    print("Model evaluation complete.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # Import here to avoid issues with non-GUI backends

    if len(sys.argv) != 5:
        print(
            "Usage: python model_evaluation.py <model_name> <test_data_path> <run_id> <params_file>"
        )
        sys.exit(1)

    run_model_evaluation(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
