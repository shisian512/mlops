# dags/training_dag.py
from __future__ import annotations

import pendulum
import os
import yaml
from datetime import datetime
import json

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator
from mlflow import MlflowClient

# DVC configuration
DVC_PROJECT_ROOT = "/opt/airflow/dvc_project"
S3_CLEANED_DATA_PATH = "cleaned_data"
DVC_DATA_VERSION = "v1.0"

# AWS and MLflow configuration
S3_BUCKET = "your-sagemaker-bucket"  # Bucket SageMaker can access
SAGEMAKER_ROLE_ARN = (
    "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"
)
SAGEMAKER_INSTANCE_TYPE = "ml.m5.large"
S3_TEST_DATA_PATH = f"s3://{S3_BUCKET}/sagemaker/test-data/{DVC_DATA_VERSION}/test.csv"
S3_TRAINING_DATA_PATH = f"s3://{S3_BUCKET}/sagemaker/training-data/{DVC_DATA_VERSION}/"
PARAMS_FILE_PATH = f"{DVC_PROJECT_ROOT}/params.yaml"

# Load MLflow config from params.yaml
with open(PARAMS_FILE_PATH, "r") as f:
    params_config = yaml.safe_load(f)
    MLFLOW_CFG = params_config["mlflow"]
    MODEL_NAME = MLFLOW_CFG["model_name"]
    MLFLOW_TRACKING_URI = MLFLOW_CFG["tracking_uri"]
    METRIC_KEY = "evaluation_mse"


# --- Helper functions for model promotion and deployment ---
def get_metric_for_alias(model_name: str, alias: str, metric_key: str) -> float:
    """Gets the metric value for a given model alias."""
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
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


def promote_challenger_if_better(model_name: str, metric_key: str):
    """
    Compares the challenger model's metric with the champion's and promotes
    the challenger to the 'Staging' alias if its performance is better.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    CHALLENGER_ALIAS = "challenger"
    CHAMPION_ALIAS = "champion"
    STAGING_ALIAS = "staging"

    chall_val = get_metric_for_alias(model_name, CHALLENGER_ALIAS, metric_key)
    champ_val = get_metric_for_alias(model_name, CHAMPION_ALIAS, metric_key)

    print(f"Current Champion {metric_key}: {champ_val:.6f}")
    print(f"Current Challenger {metric_key}: {chall_val:.6f}")

    if chall_val < champ_val:
        print(
            "Challenger performance is better than the champion. Promoting challenger to Staging."
        )

        # Get the version of the challenger model
        challenger_version = client.get_model_version_by_alias(
            model_name, CHALLENGER_ALIAS
        ).version

        # Remove old staging alias if it exists
        try:
            old_staging_version = client.get_model_version_by_alias(
                model_name, STAGING_ALIAS
            ).version
            client.delete_registered_model_alias(
                name=model_name, alias=STAGING_ALIAS, version=old_staging_version
            )
        except Exception:
            pass  # No old staging model exists

        # Promote the challenger to the 'Staging' alias
        client.set_registered_model_alias(
            name=model_name, alias=STAGING_ALIAS, version=challenger_version
        )
        print(f"Model version {challenger_version} promoted to '{STAGING_ALIAS}'.")
    else:
        print(
            "Challenger performance is not better than the champion. No promotion to Staging."
        )


with DAG(
    dag_id="sagemaker_training_pipeline",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule_interval=None,
    catchup=False,
    tags=["sagemaker", "dvc", "training", "mlflow", "cd"],
) as dag:

    # 1. DVC Data Preparation Task
    dvc_data_prep_task = BashOperator(
        task_id="dvc_data_preparation",
        bash_command=(
            f"cd {DVC_PROJECT_ROOT} && "
            f"git checkout {DVC_DATA_VERSION} && "
            f"dvc pull {S3_CLEANED_DATA_PATH}"
        ),
    )

    # 2. Upload Data to S3 Task
    upload_data_to_s3_task = BashOperator(
        task_id="upload_data_to_s3",
        bash_command=(
            f"aws s3 sync {DVC_PROJECT_ROOT}/{S3_CLEANED_DATA_PATH}/ "
            f"s3://{S3_BUCKET}/sagemaker/training-data/{DVC_DATA_VERSION}/ && "
            f"aws s3 cp {DVC_PROJECT_ROOT}/{S3_CLEANED_DATA_PATH}/test.csv "
            f"{S3_TEST_DATA_PATH}"
        ),
    )

    # 3. SageMaker Model Training Task
    sagemaker_training_task = SageMakerTrainingOperator(
        task_id="sagemaker_model_training",
        config={
            "TrainingJobName": f"my-ml-model-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "AlgorithmSpecification": {
                "TrainingImage": "257758044811.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3",
                "TrainingInputMode": "File",
            },
            "RoleArn": SAGEMAKER_ROLE_ARN,
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": S3_TRAINING_DATA_PATH,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                }
            ],
            "OutputDataConfig": {
                "S3OutputPath": f"s3://{S3_BUCKET}/sagemaker/model-artifacts/{DVC_DATA_VERSION}/"
            },
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": SAGEMAKER_INSTANCE_TYPE,
                "VolumeSizeInGB": 10,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
            "HyperParameters": {
                "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
                "mlflow_experiment_name": MLFLOW_CFG["experiment_name"],
                "model_name": MODEL_NAME,
            },
            "VpcConfig": {
                "SecurityGroupIds": [],
                "Subnets": [],
            },
            "Code": {
                "S3Source": f"s3://{S3_BUCKET}/sagemaker/scripts/train.tar.gz",
                "TrainingInputMode": "File",
            },
        },
    )

    # 4. Post-Training Evaluation and Explainability Task
    # This task now runs the dedicated evaluation script.
    model_evaluation_task = PythonOperator(
        task_id="model_evaluation",
        python_callable=run_model_evaluation,
        op_kwargs={
            "model_name": MODEL_NAME,
            "test_data_path": S3_TEST_DATA_PATH,
            "run_id": sagemaker_training_task.output,
            "params_file": PARAMS_FILE_PATH,
        },
    )

    # 5. Model Promotion Task
    # This Python task compares the challenger with the champion and promotes
    # the challenger to the 'staging' alias if it's better.
    promote_model_task = PythonOperator(
        task_id="promote_model_to_staging",
        python_callable=promote_challenger_if_better,
        op_kwargs={
            "model_name": MODEL_NAME,
            "metric_key": METRIC_KEY,
        },
    )

    # 6. GitOps Deployment Trigger Task
    # This is a placeholder that would trigger a CD pipeline
    # For example, by committing and pushing an updated YAML file
    deployment_trigger_task = BashOperator(
        task_id="trigger_cd_pipeline",
        bash_command=(
            # Placeholder command:
            f"echo 'New model version is ready for deployment.' && "
            f"echo 'Updating deployment manifest...' && "
            f"echo 'Committing and pushing changes to Git repository...' && "
            f"git add deployment.yaml && "
            f'git commit -m \'Deploying new model from run {{ ti.xcom_pull(task_ids="sagemaker_model_training", key="return_value") }}\' && '
            f"git push"
            # NOTE: Your git repository must be configured with a remote, and the Airflow
            # worker must have the necessary credentials to push.
        ),
    )

    # Define the task dependencies
    # The evaluation task must run after training. The promotion task runs after evaluation.
    # The deployment trigger runs only if the promotion task succeeds.
    [dvc_data_prep_task, upload_data_to_s3_task] >> sagemaker_training_task
    sagemaker_training_task >> model_evaluation_task
    model_evaluation_task >> promote_model_task
    promote_model_task >> deployment_trigger_task
