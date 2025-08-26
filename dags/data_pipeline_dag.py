# dags/data_pipeline_dag.py
from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.bash import BashOperator
from datetime import datetime

# Define your S3 bucket and folder names
S3_BUCKET = "your_s3_bucket_name"
S3_RAW_FOLDER = "your_data_folder"
S3_PROCESSED_FOLDER = "processed_data"
S3_CLEANED_FOLDER = "cleaned_data"

# Define your DynamoDB table name for the feature store
DYNAMODB_TABLE_NAME = "your_feature_store_table"

# Define the S3 location where you have uploaded your Spark scripts
# For example: s3://your_s3_bucket_name/scripts/
S3_SPARK_SCRIPTS_PATH = f"s3://{S3_BUCKET}/scripts"

with DAG(
    dag_id="s3_etl_pipeline_complete",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule_interval="@daily",
    catchup=False,
    tags=["s3", "pyspark", "data_ingestion", "dvc", "feature_store"],
) as dag:

    # 1. Data Ingestion Task
    s3_ingestion_task = SparkSubmitOperator(
        task_id="s3_ingestion",
        # Use the S3 path for the Spark application script
        application=f"{S3_SPARK_SCRIPTS_PATH}/s3_ingestion.py",
        conn_id="spark_default",
        application_args=[S3_BUCKET, S3_RAW_FOLDER],
    )

    # 2. Data Preprocessing Task
    s3_preprocessing_task = SparkSubmitOperator(
        task_id="s3_preprocessing",
        # Use the S3 path for the Spark application script
        application=f"{S3_SPARK_SCRIPTS_PATH}/s3_preprocess.py",
        conn_id="spark_default",
        application_args=[S3_BUCKET, S3_PROCESSED_FOLDER, S3_CLEANED_FOLDER],
    )

    # 3. Data Validation Task
    s3_validation_task = SparkSubmitOperator(
        task_id="s3_data_validation",
        # Use the S3 path for the Spark application script
        application=f"{S3_SPARK_SCRIPTS_PATH}/data_validation.py",
        conn_id="spark_default",
        application_args=[S3_BUCKET, S3_CLEANED_FOLDER],
    )

    # 4. DVC Tracking Task (BashOperator)
    # This command now syncs data from S3, runs DVC, and pushes the changes back.
    # NOTE: DVC must be installed on the Airflow worker executing this task.
    dvc_track_task = BashOperator(
        task_id="dvc_track_data",
        bash_command=(
            f"mkdir -p /tmp/dvc_data && "
            f"aws s3 sync s3://{S3_BUCKET}/{S3_CLEANED_FOLDER}/ /tmp/dvc_data/cleaned_data && "
            f"cd /tmp/dvc_data && "
            f"dvc add cleaned_data && dvc push && "
            f"aws s3 sync .dvc s3://{S3_BUCKET}/dvc_metadata/"
        ),
    )

    # 5. Feature Store Loading Task
    s3_feature_store_task = SparkSubmitOperator(
        task_id="load_feature_store",
        # Use the S3 path for the Spark application script
        application=f"{S3_SPARK_SCRIPTS_PATH}/feature_store_load.py",
        conn_id="spark_default",
        application_args=[S3_BUCKET, S3_CLEANED_FOLDER, DYNAMODB_TABLE_NAME],
    )

    # Define the task dependencies
    (
        s3_ingestion_task
        >> s3_preprocessing_task
        >> s3_validation_task
        >> dvc_track_task
        >> s3_feature_store_task
    )
