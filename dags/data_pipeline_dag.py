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

# Define the location of your Spark scripts
SPARK_SCRIPTS_PATH = "./src"

with DAG(
    dag_id="s3_etl_pipeline_complete",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule_interval='@daily',
    catchup=False,
    tags=["s3", "pyspark", "data_ingestion", "dvc", "feature_store"],
) as dag:
    
    # 1. Data Ingestion Task
    s3_ingestion_task = SparkSubmitOperator(
        task_id="s3_ingestion",
        application=f"{SPARK_SCRIPTS_PATH}/data_ingestion.py",
        conn_id="spark_default",
        application_args=[S3_BUCKET, S3_RAW_FOLDER],
    )

    # 2. Data Preprocessing Task
    s3_preprocessing_task = SparkSubmitOperator(
        task_id="s3_preprocessing",
        application=f"{SPARK_SCRIPTS_PATH}/data_preprocess.py",
        conn_id="spark_default",
        application_args=[S3_BUCKET, S3_PROCESSED_FOLDER, S3_CLEANED_FOLDER],
    )

    # 3. Data Validation Task
    s3_validation_task = SparkSubmitOperator(
        task_id="s3_data_validation",
        application=f"{SPARK_SCRIPTS_PATH}/data_validation.py",
        conn_id="spark_default",
        application_args=[S3_BUCKET, S3_CLEANED_FOLDER],
    )

    # 4. DVC Tracking Task (BashOperator)
    # NOTE: You must have DVC configured in your project directory
    #       (dvc init, dvc remote add) and the folder containing
    #       the cleaned data must be accessible to DVC.
    dvc_track_task = BashOperator(
        task_id="dvc_track_data",
        bash_command=f"cd {SPARK_SCRIPTS_PATH} && dvc add ../{S3_CLEANED_FOLDER} && dvc push",
    )

    # 5. Feature Store Loading Task
    s3_feature_store_task = SparkSubmitOperator(
        task_id="load_feature_store",
        application=f"{SPARK_SCRIPTS_PATH}/feature_store_load.py",
        conn_id="spark_default",
        application_args=[S3_BUCKET, S3_CLEANED_FOLDER, DYNAMODB_TABLE_NAME],
    )

    # Define the task dependencies
    # The pipeline is sequential: ingest -> preprocess -> validate -> track -> load
    s3_ingestion_task >> s3_preprocessing_task >> s3_validation_task >> dvc_track_task >> s3_feature_store_task
