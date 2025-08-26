# src/data_validation.py
import os
import sys
from pyspark.sql import SparkSession
from dotenv import load_dotenv


def run_data_validation(s3_bucket, source_folder):
    """
    Performs data validation on the preprocessed data.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get the credentials from the environment variables
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    # Define configurations for S3 access
    spark = (
        SparkSession.builder.appName("DataValidation")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.609",
        )
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key)
        .getOrCreate()
    )

    # Read the data to validate
    df = spark.read.parquet(f"s3a://{s3_bucket}/{source_folder}/")

    print(f"Starting data validation on {s3_bucket}/{source_folder}...")

    # --- Validation Checks ---
    # Check 1: Ensure there are no nulls after the preprocessing step
    # You should replace "your_key_column_name" with the actual column you need to validate
    key_column = "your_key_column_name"

    # Check if the key column exists
    if key_column not in df.columns:
        print(f"Error: Key column '{key_column}' not found.")
        spark.stop()
        sys.exit(1)

    null_count = df.filter(df[key_column].isNull()).count()
    if null_count > 0:
        print(
            f"Validation failed: Found {null_count} null values in column '{key_column}'."
        )
        spark.stop()
        sys.exit(1)

    print("Validation Check 1 Passed: No null values in key column.")

    # Check 2: Ensure a minimum number of rows
    min_rows = 1000  # Adjust as needed
    row_count = df.count()
    if row_count < min_rows:
        print(
            f"Validation failed: Row count {row_count} is less than the minimum required {min_rows}."
        )
        spark.stop()
        sys.exit(1)

    print(f"Validation Check 2 Passed: Row count is {row_count}.")

    print("Data validation successful! Pipeline is healthy.")

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: spark-submit s3_data_validation.py <s3_bucket> <source_folder>")
        sys.exit(1)
    run_data_validation(sys.argv[1], sys.argv[2])
