# src/data_validation.py
"""
Data Validation Module

This module provides functionality to validate data quality in the ML pipeline.
It performs various validation checks on preprocessed data stored in S3 to ensure
data integrity before proceeding with model training or inference.

Validation checks include:
1. Ensuring no null values in key columns
2. Verifying minimum row count requirements

The module uses PySpark for distributed data processing and validation.
"""

import os
import sys
from pyspark.sql import SparkSession
from dotenv import load_dotenv

def run_data_validation(s3_bucket, source_folder):
    """
    Performs data validation on the preprocessed data stored in S3.
    
    This function initializes a Spark session with AWS credentials,
    loads the data from S3, and runs a series of validation checks.
    If any validation check fails, the process exits with an error code.
    
    Args:
        s3_bucket (str): The S3 bucket name containing the data
        source_folder (str): The folder path within the S3 bucket
        
    Returns:
        None
        
    Raises:
        SystemExit: If any validation check fails or if required columns are missing
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the credentials from the environment variables
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if not aws_access_key or not aws_secret_key:
        print("Error: AWS credentials not found in environment variables.")
        sys.exit(1)

    # Define configurations for S3 access
    # Using Hadoop AWS and AWS Java SDK for S3 connectivity
    spark = SparkSession.builder \
        .appName("DataValidation") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.609") \
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key) \
        .getOrCreate()
    
    # Read the data to validate from S3 in Parquet format
    s3_path = f"s3a://{s3_bucket}/{source_folder}/"
    print(f"Starting data validation on {s3_path}...")
    
    try:
        df = spark.read.parquet(s3_path)
    except Exception as e:
        print(f"Error reading data from {s3_path}: {str(e)}")
        spark.stop()
        sys.exit(1)

    # --- Validation Checks ---
    # Check 1: Ensure there are no nulls after the preprocessing step
    # You should replace "your_key_column_name" with the actual column you need to validate
    key_column = "your_key_column_name" 
    
    # Check if the key column exists
    if key_column not in df.columns:
        print(f"Error: Key column '{key_column}' not found.")
        spark.stop()
        sys.exit(1)

    # Count null values in the key column
    null_count = df.filter(df[key_column].isNull()).count()
    if null_count > 0:
        print(f"Validation failed: Found {null_count} null values in column '{key_column}'.")
        spark.stop()
        sys.exit(1)
    
    print("Validation Check 1 Passed: No null values in key column.")

    # Check 2: Ensure a minimum number of rows
    min_rows = 1000  # Adjust as needed based on your data requirements
    row_count = df.count()
    if row_count < min_rows:
        print(f"Validation failed: Row count {row_count} is less than the minimum required {min_rows}.")
        spark.stop()
        sys.exit(1)

    print(f"Validation Check 2 Passed: Row count is {row_count}.")
    
    # All validation checks passed
    print("Data validation successful! Pipeline is healthy.")
    
    # Clean up Spark session
    spark.stop()

if __name__ == "__main__":
    # Validate command line arguments
    if len(sys.argv) != 3:
        print("Usage: spark-submit data_validation.py <s3_bucket> <source_folder>")
        print("  s3_bucket: The S3 bucket name containing the data")
        print("  source_folder: The folder path within the S3 bucket")
        sys.exit(1)
    
    # Execute data validation with command line arguments
    run_data_validation(sys.argv[1], sys.argv[2])
