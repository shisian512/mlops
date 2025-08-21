# src/data_ingestion.py
"""
Data ingestion module for the MLOps pipeline.

This module handles the extraction of data from S3 storage, which is the first step
in the ML pipeline. It reads CSV data from a specified S3 bucket and folder,
and writes it as Parquet format to a processed data location.
"""

# Standard library imports
import os

# Third-party imports
from dotenv import load_dotenv
from pyspark.sql import SparkSession

def run_s3_ingestion(s3_bucket, s3_folder):
    """
    Extract data from an S3 bucket and convert it to Parquet format.
    
    This function performs the following steps:
    1. Sets up a Spark session with AWS S3 access configurations
    2. Reads CSV data from the specified S3 location
    3. Writes the data in Parquet format to a processed data folder
    
    Args:
        s3_bucket (str): Name of the S3 bucket containing the data
        s3_folder (str): Folder path within the bucket where CSV files are stored
        
    Returns:
        None: The function writes data to S3 but doesn't return any values
    """
    # Load environment variables from .env file
    # This ensures secure handling of AWS credentials
    load_dotenv()
    
    # Get the credentials from the environment variables
    # Never hardcode credentials in the code for security reasons
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    # Define configurations for S3 access
    # Create a Spark session with necessary AWS S3 connector dependencies
    spark = SparkSession.builder \
        .appName("S3Ingestion") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.609") \
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key) \
        .getOrCreate()
    
    # Read data from a folder in an S3 bucket
    # The header=True parameter indicates the first row contains column names
    # inferSchema=True automatically detects data types for each column
    df = spark.read.csv(f"s3a://{s3_bucket}/{s3_folder}/", header=True, inferSchema=True)
    
    # Write the DataFrame to S3 in Parquet format
    # Parquet is a columnar storage format that is more efficient for analytics
    # mode="overwrite" will replace any existing data in the destination
    df.write.mode("overwrite").parquet(f"s3a://{s3_bucket}/processed_data/")

    # Stop the Spark session to release resources
    spark.stop()

# This part is for running the script directly if needed, but not required by Airflow
if __name__ == "__main__":
    # You might get these from command-line arguments in a real scenario
    # For testing purposes, we use hardcoded values
    run_s3_ingestion("your_s3_bucket_name", "your_data_folder")