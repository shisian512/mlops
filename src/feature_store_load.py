# src/feature_store_load.py
"""
Feature Store Loading Module

This module provides functionality to load cleaned data from S3 into a DynamoDB feature store.
It serves as a bridge between the data processing pipeline and the feature store,
making features available for model training and inference.

The module uses PySpark for distributed data processing and AWS SDK (boto3) for
interacting with DynamoDB. It reads processed data from S3 in Parquet format and
loads it into a specified DynamoDB table, which serves as a feature store.

Typical usage:
    spark-submit feature_store_load.py <s3_bucket> <source_folder> <dynamodb_table_name>
"""

import os
import sys
from pyspark.sql import SparkSession
from dotenv import load_dotenv

def run_feature_store_load(s3_bucket, source_folder, dynamodb_table_name):
    """
    Reads cleaned data from S3 and loads it into a DynamoDB feature store.
    
    This function initializes a Spark session with AWS credentials, reads
    processed data from S3 in Parquet format, and loads it into a DynamoDB
    table that serves as a feature store. Each record is converted to the
    appropriate DynamoDB format before insertion.
    
    Args:
        s3_bucket (str): The S3 bucket name containing the processed data
        source_folder (str): The folder path within the S3 bucket
        dynamodb_table_name (str): The name of the DynamoDB table to load features into
        
    Returns:
        None
        
    Raises:
        SystemExit: If AWS credentials are missing or if there are issues with S3/DynamoDB access
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get AWS credentials from environment variables
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-1") # Default to us-east-1 if not specified
    
    # Validate AWS credentials
    if not aws_access_key or not aws_secret_key:
        print("Error: AWS credentials not found in environment variables.")
        sys.exit(1)

    # Initialize Spark session with AWS S3 connectivity configurations
    spark = SparkSession.builder \
        .appName("FeatureStoreLoad") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.609") \
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key) \
        .getOrCreate()
    
    try:
        # Read the cleaned data from S3 in Parquet format
        s3_path = f"s3a://{s3_bucket}/{source_folder}/"
        print(f"Reading processed data from {s3_path}...")
        df = spark.read.parquet(s3_path)
        
        # Define the key column for the feature store
        # This must be a single column that uniquely identifies each record
        key_column = "your_key_column_name" 
        
        # Validate that the key column exists
        if key_column not in df.columns:
            print(f"Error: Key column '{key_column}' not found in the dataset.")
            spark.stop()
            sys.exit(1)
    
        # Import required libraries for DynamoDB interaction
        # Note: These imports are inside the function to ensure they're available
        # in the Spark executor environment
        from pyspark.sql.functions import struct
        from pyspark.sql import Row
        import boto3
    
        # Function to load a single batch of records into DynamoDB
        # This function runs on each partition of the DataFrame
        def load_to_dynamodb(iterator):
            # Create a DynamoDB client
            client = boto3.client(
                'dynamodb',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
            # Process each row in the partition
            for row in iterator:
                # Convert each column to DynamoDB String format
                # Skip None values to avoid DynamoDB errors
                item = {k: {"S": str(v)} for k, v in row.asDict().items() if v is not None}
                # Insert the item into DynamoDB
                client.put_item(TableName=dynamodb_table_name, Item=item)
    
        # Process the DataFrame in parallel and load to DynamoDB
        print(f"Loading features into DynamoDB table: {dynamodb_table_name}...")
        df.foreachPartition(load_to_dynamodb)
        
        print(f"Successfully loaded features into DynamoDB table: {dynamodb_table_name}")
    except Exception as e:
        print(f"Error loading features into DynamoDB: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up Spark session
        spark.stop()

if __name__ == "__main__":
    # Validate command line arguments
    if len(sys.argv) != 4:
        print("Usage: spark-submit feature_store_load.py <s3_bucket> <source_folder> <dynamodb_table_name>")
        print("  s3_bucket: The S3 bucket name containing the processed data")
        print("  source_folder: The folder path within the S3 bucket")
        print("  dynamodb_table_name: The name of the DynamoDB table to load features into")
        sys.exit(1)
    
    # Execute feature store loading with command line arguments
    run_feature_store_load(sys.argv[1], sys.argv[2], sys.argv[3])
