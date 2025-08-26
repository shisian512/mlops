"""
data_preprocess.py

This module handles data preprocessing operations on S3-stored Parquet data using PySpark.
It performs the following operations:
1. Connects to an S3 bucket using AWS credentials from environment variables
2. Reads Parquet data from a specified source folder
3. Performs data cleaning operations, specifically imputing missing numeric values with column means
4. Writes the cleaned data back to S3 in a destination folder

The preprocessing is essential for ensuring data quality before model training.

Environment Variables Required:
- AWS_ACCESS_KEY_ID: AWS access key for S3 bucket access
- AWS_SECRET_ACCESS_KEY: AWS secret key for S3 bucket access

Dependencies:
- pyspark: For distributed data processing
- python-dotenv: For loading environment variables
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
from dotenv import load_dotenv

def run_s3_preprocess(s3_bucket, source_folder, destination_folder):
    """
    Reads Parquet data from S3, preprocesses it by imputing missing values, and writes the output back to S3.
    
    Args:
        s3_bucket (str): The name of the S3 bucket containing the data
        source_folder (str): The folder path within the S3 bucket where source data is stored
        destination_folder (str): The folder path within the S3 bucket where cleaned data will be written
        
    Returns:
        None: The function writes the processed data to S3 and logs processing information
        
    Raises:
        ValueError: If AWS credentials are not found in environment variables
        Exception: For any errors during Spark processing or S3 operations
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the credentials from the environment variables
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if not aws_access_key or not aws_secret_key:
        raise ValueError("AWS credentials not found in environment variables")

    # Define configurations for S3 access
    spark = SparkSession.builder \
        .appName("S3Preprocess") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.609") \
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key) \
        .getOrCreate()
    
    # Read data from the S3 folder
    df = spark.read.parquet(f"s3a://{s3_bucket}/{source_folder}/")
    print(f"Loaded data with {df.count()} rows and {len(df.columns)} columns.")

    # Impute missing numeric values with the mean
    numeric_cols = [c for c, dtype in df.dtypes if dtype in ("int", "double", "float", "long")]
    
    # Calculate means for imputation
    imputation_values = {c: df.select(mean(col(c))).first()[0] for c in numeric_cols}

    # Fill nulls with the calculated means
    df_clean = df.fillna(imputation_values)

    print(f"After imputation, any NaNs left? {df_clean.toPandas().isnull().any().any()}")
    
    # Write the cleaned data back to S3
    df_clean.write.mode("overwrite").parquet(f"s3a://{s3_bucket}/{destination_folder}/")
    print(f"Cleaned data saved to 's3a://{s3_bucket}/{destination_folder}/'")

    spark.stop()

# This part is for running the script directly if needed
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess data from S3 and save back to S3")
    parser.add_argument("--s3_bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument("--source_folder", type=str, required=True, help="Source folder in S3 bucket")
    parser.add_argument("--destination_folder", type=str, required=True, help="Destination folder in S3 bucket")
    
    args = parser.parse_args()
    
    run_s3_preprocess(args.s3_bucket, args.source_folder, args.destination_folder)