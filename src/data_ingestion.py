# src/s3_ingestion.py
import os
from pyspark.sql import SparkSession
from dotenv import load_dotenv

def run_s3_ingestion(s3_bucket, s3_folder):
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the credentials from the environment variables
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    # Define configurations for S3 access
    spark = SparkSession.builder \
        .appName("S3Ingestion") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.609") \
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key) \
        .getOrCreate()
    
    # Read data from a folder in an S3 bucket
    df = spark.read.csv(f"s3a://{s3_bucket}/{s3_folder}/", header=True, inferSchema=True)
    
    # Do some processing and write the output
    df.write.mode("overwrite").parquet(f"s3a://{s3_bucket}/processed_data/")

    spark.stop()

# This part is for running the script directly if needed, but not required by Airflow
if __name__ == "__main__":
    # You might get these from command-line arguments in a real scenario
    run_s3_ingestion("your_s3_bucket_name", "your_data_folder")