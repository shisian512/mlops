# src/feature_store_load.py
import os
import sys
from pyspark.sql import SparkSession
from dotenv import load_dotenv


def run_feature_store_load(s3_bucket, source_folder, dynamodb_table_name):
    """
    Reads cleaned data and loads it into a DynamoDB feature store.
    """
    # Load environment variables
    load_dotenv()

    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-1")  # Use your preferred region

    # Define Spark Session
    spark = (
        SparkSession.builder.appName("FeatureStoreLoad")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.609",
        )
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key)
        .getOrCreate()
    )

    # Read the cleaned data from S3
    df = spark.read.parquet(f"s3a://{s3_bucket}/{source_folder}/")

    # Define the key column for the feature store
    # This must be a single column that uniquely identifies each record
    key_column = "your_key_column_name"

    # Spark requires the boto3 and dynamodb dependencies to be available
    # The simplest way is to configure the SparkSubmitOperator to use a specific
    # Python environment that has these libraries installed.
    from pyspark.sql.functions import struct
    from pyspark.sql import Row
    import boto3

    # Function to load a single batch into DynamoDB
    def load_to_dynamodb(iterator):
        client = boto3.client(
            "dynamodb",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region,
        )
        for row in iterator:
            item = {k: {"S": str(v)} for k, v in row.asDict().items() if v is not None}
            client.put_item(TableName=dynamodb_table_name, Item=item)

    # Convert the Spark DataFrame to RDD and then map to load to DynamoDB
    df.foreachPartition(load_to_dynamodb)

    print(f"Successfully loaded features into DynamoDB table: {dynamodb_table_name}")
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: spark-submit feature_store_load.py <s3_bucket> <source_folder> <dynamodb_table_name>"
        )
        sys.exit(1)
    run_feature_store_load(sys.argv[1], sys.argv[2], sys.argv[3])
