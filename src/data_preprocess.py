# src/s3_preprocess.py
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
from dotenv import load_dotenv


def run_s3_preprocess(s3_bucket, source_folder, destination_folder):
    """
    Reads Parquet data from S3, preprocesses it, and writes the output back to S3.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get the credentials from the environment variables
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    # Define configurations for S3 access
    spark = (
        SparkSession.builder.appName("S3Preprocess")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.609",
        )
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key)
        .getOrCreate()
    )

    # Read data from the S3 folder
    df = spark.read.parquet(f"s3a://{s3_bucket}/{source_folder}/")
    print(f"Loaded data with {df.count()} rows and {len(df.columns)} columns.")

    # Impute missing numeric values with the mean
    numeric_cols = [
        c for c, dtype in df.dtypes if dtype in ("int", "double", "float", "long")
    ]

    # Calculate means for imputation
    imputation_values = {c: df.select(mean(col(c))).first()[0] for c in numeric_cols}

    # Fill nulls with the calculated means
    df_clean = df.fillna(imputation_values)

    print(
        f"After imputation, any NaNs left? {df_clean.toPandas().isnull().any().any()}"
    )

    # Write the cleaned data back to S3
    df_clean.write.mode("overwrite").parquet(f"s3a://{s3_bucket}/{destination_folder}/")
    print(f"Cleaned data saved to 's3a://{s3_bucket}/{destination_folder}/'")

    spark.stop()


# This part is for running the script directly if needed
if __name__ == "__main__":
    run_s3_preprocess("your_s3_bucket_name", "processed_data", "cleaned_data")
