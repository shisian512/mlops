# src/preprocess.py
"""
Data preprocessing module for ML pipeline.

This script handles the initial data cleaning and preprocessing steps before model training.
It currently focuses on handling missing values by removing rows with NaN values.
"""

import pandas as pd
import sys
import os

if __name__ == "__main__":
    # Validate command line arguments
    if len(sys.argv) != 3:
        raise ValueError("Usage: python preprocess.py <input_csv> <output_csv>")

    # Extract input and output file paths from command line arguments
    input_csv = sys.argv[1]  # Path to raw data file
    output_csv = sys.argv[2]  # Path where processed data will be saved

    # Load raw data from CSV file
    # This loads the entire dataset into memory as a pandas DataFrame
    df = pd.read_csv(input_csv)

    # Data cleaning: Remove rows with any NaN values
    # This is a simple approach to handling missing data
    # In a production environment, you might want to consider:
    # - Imputation strategies (mean, median, mode) for numeric columns
    # - Using more sophisticated techniques like KNN imputation
    # - Handling categorical missing values differently
    # - Checking if missing data is random or has patterns
    df = df.dropna()

    # Create output directory if it doesn't exist
    # This ensures the directory structure exists before saving the file
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save the processed DataFrame to CSV
    # The index=False parameter prevents adding a row index column
    df.to_csv(output_csv, index=False)
    
    # Log the completion of preprocessing
    print(f"Processed {input_csv} -> {output_csv} (dropped NaNs)")
    # TODO: Consider adding more detailed logging with row counts before/after
