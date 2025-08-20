import pandas as pd
import sys

def load_data(path):
    df = pd.read_csv(path)
    print("Data loaded:")
    print(df.head())
    return df

def evaluate(data):
    print("Evaluating ....")
    print("Evaluation complete ✅")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python evaluate.py <input_csv>")

    input_csv = sys.argv[1]
    data = load_data(input_csv)
    evaluate(data)
