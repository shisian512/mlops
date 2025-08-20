import pandas as pd
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python preprocess.py <input_csv> <output_csv>")

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    # Load raw
    df = pd.read_csv(input_csv)

    # (No changes — just pass through)

    # Save to processed
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Copied {input_csv} -> {output_csv}")
