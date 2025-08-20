import pandas as pd
import numpy as np

# Step 1: Generate dataset
def generate_dataset(n_samples=10, n_features=4):
    X = np.random.randint(0, 100, size=(n_samples, n_features))
    y = X.mean(axis=1) + np.random.randn(n_samples)  # dummy target
    columns = [f"feature_{i}" for i in range(n_features)] + ["y"]
    data = np.column_stack([X, y])
    df = pd.DataFrame(data, columns=columns)
    return df

# Step 2: "Train" (just dummy)
def train_model():
    print("Training ....")
    print("Trained")

if __name__ == "__main__":
    df = generate_dataset(n_samples=5, n_features=4)
    print("Generated Dataset:")
    print(df)

    train_model()
