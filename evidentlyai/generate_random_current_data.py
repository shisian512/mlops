import pandas as pd
import numpy as np

df = pd.read_csv("../data/train.csv")

df_fake = pd.DataFrame()

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        mean = df[col].mean()
        std = df[col].std()
        df_fake[col] = np.random.normal(loc=mean, scale=std, size=len(df))
    elif pd.api.types.is_bool_dtype(df[col]):
        df_fake[col] = np.random.choice([True, False], size=len(df))
    elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
        df_fake[col] = np.random.choice(df[col].dropna().unique(), size=len(df))
    else:
        df_fake[col] = df[col]

df_fake.to_csv("../data/current.csv", index=False)

print("✅ Generated file: current.csv")
