import pandas as pd

df = pd.read_csv("data/train.csv")
df.fillna(df.median(), inplace=True)
df.to_csv("data/prepared.csv", index=False)
