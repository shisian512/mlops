import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset

ref_df = pd.read_csv("../data/train.csv")
cur_df = pd.read_csv("../data/current.csv")

cur_df["y"] = np.random.normal(
    loc=ref_df["y"].mean(),
    scale=ref_df["y"].std() * 1.5,
    size=len(cur_df)
)
ref_df["y"] = ref_df["y"]

if "y" in cur_df.columns:
    cur_df["y"] = np.where(np.random.rand(len(cur_df)) < 0.7, 1, 0)

if "text" in cur_df.columns:
    cur_df["text"] = cur_df["text"].str.replace("old_phrase", "new_phrase_123")

report = Report(
    metrics=[DataDriftPreset()],
    include_tests=True
)

result = report.run(
    reference_data=ref_df,
    current_data=cur_df
)

report.save_html("../report.html")
