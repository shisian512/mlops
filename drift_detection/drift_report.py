from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd
import os

# Example data loading
ref = pd.read_csv("../data/train.csv")
cur = pd.read_csv("../data/current.csv")

# Generate report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)
report.save_html("report.html")
