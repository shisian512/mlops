from evidently import Report
from evidently.metrics import DataDriftPreset
import pandas as pd
import os

# Example data loading
ref = pd.read_csv("train.csv")
cur = pd.read_csv("current.csv")

# Generate report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)

# # Create reports directory if it doesn't exist
# os.makedirs("reports", exist_ok=True)

# Save the report
report.save_html("full_drift_report.html")
report.save_json("full_drift_report.json")