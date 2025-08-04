import json

with open("reports/full_drift_report.json") as f:
    report = json.load(f)

drift_score = report["metrics"][0]["result"]["dataset_drift"]
if drift_score > 0.5:
    print("⚠️ Drift detected!")


# You can also inspect feature-level drift if needed:
# for feature in report["metrics"][0]["result"]["drift_by_columns"]:
#     if feature["drift_score"] > 0.3:
#         print(f"Feature {feature['column_name']} drifted!")

# Email:
import smtplib

if drift_score > 0.5:
    server = smtplib.SMTP("smtp.example.com", 587)
    server.starttls()
    server.login("you@example.com", "password")
    server.sendmail("you@example.com", "team@example.com", "Subject: Drift Alert\n\nDrift score exceeded threshold!")
    server.quit()


# Slack:
import requests

if drift_score > 0.5:
    requests.post("https://hooks.slack.com/services/XXX", json={"text": "🚨 Drift detected! Score: {:.2f}".format(drift_score)})
