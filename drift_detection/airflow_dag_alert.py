from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import json, requests

def check_drift():
    with open("/opt/mlops/reports/full_drift_report.json") as f:
        report = json.load(f)
    drift_score = report["metrics"][0]["result"]["dataset_drift"]
    if drift_score > 0.5:
        requests.post("https://hooks.slack.com/services/XXX", json={"text": f"🚨 Drift score: {drift_score}"})

with DAG("drift_alerting_dag",
         start_date=datetime(2025, 1, 1),
         schedule_interval="@daily",
         catchup=False) as dag:

    alert_task = PythonOperator(task_id="check_drift", python_callable=check_drift)
