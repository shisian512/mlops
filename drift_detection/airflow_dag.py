from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG('daily_drift_detection',
    start_date=datetime(2025, 1, 1),
    # if you want faster, just replace '@daily' to
    # schedule_interval='0 */6 * * *'  # Every 6 hours
    schedule_interval='@daily',
    catchup=False) as dag:

    run_drift = BashOperator(
        task_id='run_drift_report',
        bash_command='python /opt/mlops/drift_report.py'
    )

# After Model Retraining
# from airflow.operators.python import PythonOperator

# def retrain_model():
#     # model training code here
#     pass

# def run_drift():
#     import subprocess
#     subprocess.run(["python", "/opt/mlops/drift_report.py"])

# with DAG('event_triggered_drift_check',
#          start_date=datetime(2025, 1, 1),
#          schedule_interval=None,
#          catchup=False) as dag:

#     training = PythonOperator(task_id="train_model", python_callable=retrain_model)
#     drift_check = PythonOperator(task_id="run_drift", python_callable=run_drift)

#     training >> drift_check
