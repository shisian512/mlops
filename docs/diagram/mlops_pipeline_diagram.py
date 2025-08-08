from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.onprem.client import Users
from diagrams.programming.language import Python
from diagrams.onprem.container import Docker

with Diagram(
    "🚀 MLOps E2E Pipeline", show=False, filename="mlops_e2e_pipeline", outformat="png"
):

    user = Users("User")

    with Cluster("Data Versioning"):
        dvc = Custom("DVC", "icons/dvc.png")
        prepare = Python("prepare.py")
        dvc >> prepare

    with Cluster("Training"):
        train = Python("train.py")
        mlflow = Custom("MLflow", "icons/mlflow.png")
        prepare >> train >> mlflow

    with Cluster("CI/CD"):
        gha = Custom("GitHub Actions", "icons/github_actions.png")
        docker = Docker("Docker Image")
        gha >> Edge(label="lint/test/deploy") >> docker

    with Cluster("Serving"):
        fastapi = Python("FastAPI\n(app.py)")
        predict = Python("predict.py")
        mlflow >> predict >> fastapi >> user

    with Cluster("UI"):
        streamlit = Custom("Streamlit", "icons/streamlit.png")
        mlflow >> streamlit >> user

    with Cluster("Monitoring"):
        prometheus = Custom("Prometheus", "icons/prometheus.png")
        grafana = Custom("Grafana", "icons/grafana.jpg")
        alert = Custom("Alertmanager", "icons/prometheus.png")
        fastapi >> prometheus >> grafana
        alert << Edge(label="alerts") << prometheus

    with Cluster("Drift Detection"):
        airflow = Custom("Airflow", "icons/airflow.png")
        evidently = Custom("EvidentlyAI", "icons/evidentlyai.jpg")
        (
            airflow
            >> Edge(label="run DAGs")
            >> evidently
            >> Edge(label="metrics")
            >> prometheus
        )
