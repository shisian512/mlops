
# MLOps End-to-End Pipeline

A modular, production-grade MLOps pipeline for regression tasks, featuring:

- **Data Preparation**: Automated cleaning and preprocessing.
- **Model Training**: MLflow-tracked RandomForest regression with champion/challenger logic.
- **Model Registry**: MLflow Model Registry for versioning and alias management.
- **API Service**: FastAPI backend for real-time predictions.
- **UI**: Streamlit frontend for user-friendly predictions.
- **CI/CD**: GitHub Actions for automated testing and deployment.
- **Containerization**: Docker Compose and Kubernetes manifests for local and cloud deployment.
- **Experiment Tracking**: MLflow UI for experiment management.
- **Reproducibility**: DVC for data and pipeline versioning.



## Project Structure

```text
mlops/
├── src/
│   ├── api/         # FastAPI app and endpoints (src/api/app.py)
│   ├── data/        # Data preparation scripts (src/data/prepare.py)
│   ├── models/      # Model loading and prediction logic
│   ├── services/    # Training, registration, and orchestration (src/services/train.py)
│   └── utils/       # Utility modules (config, helpers)
├── data/            # Raw and prepared datasets
├── mlartifacts/     # MLflow artifact storage
├── mlruns/          # MLflow experiment tracking
├── dvc.yaml         # DVC pipeline definition
├── Dockerfile.fastapi      # Dockerfile for FastAPI backend
├── Dockerfile.streamlit    # Dockerfile for Streamlit frontend
├── docker-compose.yml      # Docker Compose for local multi-service setup
├── deployment.yaml         # Kubernetes manifests
├── requirements.txt        # Python dependencies
├── params.yaml             # Model and data parameters
├── README.md               # Project documentation
└── ...
```

---

## Future Plans

- **Monitoring & Observability**: Integrate Prometheus and Grafana for real-time monitoring of service health, resource usage, and application metrics.
- **Drift Detection**: Implement automated data drift and model drift detection using tools like Evidently or custom monitoring scripts, with alerts for significant changes.
- **Model Retraining Automation**: Add pipelines for scheduled or triggered retraining based on drift or performance degradation.
- **Advanced CI/CD**: Expand CI/CD to include canary deployments, blue/green deployments, and automated rollback on failure.
- **Security Enhancements**: Add authentication, authorization, and secrets management for API and UI services.
- **Cloud Deployment**: Provide Terraform or Helm charts for easy deployment to major cloud providers (AWS, GCP, Azure).
- **Feature Store Integration**: Integrate a feature store for better feature management and reuse.
- **Testing Improvements**: Add more comprehensive unit, integration, and end-to-end tests.
- **Documentation**: Expand documentation with architecture diagrams, troubleshooting, and best practices.

## Key Features

- **Modular Codebase**: All logic is separated by concern for maintainability and reusability.
- **MLflow Integration**: Track experiments, register models, and manage model lifecycle.
- **Champion/Challenger Promotion**: Only the best model is promoted to production.
- **API & UI**: Serve predictions via FastAPI and interact via Streamlit.
- **CI/CD**: Automated build, test, and deploy with GitHub Actions.
- **Reproducibility**: DVC ensures data and pipeline reproducibility.
- **Containerization**: Run locally or deploy to Kubernetes with minimal changes.

---


## Quickstart

### 1. Start All Services with Docker Compose

docker-compose up --build
This project uses Docker Compose to run MLflow, FastAPI backend, and Streamlit frontend locally. Make sure Docker and Docker Compose are installed on your system.

#### a. Start All Services

```bash
docker-compose up --build
```

This will start the following services:
- **MLflow UI** (experiment tracking): [http://localhost:5000](http://localhost:5000)
- **FastAPI backend** (prediction API): [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit UI** (user interface): [http://localhost:8501](http://localhost:8501)

> **Note:** The backend and frontend containers use images built from `Dockerfile.fastapi` and `Dockerfile.streamlit`. You should have these services running before attempting to train or retrain the model locally.

> **Note:** The backend and frontend containers use images built from `Dockerfile.fastapi` and `Dockerfile.streamlit`.

---

If you want to train or update the model, you must do this on your native system (not inside Docker). This is because the training process may require direct access to files and dependencies.

### Start All Services with Docker Compose

This project uses Docker Compose to run MLflow, FastAPI backend, and Streamlit frontend locally. Make sure Docker and Docker Compose are installed on your system.

```bash
docker-compose up --build
```

This will start the following services:

- **MLflow UI** (experiment tracking): [http://localhost:5000](http://localhost:5000)
- **FastAPI backend** (prediction API): [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit UI** (user interface): [http://localhost:8501](http://localhost:8501)

> **Note:** The backend and frontend containers use images built from `Dockerfile.fastapi` and `Dockerfile.streamlit`.

---

## Author

Personal project by [shisian512](https://github.com/shisian512) for learning and demonstration purposes.
