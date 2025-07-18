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

---

## Project Structure

```text
mlops/
├── src/
│   ├── api/         # FastAPI app and endpoints
│   ├── data/        # Data preparation scripts
│   ├── models/      # Model loading and prediction logic
│   ├── services/    # Training, registration, and orchestration
│   └── utils/       # Utility modules (config, helpers)
├── data/            # Raw and prepared datasets
├── mlartifacts/     # MLflow artifact storage
├── mlruns/          # MLflow experiment tracking
├── dvc.yaml         # DVC pipeline definition
├── Dockerfile.*     # Dockerfiles for each service
├── docker-compose.yml
├── deployment.yaml  # Kubernetes manifests
├── requirements.txt # Python dependencies
├── README.md        # Project documentation
└── ...
```

---

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

1. **Install dependencies**
   
   
   
   ```bash
   pip install -r requirements.txt
   
2. **Prepare data**
   
   
   ```bash
   python src/data/prepare.py
   
3. **Train model**
   
   
   ```bash
   python src/services/train.py
   
4. **Run API**
   
   
   ```bash
   uvicorn src.api.app:app --reload --port 8000
   
5. **Run UI**
   
   
   ```bash
   streamlit run ui.py
   ```
   
6. **MLflow UI**
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   ```
1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize DVC (if not done yet)**

   ```bash
   dvc init
   dvc remote add -d storage ./dvc-storage
   ```

3. **Run the DVC pipeline**

   ```bash
   dvc repro
   ```

4. **Prepare data manually (optional)**

   ```bash
   python src/data/prepare.py
   ```

5. **Train model manually (optional)**

   ```bash
   python src/services/train.py
   ```

6. **Run API**

   ```bash
   uvicorn src.api.app:app --reload --port 8000
   ```

7. **Run UI**

   ```bash
   streamlit run src/frontend/ui.py
   ```

8. **MLflow UI**
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   ```
   ```

---

## Deployment

- **Docker Compose**: `docker-compose up --build`
- **Kubernetes**: Apply `deployment.yaml` to your cluster.

---

## CI/CD

- Automated with GitHub Actions (`.github/workflows/main.yml`).
- Linting, testing, and deployment steps included.

---

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)

---

## Author

Personal project by [shisian512](https://github.com/shisian512) for learning and demonstration purposes.
