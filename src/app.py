"""
FastAPI application exposing an endpoint for making predictions
using a champion-aliased RandomForest regression model loaded from MLflow.
"""

from typing import List
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from prometheus_fastapi_instrumentator import Instrumentator

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "regression_model"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
FEATURE_COUNT = 4  # Number of features expected per record

load_dotenv()
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
print("MLflow URI is", mlflow_uri)
mlflow.set_tracking_uri(mlflow_uri)

# ─── Load Model ────────────────────────────────────────────────────────────────
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as e:
    print(f"Warning: Failed to load model at '{MODEL_URI}': {e}")
    model = None

# ─── App Initialization ─────────────────────────────────────────────────────────
app = FastAPI(
    title="ML Regression Prediction API",
    description="Predict endpoint for RandomForest regression model",
    version="1.0.0"
)
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    excluded_handlers=["/metrics"],
    should_instrument_requests_inprogress=True,
)
instrumentator.instrument(app).expose(app)
# ─── Request Schema ────────────────────────────────────────────────────────────
class InputFeatures(BaseModel):
    data: List[List[float]]

# ─── Prediction Endpoint ───────────────────────────────────────────────────────
@app.post("/predict")
def predict(input: InputFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No model is currently available. Please train and register a model first."
        )
    # Validate input shape
    for record in input.data:
        if len(record) != FEATURE_COUNT:
            raise HTTPException(
                status_code=422,
                detail=f"Each record must have exactly {FEATURE_COUNT} features"
            )
    # Prepare DataFrame for prediction
    column_names = [f"feature_{i}" for i in range(FEATURE_COUNT)]
    df = pd.DataFrame(input.data, columns=column_names)
    # Perform prediction
    try:
        preds = model.predict(df)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {e}"
        )
    return {"predictions": preds.tolist()}
