"""
api.py

FastAPI application exposing an endpoint for making predictions
using a champion-aliased RandomForest regression model loaded from MLflow.

Steps:
  1. Load the MLflow-registered model using the 'champion' alias.
  2. Define a Pydantic schema for input validation.
  3. Expose a POST /predict endpoint.
  4. Validate incoming payload, make prediction, and return results.

Usage:
    uvicorn api:app --reload --port 8000
"""

from typing import List

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ─── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME = "regression_model"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
FEATURE_COUNT = 4  # Number of features expected per record


# ─── Load Model ────────────────────────────────────────────────────────────────

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(f"Failed to load model at '{MODEL_URI}': {e}")


# ─── App Initialization ─────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Regression Prediction API",
    description="Predict endpoint for RandomForest regression model",
    version="1.0.0"
)


# ─── Request Schema ────────────────────────────────────────────────────────────

class InputFeatures(BaseModel):
    data: List[List[float]]


# ─── Prediction Endpoint ───────────────────────────────────────────────────────

@app.post("/predict")
def predict(input: InputFeatures):
    """
    Make predictions for a batch of input feature records.

    - Validates that each record contains exactly FEATURE_COUNT floats.
    - Returns a JSON dict with a 'predictions' list.
    """
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

    # Return results
    return {"predictions": preds.tolist()}
