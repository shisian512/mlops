"""FastAPI application exposing an endpoint for making predictions
using a champion-aliased RandomForest regression model loaded from MLflow.

This module provides a REST API service that:
1. Loads a pre-trained ML model from MLflow Model Registry
2. Exposes an endpoint for making predictions
3. Includes input validation and error handling
4. Provides Prometheus metrics for monitoring
"""

# Standard library imports
import os
from typing import List

# Third-party imports
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

# ─── Configuration ─────────────────────────────────────────────────────────────
# Model configuration parameters
MODEL_NAME = "regression_model"  # Name of the registered model in MLflow
MODEL_ALIAS = "champion"  # Using the champion alias ensures we always use the best model
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"  # MLflow model URI format
FEATURE_COUNT = 4  # Number of features expected per input record

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
    version="1.0.0",
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
    """
    Pydantic model defining the expected request body format.
    
    Attributes:
        data (List[List[float]]): A list of records, where each record is a list of
                                  floating-point feature values. Each record must
                                  contain exactly FEATURE_COUNT features.
    """
    data: List[List[float]]

# ─── Prediction Endpoint ───────────────────────────────────────────────────────
@app.post("/predict")
def predict(input: InputFeatures):
    """
    Generate predictions using the loaded ML model.
    
    This endpoint accepts a batch of records, validates them, and returns
    predictions for each record.
    
    Args:
        input (InputFeatures): The request body containing records to predict.
        
    Returns:
        dict: A dictionary with a 'predictions' key containing the model's predictions.
        
    Raises:
        HTTPException: If the model is not available, input format is invalid,
                      or prediction fails.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No model is currently available. Please train and register a model first.",
        )
    # Validate input shape
    for record in input.data:
        if len(record) != FEATURE_COUNT:
            raise HTTPException(
                status_code=422,
                detail=f"Each record must have exactly {FEATURE_COUNT} features",
            )
    # Prepare DataFrame for prediction
    column_names = [f"feature_{i}" for i in range(FEATURE_COUNT)]
    df = pd.DataFrame(input.data, columns=column_names)
    # Perform prediction
    try:
        preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    return {"predictions": preds.tolist()}