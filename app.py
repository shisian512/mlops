import time
import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
import mlflow.sklearn
import pandas as pd

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops_app")

# set MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
model_name = "sk-learn-random-forest-reg-model"
model_uri = f"models:/{model_name}@production"
model = mlflow.sklearn.load_model(model_uri)

app = FastAPI()


class InputFeatures(BaseModel):
    data: List[List[float]]


@app.middleware("http")
async def log_requests_and_latency(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        logger.error(f"Error at {request.method} {request.url.path} in {duration:.2f}ms: {e}")
        raise
    duration = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} completed in {duration:.2f}ms")
    response.headers["X-Process-Time-ms"] = f"{duration:.2f}"
    return response


@app.post("/predict")
def predict(input: InputFeatures):
    # Validate input shape
    if not all(len(row) == 4 for row in input.data):
        raise HTTPException(status_code=422, detail="Each record must have exactly 4 features")

    # prepare DataFrame
    columns = [f"feature_{i}" for i in range(4)]
    df = pd.DataFrame(input.data, columns=columns)

    # run inference
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}
