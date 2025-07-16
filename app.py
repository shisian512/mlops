import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import mlflow.sklearn
import pandas as pd
from predict import predict

# # configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("mlops_app")

# # set MLflow tracking
# mlflow.set_tracking_uri("http://mlflow:5000")
# # mlflow.set_tracking_uri("http://192.168.0.124:5000")
# model_name = "sk-learn-random-forest-reg-model@production"
# model_uri = f"models:/{model_name}"
# model = mlflow.sklearn.load_model(model_uri)

app = FastAPI()


class InputFeatures(BaseModel):
    data: List[List[float]]


@app.post("/predict")
def predict(input: InputFeatures):
    # Validate input shape
    if not all(len(row) == 4 for row in input.data):
        raise HTTPException(status_code=422, detail="Each record must have exactly 4 features")

    # prepare DataFrame
    columns = [f"feature_{i}" for i in range(4)]
    df = pd.DataFrame(input.data, columns=columns)

    # run inference
    predictions = predict(df)
    return {"predictions": predictions.tolist()}
