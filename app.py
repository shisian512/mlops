from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import mlflow.sklearn
import pandas as pd

# Load MLflow model
model_name = "regression_model"
# version = "3"
# model_uri = f"models:/{model_name}/{version}"
model_uri = f"models:/{model_name}@champion"
model = mlflow.sklearn.load_model(model_uri)

app = FastAPI()


class InputFeatures(BaseModel):
    data: List[List[float]]


# API endpoint for prediction
@app.post("/predict")
def predict(input: InputFeatures):
    # Validate input shape
    if not all(len(row) == 4 for row in input.data):
        raise HTTPException(status_code=422, detail="Each record must have exactly 4 features")

    # Convert input data to DataFrame and make predictions
    df = pd.DataFrame(input.data, columns=[f"feature_{i}" for i in range(4)])
    preds = model.predict(df)
    return {"predictions": preds.tolist()}
