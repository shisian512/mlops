from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import mlflow.sklearn
import pandas as pd
from utils import get_model_path

# Load the model
model_path = get_model_path()
print(f"Loading model from: {model_path}")
model = mlflow.sklearn.load_model(model_path)

app = FastAPI()

class InputFeatures(BaseModel):
    data: List[List[float]]  # 2D list

@app.post("/predict")
def predict(input: InputFeatures):
    columns = ["sepal length (cm)", "sepal width (cm)",
               "petal length (cm)", "petal width (cm)"]
    df = pd.DataFrame(input.data, columns=columns)
    preds = model.predict(df)
    return {"predictions": preds.tolist()}
