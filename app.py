from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri('http://localhost:5000')

# Define model name and version from MLflow Model Registry
model_name = "sk-learn-random-forest-reg-model"
model_version = "latest"  # You can also use a specific version number or stage like "Production"

# Load the model from the registry
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

# Initialize FastAPI app
app = FastAPI()

# Define input data schema
class InputFeatures(BaseModel):
    data: List[List[float]]  # Each sublist represents one sample with 4 features

# Define prediction route
@app.post("/predict")
def predict(input: InputFeatures):
    # Create DataFrame from input data
    columns = [f"feature_{i}" for i in range(4)]  # Adjust if your model has different feature names
    df = pd.DataFrame(input.data, columns=columns)
    
    # Make predictions
    predictions = model.predict(df)
    
    # Return the result as JSON
    return {"predictions": predictions.tolist()}
