from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

client = MlflowClient()

mlflow.set_tracking_uri('http://localhost:5000')

# Set model version alias
model_name = "sk-learn-random-forest-reg-model"
model_version_alias = "the_best_model_ever"
client.set_registered_model_alias(
    model_name, model_version_alias, "4"
)  # Duplicate of step in UI

# Get information about the model
model_info = client.get_model_version_by_alias(model_name, model_version_alias)
model_tags = model_info.tags
print(model_tags)

# Get the model version using a model URI
model_uri = f"models:/{model_name}@{model_version_alias}"
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
