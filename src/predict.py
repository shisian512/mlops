# src/predict.py
"""
This module provides prediction functionality using a pre-trained machine learning model
loaded from MLflow model registry. It handles loading the model and making predictions
on new data.
"""

import mlflow
import mlflow.sklearn

# Set the MLflow tracking server URI to connect to the MLflow service
mlflow.set_tracking_uri("http://mlflow:5000")

# Define the model name with the production alias to ensure we're using the latest production model
model_name = "sk-learn-random-forest-reg-model@production"

# Construct the model URI using the MLflow models:/ scheme to load from the Model Registry
model_uri = f"models:/{model_name}"

# Load the model from MLflow Model Registry
# This will download the model artifacts and deserialize the model
model = mlflow.sklearn.load_model(model_uri)


def predict(input_data):
    """
    Generate predictions using the loaded model.
    
    Args:
        input_data (pandas.DataFrame or numpy.ndarray): The input features for prediction.
            Must match the format and features expected by the model.
            
    Returns:
        numpy.ndarray: The model's predictions for the provided input data.
    """
    predictions = model.predict(input_data)
    return predictions
