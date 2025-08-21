"""
ui.py

Streamlit-based UI for making predictions with a deployed
RandomForest regression model via a REST API.

Steps:
  1. Display title and instructions.
  2. Collect four feature inputs from the user.
  3. Send input data to prediction endpoint.
  4. Show the returned prediction or error message.

Usage:
    streamlit run ui.py
"""

# Standard library imports
import os

# Third-party imports
import requests
import streamlit as st
from dotenv import load_dotenv

# ─── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()
API_URL = os.getenv("API_URL")
TIMEOUT = 5  # Request timeout in seconds
FEATURE_PROMPTS = [
    "Feature 0",
    "Feature 1",
    "Feature 2",
    "Feature 3",
]

# ─── UI Setup ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ML Regression Predictor", layout="centered")


def render_input_fields() -> list:
    """
    Render numeric input fields for each feature.

    Returns:
        List of float values entered by the user.
    """
    st.write("Enter feature values below:")
    return [st.number_input(label, value=0.0, format="%f") for label in FEATURE_PROMPTS]


def get_prediction(features: list) -> float:
    response = requests.post(f"{API_URL}", json={"data": [features]}, timeout=TIMEOUT)
    if response.status_code == 503:
        raise Exception(
            "No model is currently available. Please train and register a model first."
        )
    response.raise_for_status()
    return response.json()["predictions"][0]


def main():
    """
    Main entry point for the Streamlit app.
    """
    st.title("ML Regression Predictor")
    features = render_input_fields()

    if st.button("Predict"):
        try:
            prediction = get_prediction(features)
            st.success(f"Prediction: {prediction:.4f}")
        except Exception as err:
            st.error(f"Error: {err}")


if __name__ == "__main__":
    main()
