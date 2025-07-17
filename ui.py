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

import streamlit as st
import requests

# ─── Configuration ─────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/predict"  # Prediction endpoint URL
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
    return [
        st.number_input(label, value=0.0, format="%f")
        for label in FEATURE_PROMPTS
    ]


def get_prediction(features: list) -> float:
    """
    Send a POST request to the prediction API with the feature list.

    Args:
        features: List of float feature values.

    Returns:
        The predicted value as float.

    Raises:
        requests.HTTPError: If the HTTP request returned an unsuccessful status code.
        requests.RequestException: For network-related errors or timeouts.
    """
    payload = {"data": [features]}
    response = requests.post(API_URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    result = response.json()
    # Expecting {'predictions': [value]}
    return result.get("predictions", [None])[0]


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
