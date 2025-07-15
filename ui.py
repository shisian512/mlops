import streamlit as st
import requests

# Title
st.title("🧠 ML Regression Predictor")

# User input
st.write("Enter feature values below:")
f0 = st.number_input("Feature 0", value=1.0)

# Press button to test API for prediction
if st.button("Predict"):
    input_data = {"data": [[f0]}
    try:
        resp = requests.post("http://backend:8000/predict", json=input_data, timeout=5)
        resp.raise_for_status()
        pred = resp.json().get("predictions", [])[0]
        st.success(f"✅ Prediction: {pred:.4f}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
