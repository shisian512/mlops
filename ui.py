import streamlit as st
import requests

# Title
st.title("🧠 ML Regression Predictor")

# User input
st.write("Enter feature values below:")
f0 = st.number_input("Feature 0", value=1.0)
f1 = st.number_input("Feature 1", value=2.0)
f2 = st.number_input("Feature 2", value=3.0)
f3 = st.number_input("Feature 3", value=4.0)

# Press button to test API for prediction
if st.button("Predict"):
    input_data = {"data": [[f0, f1, f2, f3]]}
    try:
        resp = requests.post("http://backend:8000/predict", json=input_data, timeout=5)
        # resp = requests.post("http://localhost:8000/predict", json=input_data, timeout=5)
        resp.raise_for_status()
        pred = resp.json().get("predictions", [])[0]
        st.success(f"✅ Prediction: {pred:.4f}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
