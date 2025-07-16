import streamlit
import requests

# Title
streamlit.title("🧠 ML Regression Predictor")

# Description
streamlit.write("Enter feature values below:")
f0 = streamlit.number_input("Feature 0", value=1.0)
f1 = streamlit.number_input("Feature 1", value=2.0)
f2 = streamlit.number_input("Feature 2", value=3.0)
f3 = streamlit.number_input("Feature 3", value=4.0)

# Press button to test API for prediction
if streamlit.button("Predict"):
    input_data = {"data": [[f0, f1, f2, f3]]}
    try:
        # Make a POST request to the prediction API
        resp = requests.post("http://localhost:8000/predict", json=input_data, timeout=5)
        
        # Check if the response is successful
        resp.raise_for_status()
        pred = resp.json().get("predictions", [])[0]
        
        # Display the prediction result
        streamlit.success(f"✅ Prediction: {pred:.4f}")
    except Exception as e:
        streamlit.error(f"❌ Error: {e}")
