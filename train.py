from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.sklearn
import json
import joblib
import yaml

# Use Mlflow autologging to automatically log parameters, metrics, and models
mlflow.sklearn.autolog()

# Load the prepared dataset
df = pd.read_csv("data/prepared.csv")
feature_cols = ['feature_0','feature_1','feature_2','feature_3']
X = df[feature_cols]
y = df['y']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Train the model and log parameters, metrics, and model artifacts using MLflow
with mlflow.start_run():
    # Define the model parameters and train the model
    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)
    params = cfg["train"]
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    # Save the model using joblib
    joblib.dump(model, "model.pkl")

    # Example of logging custom parameters
    mse_value = mean_squared_error(y_test, model.predict(X_test))
    mlflow.log_metric("mse_custom", mse_value)

    # Log the model metrics in json, for DVC repro
    with open("metrics.json", "w") as f:
        json.dump({"mse": mse_value}, f)
    print("Saved metrics.json:", mse_value)
    
    # Log the sklearn model and register
    mlflow.sklearn.log_model(
        sk_model=model,
        name="regression_model",
        input_example=X_train.iloc[:5],
        registered_model_name="regression_model",
    )
