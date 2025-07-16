from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.sklearn
import json
import joblib
# mlflow.set_tracking_uri('http://mlflow:5000')
mlflow.set_tracking_uri('http://localhost:5000')
df = pd.read_csv("data/prepared.csv")
X = df[['x']]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    params = {"max_depth": 2, "random_state": 42}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    joblib.dump(model, "model.pkl")
    # Log parameters and metrics using the MLflow APIs
    mlflow.log_params(params)

    y_pred = model.predict(X_test)
    mse_value = mean_squared_error(y_test, y_pred)
    mlflow.log_metrics({"mse": mse_value})

    with open("metrics.json", "w") as f:
        json.dump({"mse": mse_value}, f)
    print("Saved metrics.json:", mse_value)
    
    # Log the sklearn model and register
    mlflow.sklearn.log_model(
        sk_model=model,
        name="sklearn-model",
        input_example=X_train,
        registered_model_name="sk-learn-random-forest-reg-model",
    )
