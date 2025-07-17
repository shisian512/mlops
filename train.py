from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml, mlflow, mlflow.sklearn, joblib, json
from mlflow import MlflowClient

# Use Mlflow autologging to automatically log parameters, metrics, and models
# Set experiment name to avoid 'experiment ID 0 not found' error
mlflow.set_experiment("regression_experiment")
mlflow.sklearn.autolog()
client = MlflowClient()

# Load the prepared dataset
df = pd.read_csv("data/prepared.csv")
feature_cols = ['feature_0','feature_1','feature_2','feature_3']
X = df[feature_cols]
y = df['y']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

model_name = "regression_model"
alias = "challenger"


def run_comparison():
    return True


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
    
    # Log the sklearn model and register
    mlflow.sklearn.log_model(
        sk_model=model,
        name=model_name,
        input_example=X_train.iloc[:5],
        registered_model_name=model_name,
    )
    
    # Register the latest model with alias of 'challenger'
    latest_version = client.get_latest_versions(model_name, stages=None)[0].version
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=latest_version,  # Use the latest version number
    )
    
    # Run comparison of challenger and champion models
    current_model_is_better = run_comparison() # either return true or false
    
    # If challenger is better, it will be registered as the champion
    if current_model_is_better:
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=latest_version,  # Use the latest version number
        )
    else:
        # If the challenger is not better, we do not change the champion alias
        pass
        
    # # Print the model information of latest champion (deployed in production)
    # model_version_alias = "champion"
    # model_info = client.get_model_version_by_alias(model_name, model_version_alias)
    # print("Model Info for Champion model : ", model_info)
