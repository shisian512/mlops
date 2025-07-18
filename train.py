"""
End-to-end training, registration, and champion/challenger promotion
for a RandomForest regression model using MLflow.
"""
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml
import warnings
import os
import mlflow
from dotenv import load_dotenv

warnings.simplefilter(action='ignore', category=FutureWarning)

EXPERIMENT_NAME  = "regression_experiment"
MODEL_NAME       = "regression_model"
CHALLENGER_ALIAS = "challenger"
CHAMPION_ALIAS   = "champion"
PARAMS_FILE      = "params.yaml"
DATA_PATH        = "data/prepared.csv"
METRICS_PATH     = "metrics.json"
MODEL_PICKLE     = "model.pkl"
FEATURE_COLS     = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
TARGET_COL       = 'y'
METRIC_KEY       = "mse_custom"

load_dotenv()
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
print("MLflow URI is", mlflow_uri)
mlflow.set_tracking_uri(mlflow_uri)

mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()
client = MlflowClient()

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_data(path: str, cfg: dict):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    ds_cfg = cfg['data']
    return train_test_split(
        X, y,
        test_size=ds_cfg['test_size'],
        random_state=ds_cfg['random_state']
    )

def get_latest_version(model_name: str) -> str:
    versions = client.search_model_versions(f"name='{model_name}'")
    latest = max(int(v.version) for v in versions)
    return str(latest)

def get_metric_for_alias(model_name: str, alias: str, metric_key: str) -> float:
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            if alias in v.aliases:
                run_id = v.run_id
                metrics = client.get_run(run_id).data.metrics
                return metrics.get(metric_key, float('inf'))
        return float('inf')
    except Exception:
        return float('inf')

def champion_challenger_test(model_name: str, metric_key: str) -> bool:
    champ_val = get_metric_for_alias(model_name, CHAMPION_ALIAS, metric_key)
    chall_val = get_metric_for_alias(model_name, CHALLENGER_ALIAS, metric_key)
    print(f"Current Champion {metric_key}: {champ_val:.6f}")
    print(f"Current Challenger {metric_key}: {chall_val:.6f}")
    return chall_val < champ_val

def promote_alias(model_name: str, alias: str, version: str):
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version
    )
    print(f"Alias '{alias}' → version {version}")

def main():
    cfg = load_config(PARAMS_FILE)
    X_train, X_test, y_train, y_test = load_data(DATA_PATH, cfg)
    model_cfg = cfg['model']['hyperparams']
    model = RandomForestRegressor(**model_cfg)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.6f}")
    with open(METRICS_PATH, 'w') as f:
        f.write(f'{{"mse_custom": {mse}}}\n')
    import pickle
    with open(MODEL_PICKLE, 'wb') as f:
        pickle.dump(model, f)
    with mlflow.start_run() as run:
        mlflow.log_metric(METRIC_KEY, mse)
        mlflow.sklearn.log_model(model, "model")
        mv = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=MODEL_NAME
        )
        client.set_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS, mv.version)
        if champion_challenger_test(MODEL_NAME, METRIC_KEY):
            promote_alias(MODEL_NAME, CHAMPION_ALIAS, mv.version)

if __name__ == "__main__":
    main()
