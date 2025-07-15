import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import json

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters
n_estimators = 100
max_depth = 3
random_state = 42

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# # Create a new MLflow Experiment
# mlflow.set_experiment("mlops-iris-classification")

mlflow.sklearn.autolog()

# Train
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Model trained & logged with accuracy: {acc:.4f}")

# \mlops\mlartifacts\803409465960920905\models\m-8dfe7760e11d4013b3b782a74f2be5d4
run_id = mlflow.last_active_run().info.run_id
print(f"Logged data and model in run {run_id}")

# mlartifacts\0\models\m-8e37e44f26ff4871a6c4c485df19ef38
model_path = f"mlartifacts/0/{run_id}/models/m-{run_id}"

with open("last_model_path.json", "w") as f:
    json.dump({"model_path": model_path}, f)
