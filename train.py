import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters
n_estimators = 100
max_depth = 3
random_state = 42

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", acc)

    # Log model
    mlflow.sklearn.log_model(clf, "model")

    print(f"✅ Model trained & logged with accuracy: {acc:.4f}")
