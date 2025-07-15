import mlflow.sklearn
import pandas as pd
from utils import get_model_path

# Load the model
model_path = get_model_path()
model = mlflow.sklearn.load_model(model_path)

def predict_iris(input_features):
    # input_features is list of list, [[5.1, 3.5, 1.4, 0.2]]
    columns = ["sepal length (cm)", "sepal width (cm)",
               "petal length (cm)", "petal width (cm)"]
    X = pd.DataFrame(input_features, columns=columns)
    preds = model.predict(X)
    return preds.tolist()
