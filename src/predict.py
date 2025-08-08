import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://mlflow:5000")
model_name = "sk-learn-random-forest-reg-model@production"
model_uri = f"models:/{model_name}"
model = mlflow.sklearn.load_model(model_uri)


def predict(input_data):
    predictions = model.predict(input_data)
    return predictions
