from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_predict():
    response = client.post("/predict", json={"data": [[0.1]]})
    assert response.status_code == 200
    assert "predictions" in response.json()
