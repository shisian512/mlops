from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={"data": [[0.1,0.2,0.3,0.4]]})
    assert response.status_code == 200
    assert "predictions" in response.json()
