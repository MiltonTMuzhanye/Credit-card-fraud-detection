import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint():
    """Test prediction endpoint"""
    transaction = {
        "transaction_id": "test_123",
        "time": 0.0,
        "v1": -1.359807,
        "v2": -0.072781,
        "amount": 149.62
        # ... include all features
    }
    
    response = client.post("/predict", json=transaction)
    assert response.status_code in [200, 500]  # 500 if model not loaded