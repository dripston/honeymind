from fastapi.testclient import TestClient
from backend.gateway.main import app
import pytest
from unittest.mock import patch, MagicMock

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "HoneyMind Gateway"}

@patch("httpx.AsyncClient.post")
def test_predict_proxy_success(mock_post):
    # Mock the Victim API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"prediction": "legitimate", "confidence": 0.98}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    payload = {
        "Time": 0.0, "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, 
        "V5": 0.0, "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, 
        "V10": 0.0, "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, 
        "V15": 0.0, "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, 
        "V20": 0.0, "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, 
        "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0, "Amount": 100.0
    }
    
    response = client.post("/api/v1/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "legitimate"
    assert data["confidence"] == 0.98
    assert "session_id" in data
    assert "risk_score" in data

@patch("httpx.AsyncClient.post")
def test_risk_score_increases(mock_post):
    # Mock the Victim API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"prediction": "fraud", "confidence": 0.90}
    mock_post.return_value = mock_response
    
    payload = {
        "Time": 0.0, "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, 
        "V5": 0.0, "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, 
        "V10": 0.0, "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, 
        "V15": 0.0, "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, 
        "V20": 0.0, "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, 
        "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0, "Amount": 100.0
    }
    
    # Send a few requests to simulate activity
    scores = []
    for _ in range(5):
        resp = client.post("/api/v1/predict", json=payload)
        scores.append(resp.json()["risk_score"])
        
    # Check that risk score increases as request count goes up
    # It should be 0.01, 0.02, 0.03, 0.04, 0.05
    assert scores == sorted(scores)
    assert scores[-1] > scores[0]
