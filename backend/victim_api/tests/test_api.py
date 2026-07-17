from fastapi.testclient import TestClient
from backend.victim_api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_predict_endpoint_missing_body():
    response = client.post("/api/v1/predict")
    assert response.status_code == 422 # Unprocessable Entity (missing body)

def test_predict_endpoint_valid_payload():
    # Only test if the model is loaded
    health = client.get("/api/v1/health").json()
    if not health.get("model_loaded"):
        return # Skip test if model isn't trained yet
        
    payload = {
        "Time": 0.0,
        "V1": -1.3598071336738, "V2": -0.0727811733098497, "V3": 2.53634673796914,
        "V4": 1.37815522427443, "V5": -0.338320769942518, "V6": 0.462387777762292,
        "V7": 0.239598554061257, "V8": 0.0986979012610507, "V9": 0.363786969611213,
        "V10": 0.0907941719789316, "V11": -0.551599533260813, "V12": -0.617800855762348,
        "V13": -0.991389847235408, "V14": -0.311169353699879, "V15": 1.46817697209427,
        "V16": -0.470400525259478, "V17": 0.207971241929242, "V18": 0.0257905801985591,
        "V19": 0.403992960255733, "V20": 0.251412098239705, "V21": -0.018306777944153,
        "V22": 0.277837575558899, "V23": -0.110473910188767, "V24": 0.0669280749146731,
        "V25": 0.128539358273528, "V26": -0.189114843888824, "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Amount": 149.62
    }
    
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert data["prediction"] in ["fraud", "legitimate"]
    assert 0.0 <= data["confidence"] <= 1.0
