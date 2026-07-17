# Victim API: Credit Card Fraud Detection

This is the foundational Target API for the HoneyMind project. It exposes a simple `POST /predict` endpoint that takes credit card transaction features and returns a fraud prediction.

In later phases, HoneyMind will intercept traffic to this API to detect and deceive attackers attempting model extraction or membership inference.

## Quick Start

### 1. Setup Environment
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model
You must train the model before running the API, otherwise the `/predict` endpoint will return 503 errors.

```powershell
# Add the project root to PYTHONPATH so imports work
$env:PYTHONPATH = "..\.."
python ml/train.py
```
This will generate a synthetic dataset mirroring the Kaggle Credit Card Fraud schema and save a `fraud_model.joblib` artifact.

### 3. Run the Server
```powershell
$env:PYTHONPATH = "..\.."
uvicorn backend.victim_api.main:app --reload
```
The API will be available at: http://127.0.0.1:8000/docs (Swagger UI)

### 4. Run Tests
```powershell
$env:PYTHONPATH = "..\.."
pytest tests/
```

## Architecture Note
Currently, this is a direct Client -> API relationship. When HoneyMind is implemented, the architecture will become:
Client -> HoneyMind Gateway (Deception/Monitoring) -> Victim API.
