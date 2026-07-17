from pydantic import BaseModel, Field
from typing import Dict, Any

class GatewayPredictionResponse(BaseModel):
    prediction: str = Field(..., description="'fraud' or 'legitimate'")
    confidence: float = Field(..., description="Probability of the prediction (0.0 to 1.0)")
    session_id: str = Field(..., description="HoneyMind Session ID")
    risk_score: float = Field(..., description="Calculated Risk Score (0.0 to 1.0)")
    attack_type: str = Field(default="Normal", description="Detected attack type by XGBoost")
