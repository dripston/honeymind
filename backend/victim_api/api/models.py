from pydantic import BaseModel, Field
from typing import Optional

class FraudPredictionRequest(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    V29: float
    V30: float

class FraudPredictionResponse(BaseModel):
    prediction: str = Field(..., description="'fraud' or 'legitimate'")
    confidence: float = Field(..., description="Probability of the prediction (0.0 to 1.0)")
