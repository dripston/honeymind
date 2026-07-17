from fastapi import APIRouter, HTTPException
from backend.victim_api.api.models import FraudPredictionRequest, FraudPredictionResponse
from backend.victim_api.ml.inference import predictor
from backend.victim_api.core.logger import logger

router = APIRouter()

@router.get("/health", summary="Health Check")
async def health_check():
    """Returns the API health status and whether the ML model is loaded."""
    model_loaded = predictor.model is not None
    status = "healthy" if model_loaded else "degraded"
    
    return {
        "status": status,
        "model_loaded": model_loaded
    }

@router.post("/predict", response_model=FraudPredictionResponse, summary="Predict Credit Card Fraud")
async def predict_fraud(request: FraudPredictionRequest):
    """
    Accepts a credit card transaction and returns a fraud prediction.
    """
    logger.info("Received prediction request")
    
    try:
        prediction, confidence = predictor.predict(request)
        
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
        
        # Append to access log for the UI terminal stream
        import os
        import datetime
        log_line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] POST /api/v1/predict - Pred: {prediction} (Conf: {confidence:.2f})\n"
        log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "victim_access.log")
        with open(log_path, "a") as f:
            f.write(log_line)

        return FraudPredictionResponse(
            prediction=prediction,
            confidence=confidence
        )
    except ValueError as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Model is not loaded. Please contact an administrator.")
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")
