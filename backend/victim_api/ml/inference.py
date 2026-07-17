import os
import joblib
import pandas as pd
from backend.victim_api.core.logger import logger
from backend.victim_api.api.models import FraudPredictionRequest
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")

class FraudModelPredictor:
    def __init__(self):
        self.model = None

    def load_model(self):
        try:
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model file not found at {MODEL_PATH}. Please run train.py first.")
                return False
            
            logger.info("Loading victim fraud model...")
            self.model = joblib.load(MODEL_PATH)
            logger.info("Victim fraud model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

    def predict(self, request: FraudPredictionRequest):
        if self.model is None:
            raise ValueError("Model is not loaded.")
        
        # Convert request to DataFrame (ensure feature order matches training data)
        feature_names = [f"V{i}" for i in range(1, 31)]
        req_dict = request.model_dump()
        
        # Create a single-row dataframe
        df = pd.DataFrame([req_dict], columns=feature_names)
        
        # Predict class and probabilities
        pred_class = self.model.predict(df)[0]
        probabilities = self.model.predict_proba(df)[0]
        
        # Map 0 to legitimate and 1 to fraud to match the Kaggle dataset
        prediction_label = "legitimate" if pred_class == 0 else "fraud"
        confidence = float(probabilities[pred_class])
        
        return prediction_label, confidence

# Global instance
predictor = FraudModelPredictor()
