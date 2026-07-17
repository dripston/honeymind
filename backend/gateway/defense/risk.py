from typing import Dict

class RiskEngine:
    """
    Evaluates risk based on extracted features.
    In Phase 3, this will be replaced by the XGBoost Threat Detector.
    """
    
    @staticmethod
    def evaluate_risk(features: Dict[str, float]) -> float:
        """
        Simple heuristic risk scoring.
        Formula: risk_score = min(request_count / 100, 1.0)
        """
        req_count = features.get("request_count", 0.0)
        
        # Naive approach: if you make 100+ requests, you're 100% suspicious
        risk_score = req_count / 100.0
        
        return min(risk_score, 1.0)

risk_engine = RiskEngine()
