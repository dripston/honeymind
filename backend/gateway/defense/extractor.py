import hashlib
import json
from typing import Dict, Any

class FeatureExtractor:
    """
    Extracts behavioral signals from the current request and session history.
    """
    
    @staticmethod
    def hash_payload(payload_dict: Dict[str, Any]) -> str:
        """Simple hash to track unique inputs."""
        payload_str = json.dumps(payload_dict, sort_keys=True)
        return hashlib.md5(payload_str.encode('utf-8')).hexdigest()

    @staticmethod
    def extract_features(session_data: Dict[str, Any], current_payload: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculates extraction features.
        Currently returns simple metrics. Future phases will add ML-driven feature extraction.
        """
        request_count = session_data.get("request_count", 0)
        
        # Placeholders for Phase 3 features
        features = {
            "request_count": float(request_count),
            "query_frequency": 0.0,       # placeholder
            "input_diversity": 0.0,       # placeholder
            "request_similarity": 0.0,    # placeholder
            "entropy_score": 0.0,         # placeholder
            "confidence_probing": 0.0     # placeholder
        }
        
        # Basic RPM (Requests Per Minute) calculation
        if request_count > 0 and "created_at" in session_data:
            import time
            elapsed_minutes = (time.time() - session_data["created_at"]) / 60.0
            if elapsed_minutes > 0:
                features["query_frequency"] = request_count / elapsed_minutes
                
        return features

extractor = FeatureExtractor()
