import numpy as np
import json
from scipy.spatial import ConvexHull
from scipy.stats import entropy
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from typing import Dict, Any, List

class AdvancedFeatureExtractor:
    def __init__(self):
        # Pre-train Isolation Forest for Manifold Deviation (Feature 8)
        self.iso_forest = IsolationForest(n_estimators=100, random_state=42)
        # Generate 500 synthetic normal queries
        np.random.seed(42)
        v_features = np.random.randn(500, 28)
        time_features = np.random.uniform(0, 172800, (500, 1))
        amount_features = np.random.uniform(0, 500, (500, 1))
        
        # Order: Time, V1..V28, Amount (matching our Pydantic model exactly)
        self.baseline_data = np.hstack((time_features, v_features, amount_features))
        self.iso_forest.fit(self.baseline_data)

    def extract(self, session_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extracts 8 advanced mathematical features from a session record.
        session_data comes directly from the SQLite database.
        """
        query_count = session_data.get("query_count", 0)
        
        # Parse JSON fields if they are strings (SQLite returns them as strings)
        raw_payloads = session_data.get("query_payloads", "[]")
        if isinstance(raw_payloads, str):
            payloads = json.loads(raw_payloads)
        else:
            payloads = raw_payloads
            
        raw_confidences = session_data.get("confidence_scores", "[]")
        if isinstance(raw_confidences, str):
            confidences = json.loads(raw_confidences)
        else:
            confidences = raw_confidences
            
        # Convert payloads to numpy matrix
        # Keys matching our FraudPredictionRequest schema
        feature_keys = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        
        matrix = []
        for p in payloads:
            row = [p.get(k, 0.0) for k in feature_keys]
            matrix.append(row)
            
        X = np.array(matrix)
        y_conf = np.array(confidences)
        
        # Safe defaults
        features = {
            "pairwise_l2": 0.0,
            "convex_hull_volume": 0.0,
            "confidence_entropy": 0.0,
            "low_confidence_ratio": 0.0,
            "feature_space_coverage": 0.0,
            "linear_independence_score": 0.0,
            "query_rate": 0.0,
            "manifold_deviation": 0.0
        }

        if query_count == 0 or len(X) == 0:
            return features

        # 1. pairwise_l2
        if query_count >= 2:
            diffs = np.diff(X, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            features["pairwise_l2"] = float(np.mean(distances))

        # 2. convex_hull_volume
        # Needs at least num_features + 2 points.
        # Computing ConvexHull in 30D is computationally impossible (O(n^(d/2))).
        # We will use PCA to project down to 3 dimensions first, then compute the 3D volume.
        if query_count >= 5:
            try:
                pca = PCA(n_components=3)
                X_3d = pca.fit_transform(X)
                hull = ConvexHull(X_3d)
                features["convex_hull_volume"] = float(hull.volume)
            except Exception:
                features["convex_hull_volume"] = 0.0
        
        # 3. confidence_entropy
        if len(y_conf) > 0:
            # entropy function normalizes the input
            # If all scores are identical, entropy is max. If one is 1 and rest 0, entropy is 0.
            # Adding small epsilon to avoid divide by zero inside entropy
            features["confidence_entropy"] = float(entropy(y_conf + 1e-9))

        # 4. low_confidence_ratio
        if len(y_conf) > 0:
            low_conf_count = np.sum(y_conf < 0.6)
            features["low_confidence_ratio"] = float(low_conf_count / len(y_conf))

        # 5. feature_space_coverage
        # Mean standard deviation across all 30 features (or 28 if we strictly drop Time/Amount)
        # We'll calculate across all 30 since it says "entire feature range"
        # but wait, the prompt specifically says "across all 28 features".
        # V1-V28 are indices 1 through 28 (inclusive)
        if query_count >= 2:
            v_features_only = X[:, 1:29]
            std_devs = np.std(v_features_only, axis=0)
            features["feature_space_coverage"] = float(np.mean(std_devs))

        # 6. linear_independence_score
        rank = np.linalg.matrix_rank(X)
        features["linear_independence_score"] = min(float(rank / query_count), 1.0)

        # 7. query_rate
        first_seen = session_data.get("first_seen", 0)
        last_seen = session_data.get("last_seen", 0)
        delta_seconds = last_seen - first_seen
        delta_minutes = delta_seconds / 60.0
        if delta_minutes > 0:
            features["query_rate"] = float(query_count / delta_minutes)
        else:
            # If all in same instant, rate is effectively the count per 1 second extrapolated
            features["query_rate"] = float(query_count * 60.0)

        # 8. manifold_deviation
        # score_samples returns negative values (lower = more anomalous)
        # We want higher = more anomalous. So we multiply by -1.
        scores = self.iso_forest.score_samples(X)
        anomaly_score = -1.0 * np.mean(scores)
        features["manifold_deviation"] = float(anomaly_score)

        return features

# Global singleton
advanced_extractor = AdvancedFeatureExtractor()
