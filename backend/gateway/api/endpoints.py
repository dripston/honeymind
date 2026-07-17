from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool
import httpx
import joblib
import pandas as pd
import os

from backend.gateway.api.models import GatewayPredictionResponse
from backend.gateway.core import database
from backend.gateway.defense.features import advanced_extractor
from backend.gateway.core.logger import logger

router = APIRouter()

# Global HTTP client for connection pooling to prevent socket exhaustion
proxy_client = httpx.AsyncClient()

# Load XGBoost model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_training", "xgboost_detector.pkl")
try:
    xgboost_model = joblib.load(MODEL_PATH)
    logger.info("Loaded XGBoost Threat Detector.")
except Exception as e:
    logger.warning(f"Failed to load XGBoost model: {e}. Will fallback to Normal.")
    xgboost_model = None

ATTACK_LABELS = {
    0: "Normal",
    1: "Knockoff Nets",
    2: "JBDA",
    3: "Analytical Solver",
    4: "Evolutionary Boundary"
}

# In a real app, this would come from env vars
VICTIM_API_URL = "http://127.0.0.1:8000/api/v1/predict"

@router.get("/health", summary="Gateway Health Check")
async def health_check():
    return {"status": "healthy", "service": "HoneyMind Gateway"}

@router.post("/predict", response_model=GatewayPredictionResponse, summary="Intercepted Fraud Prediction")
async def predict_proxy(request: Request):
    """
    Acts as a middleware. Intercepts the request, extracts features, 
    evaluates risk, and proxies to the Victim API.
    """
    # 1. Parse Payload
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # 2. Spatial Grid Hashing (Replaces IP Tracking)
    # We round numerical features to 1 decimal place to create a coarse spatial grid.
    # All queries that are spatially close will organically fall into the same cluster hash,
    # completely neutralizing IP-spoofing by attackers like Knockoff Nets and Evolutionary.
    numeric_values = [round(float(v), 1) for k, v in payload.items() if isinstance(v, (int, float))]
    cluster_id = str(hash(tuple(numeric_values)))
    
    # 3. Add to Global Stream for Covariance Tracking
    await run_in_threadpool(database.insert_global_stream, payload)

    # 4. Session Tracking via Spatial Cluster
    session_data = await run_in_threadpool(database.get_session_by_cluster, cluster_id)
    if not session_data:
        session_id = await run_in_threadpool(database.create_session, cluster_id)
        session_data = await run_in_threadpool(database.get_session_by_cluster, cluster_id)
    else:
        session_id = session_data["session_id"]
    
    # 4. Proxy to Victim API
    try:
        victim_response = await proxy_client.post(VICTIM_API_URL, json=payload, timeout=10.0)
        victim_response.raise_for_status()
        victim_data = victim_response.json()
    except httpx.RequestError as e:
        logger.error(f"Error connecting to Victim API: {e}")
        raise HTTPException(status_code=502, detail="Victim API unavailable")
    except httpx.HTTPStatusError as e:
        logger.error(f"Victim API returned an error: {e.response.status_code}")
        try:
            error_detail = e.response.json().get("detail", "Victim API error")
        except Exception:
            error_detail = "Victim API error"
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)

    prediction = victim_data.get("prediction", "unknown")
    confidence = victim_data.get("confidence", 0.0)

    # 5. Track the Request in SQLite
    await run_in_threadpool(database.update_session, session_id, payload, confidence)
    
    # Fetch updated session data for feature extraction
    session_data = await run_in_threadpool(database.get_session_by_cluster, cluster_id)

    # 6. Advanced Feature Extraction
    features = await run_in_threadpool(advanced_extractor.extract, session_data)
    
    # 7. Threat Detection (Local Cluster & Global Stream)
    attack_type = "Normal"
    risk_score = 0.0
    
    is_global_scraping = await run_in_threadpool(database.check_global_scraping)
    
    if is_global_scraping:
        risk_score = 1.0
        attack_type = "Distributed Scraping (Analytical Solver)"
    elif xgboost_model:
        # Pass all 8 features to XGBoost
        df_features = pd.DataFrame([features])
        
        # XGBoost predict_proba returns [[prob_class0, prob_class1]]
        probs = xgboost_model.predict_proba(df_features)[0]
        pred_class = int(xgboost_model.predict(df_features)[0])
        
        # The model predicts the attack type class index (0-4)
        attack_type = ATTACK_LABELS.get(pred_class, "Normal")
        risk_score = float(max(probs))
        
        # If prediction is Normal, risk_score is technically high confidence of being normal.
        # So we should invert it or just use probability of non-normal if we want risk.
        # But the prompt says: "risk_score (max class probability)" so I will just return max(probs).
        # Actually, if pred_class == 0, risk score should probably be 1.0 - probs[0]. 
        if pred_class == 0:
            risk_score = float(1.0 - probs[0])



    # 8. Log Telemetry
    if attack_type != "Normal" and risk_score > 0.5:
        logger.info(f"  [OOD Detector] Cluster {cluster_id} flagged ({risk_score*100:.1f}% confidence). Route: HoneyMind Poisoning. ({attack_type})")
    elif xgboost_model and session_data.get("query_count", 0) >= 5:
        logger.info(f"  [OOD Detector] Cluster {cluster_id} verified as In-Distribution. Route: Legitimate.")
        
    logger.info(
        f"Session: {session_id} | Cluster: {cluster_id} | Req Count: {session_data.get('query_count')} | "
        f"Attack: {attack_type} | Risk: {risk_score:.2f} | Prediction: {prediction} | Confidence: {confidence:.4f}"
    )

    # Apply Deception Engine Phase 6 (Only if toggle is ON)
    from backend.gateway.api.control_endpoints import global_state
    if not global_state.get("HONEYMIND_ENABLED", True):
        return GatewayPredictionResponse(
            prediction=prediction,
            confidence=confidence,
            session_id=session_id,
            risk_score=risk_score,
            attack_type=attack_type
        )

    from backend.gateway.defense.deception import deception_engine
    
    # Calculate a deterministic hash of the payload for decoy boundaries
    payload_hash = sum(abs(v) for v in payload.values() if isinstance(v, (int, float)))
    
    final_prediction, final_confidence, strategy = deception_engine.apply(
        attack_type=attack_type,
        risk_score=risk_score,
        prediction=prediction,
        confidence=confidence,
        session_id=session_id,
        query_count=session_data.get('query_count', 0),
        payload_hash=payload_hash,
        payload=payload
    )
    
    if strategy:
        await run_in_threadpool(database.increment_poisoned, session_id, strategy, attack_type)

    # 9. Return Decorated Response
    return GatewayPredictionResponse(
        prediction=final_prediction,
        confidence=final_confidence,
        session_id=session_id,
        risk_score=risk_score,
        attack_type=attack_type
    )

@router.get("/stats", summary="Get Gateway Statistics")
async def get_stats():
    return database.get_stats()
