import uuid
import time
from typing import Dict, Any

class SessionManager:
    """
    In-memory session tracker.
    Later phases will replace this with Redis.
    """
    def __init__(self):
        # Maps session_id -> Session State Dict
        self._sessions: Dict[str, Dict[str, Any]] = {}
        # Maps IP -> session_id (simplistic for now)
        self._ip_to_session: Dict[str, str] = {}

    def get_or_create_session(self, client_ip: str) -> str:
        if client_ip in self._ip_to_session:
            session_id = self._ip_to_session[client_ip]
            return session_id
        
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "session_id": session_id,
            "client_ip": client_ip,
            "created_at": time.time(),
            "request_count": 0,
            "requests": []  # Store full request history for feature extraction
        }
        self._ip_to_session[client_ip] = session_id
        return session_id

    def track_request(self, session_id: str, payload_hash: str, prediction: str, confidence: float):
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session["request_count"] += 1
            session["requests"].append({
                "timestamp": time.time(),
                "payload_hash": payload_hash,
                "prediction": prediction,
                "confidence": confidence
            })

    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        return self._sessions.get(session_id, {})

# Global instance
session_store = SessionManager()
