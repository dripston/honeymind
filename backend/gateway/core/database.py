import sqlite3
import json
import uuid
import time
import threading
from typing import Dict, Any, Optional

DB_PATH = "honeymind.db"
db_lock = threading.Lock()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                cluster_id TEXT UNIQUE,
                first_seen REAL,
                last_seen REAL,
                query_count INTEGER,
                query_payloads TEXT,
                confidence_scores TEXT,
                time_deltas TEXT,
                poisoned_count INTEGER DEFAULT 0,
                deception_strategy TEXT,
                attack_type TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS global_stream (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                payload TEXT
            )
        ''')
        
        # Handle migration if table already exists without the new columns
        try:
            cursor.execute("ALTER TABLE sessions RENAME COLUMN ip_address TO cluster_id")
        except sqlite3.OperationalError:
            pass
        
        try:
            cursor.execute("ALTER TABLE sessions ADD COLUMN poisoned_count INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE sessions ADD COLUMN deception_strategy TEXT")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE sessions ADD COLUMN attack_type TEXT")
        except sqlite3.OperationalError:
            pass
        
        conn.commit()
        conn.close()

def insert_global_stream(payload: Dict[str, Any]):
    # Keep only numeric features
    numeric_payload = {k: v for k, v in payload.items() if isinstance(v, (int, float))}
    now = time.time()
    
    # We don't use db_lock here heavily to avoid blocking the main proxy, 
    # but sqlite needs some safety. We use the lock.
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO global_stream (timestamp, payload) VALUES (?, ?)
        ''', (now, json.dumps(numeric_payload)))
        
        # Cleanup old entries (keep last 1000)
        cursor.execute('''
            DELETE FROM global_stream WHERE id NOT IN (
                SELECT id FROM global_stream ORDER BY id DESC LIMIT 1000
            )
        ''')
        conn.commit()
        conn.close()

def check_global_scraping() -> bool:
    """
    Checks if the global stream is a uniform/independent scraping attack.
    If the determinant of the correlation matrix is near 1 (all features independent),
    it flags the attack.
    """
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT payload FROM global_stream ORDER BY id DESC LIMIT 500")
        rows = cursor.fetchall()
        conn.close()
        
    if len(rows) < 100:
        return False
        
    import pandas as pd
    import numpy as np
    
    data = [json.loads(row["payload"]) for row in rows]
    df = pd.DataFrame(data)
    
    # Calculate correlation matrix
    try:
        corr = df.corr().fillna(0).values
        # The determinant of a correlation matrix for truly independent variables is 1.0.
        # Real-world data typically has strong correlations, dropping the determinant near 0.
        det = np.linalg.det(corr)
        return det > 0.8  # If highly independent, it's Analytical Solver
    except Exception:
        return False

def get_session_by_cluster(cluster_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE cluster_id = ?", (cluster_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None

def create_session(cluster_id: str) -> str:
    session_id = str(uuid.uuid4())
    now = time.time()
    
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sessions (
                session_id, cluster_id, first_seen, last_seen, query_count, 
                query_payloads, confidence_scores, time_deltas
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, cluster_id, now, now, 0, "[]", "[]", "[]"))
        conn.commit()
        conn.close()
    return session_id

def update_session(session_id: str, payload: Dict[str, Any], confidence: float):
    now = time.time()
    
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get current state
        cursor.execute("SELECT last_seen, query_count, query_payloads, confidence_scores, time_deltas FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return

        last_seen = row["last_seen"]
        query_count = row["query_count"]
        
        # Parse JSON arrays and keep only the last 150 to prevent O(N^2) slowdowns
        query_payloads = json.loads(row["query_payloads"])[-150:]
        confidence_scores = json.loads(row["confidence_scores"])[-150:]
        time_deltas = json.loads(row["time_deltas"])[-150:]
        
        # Calculate delta
        time_delta = now - last_seen
        
        # Append new data
        query_payloads.append(payload)
        confidence_scores.append(confidence)
        time_deltas.append(time_delta)
        
        # Update DB
        cursor.execute('''
            UPDATE sessions
            SET last_seen = ?,
                query_count = ?,
                query_payloads = ?,
                confidence_scores = ?,
                time_deltas = ?
            WHERE session_id = ?
        ''', (
            now,
            query_count + 1,
            json.dumps(query_payloads),
            json.dumps(confidence_scores),
            json.dumps(time_deltas),
            session_id
        ))
        
        conn.commit()
        conn.close()

def increment_poisoned(session_id: str, strategy: str, attack_type: str):
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE sessions
            SET poisoned_count = poisoned_count + 1,
                deception_strategy = ?,
                attack_type = ?
            WHERE session_id = ?
        ''', (strategy, attack_type, session_id))
        conn.commit()
        conn.close()

def get_stats() -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # total sessions
    cursor.execute("SELECT COUNT(*) as cnt FROM sessions")
    total_sessions = cursor.fetchone()["cnt"]
    
    # attack sessions
    cursor.execute("SELECT COUNT(*) as cnt FROM sessions WHERE poisoned_count > 0")
    attack_sessions = cursor.fetchone()["cnt"]
    
    # poisoned responses
    cursor.execute("SELECT SUM(poisoned_count) as total_poisoned FROM sessions")
    row = cursor.fetchone()
    poisoned_responses = row["total_poisoned"] if row["total_poisoned"] else 0
    
    # attack breakdown
    cursor.execute("SELECT attack_type, COUNT(*) as cnt FROM sessions WHERE attack_type IS NOT NULL GROUP BY attack_type")
    breakdown_rows = cursor.fetchall()
    attack_breakdown = {r["attack_type"]: r["cnt"] for r in breakdown_rows}
    
    conn.close()
    
    return {
        "total_sessions": total_sessions,
        "attack_sessions": attack_sessions,
        "poisoned_responses": poisoned_responses,
        "attack_breakdown": attack_breakdown
    }

# Initialize DB on import
init_db()
