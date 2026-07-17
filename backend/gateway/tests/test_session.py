import os
import sys
import json
import sqlite3

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.gateway.core import database
from backend.gateway.core.database import DB_PATH

def simulate_ip_requests(ip: str, num_requests: int):
    print(f"\n--- Simulating {num_requests} requests for IP: {ip} ---")
    
    # Ensure session exists
    session_data = database.get_session_by_ip(ip)
    if not session_data:
        session_id = database.create_session(ip)
    else:
        session_id = session_data["session_id"]
        
    for i in range(num_requests):
        payload = {"Time": i, "Amount": 100 + i, "V1": 0.1 * i}
        confidence = 0.90 + (0.01 * i)
        database.update_session(session_id, payload, confidence)
        import time
        time.sleep(0.1) # Sleep to see time_deltas > 0

def print_database_records():
    print("\n=== DATABASE RECORDS ===")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions")
    
    for row in cursor.fetchall():
        d = dict(row)
        print(f"\nSession ID: {d['session_id']}")
        print(f"IP Address: {d['ip_address']}")
        print(f"First Seen: {d['first_seen']}")
        print(f"Last Seen:  {d['last_seen']}")
        print(f"Query Count: {d['query_count']}")
        
        # Pretty print JSON arrays
        print(f"Payloads: {len(json.loads(d['query_payloads']))} items")
        print(f"Confidences: {d['confidence_scores']}")
        print(f"Time Deltas: {d['time_deltas']}")
    
    conn.close()

if __name__ == "__main__":
    # Ensure a fresh db for the test
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    database.init_db()

    # Simulate
    simulate_ip_requests("192.168.1.100", 3)
    simulate_ip_requests("10.0.0.5", 2)
    simulate_ip_requests("172.16.0.2", 4)
    
    # Print results
    print_database_records()
