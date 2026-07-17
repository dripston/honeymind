import requests
import random
import time

GATEWAY_URL = "http://127.0.0.1:8001/api/v1/predict"

print("Starting HoneyMind Traffic Generator for Dashboard visualization...")
for i in range(100):
    # Simulate a fake "payload" to trigger the XGBoost + DRL agent
    payload = {
        "Time": random.uniform(0, 10000),
        "Amount": random.uniform(0, 100)
    }
    for j in range(1, 29):
        payload[f"V{j}"] = random.uniform(-2, 2)
        
    try:
        # Use X-Forwarded-For to simulate different IPs occasionally, but DRL agent ignores it anyway.
        headers = {"X-Forwarded-For": f"192.168.1.{i % 5}"}
        resp = requests.post(GATEWAY_URL, json=payload, headers=headers, timeout=2)
        print(f"Sent query {i+1}/100 - Response: {resp.status_code}")
    except Exception as e:
        print("Timeout:", e)
    
    time.sleep(0.5)
