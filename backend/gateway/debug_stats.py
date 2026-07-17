import numpy as np
import sys
import os
import time

sys.path.insert(0, "d:\\honeypot")
from backend.gateway.defense.features import advanced_extractor

# Simulate a live Knockoff session exactly as test_detector.py would build it
# (20 queries, each uniformly random over huge space)
payloads = []
confidences = []
for _ in range(20):
    p = {"Time": float(np.random.uniform(0, 172800)), "Amount": float(np.random.uniform(0, 1000))}
    for j in range(1, 29):
        p[f"V{j}"] = float(np.random.uniform(-10, 10))
    payloads.append(p)
    confidences.append(0.99)  # What victim API actually returns

first_seen = time.time() - 3  # 3 seconds of attack
last_seen = time.time()

session = {
    "query_count": 20,
    "first_seen": first_seen,
    "last_seen": last_seen,
    "query_payloads": payloads,
    "confidence_scores": confidences
}

features = advanced_extractor.extract(session)
print("=== LIVE KNOCKOFF FEATURES (20 queries) ===")
for k, v in features.items():
    print(f"  {k:30s} = {v:.6f}")

# Now compare with what generate_training_data.py simulates
from backend.gateway.ml_training.generate_training_data import generate_session
train_session = generate_session("knockoff")
train_features = advanced_extractor.extract(train_session)
print("\n=== TRAINING KNOCKOFF FEATURES ===")
for k, v in train_features.items():
    print(f"  {k:30s} = {v:.6f}")
