"""
PROOF: The detection is based on input geometry, not IP addresses.
No HTTP. No sessions. No IPs. Just raw math.
"""
import numpy as np
import sys
sys.path.insert(0, "d:\\honeypot")
from backend.gateway.defense.features import advanced_extractor
import joblib, pandas as pd, time

model = joblib.load("ml_training/xgboost_detector.pkl")
LABELS = {0: "Normal", 1: "Knockoff Nets", 2: "JBDA", 3: "Analytical", 4: "Evolutionary"}

def make_session(payloads):
    return {
        "query_count": len(payloads),
        "first_seen": time.time() - 2,
        "last_seen": time.time(),
        "query_payloads": payloads,
        "confidence_scores": [0.99] * len(payloads)  # Same confidence for all
    }

# === NORMAL USER: random inputs from a bell curve ===
normal_payloads = []
for _ in range(15):
    p = {"Time": np.random.uniform(100, 5000), "Amount": np.random.uniform(5, 50)}
    for j in range(1, 29): p[f"V{j}"] = float(np.random.randn())  # std dev = 1
    normal_payloads.append(p)

# === KNOCKOFF ATTACKER: uniform random across huge range ===
knockoff_payloads = []
for _ in range(15):
    p = {"Time": np.random.uniform(0, 172800), "Amount": np.random.uniform(0, 1000)}
    for j in range(1, 29): p[f"V{j}"] = float(np.random.uniform(-10, 10))  # range = 20
    knockoff_payloads.append(p)

# === JBDA ATTACKER: tiny perturbations from a single point ===
jbda_payloads = []
seed = {"Time": 500.0, "Amount": 100.0}
for j in range(1, 29): seed[f"V{j}"] = 0.5
for _ in range(15):
    p = {"Time": seed["Time"], "Amount": seed["Amount"]}
    for j in range(1, 29): p[f"V{j}"] = seed[f"V{j}"] + np.random.uniform(-0.01, 0.01)
    jbda_payloads.append(p)
    seed = p

print("=" * 70)
print("PROOF: Same confidence, same query count, no IPs involved.")
print("Only the INPUT GEOMETRY differs.")
print("=" * 70)

for name, payloads in [("NORMAL USER", normal_payloads), ("KNOCKOFF ATTACKER", knockoff_payloads), ("JBDA ATTACKER", jbda_payloads)]:
    session = make_session(payloads)
    features = advanced_extractor.extract(session)
    df = pd.DataFrame([features])
    pred = int(model.predict(df)[0])
    probs = model.predict_proba(df)[0]
    
    print(f"\n--- {name} ---")
    print(f"  pairwise_l2          = {features['pairwise_l2']:.4f}")
    print(f"  convex_hull_volume   = {features['convex_hull_volume']:.4f}")
    print(f"  feature_space_coverage = {features['feature_space_coverage']:.4f}")
    print(f"  manifold_deviation   = {features['manifold_deviation']:.4f}")
    print(f"  >> XGBoost says: {LABELS[pred]} (confidence: {max(probs):.4f})")
