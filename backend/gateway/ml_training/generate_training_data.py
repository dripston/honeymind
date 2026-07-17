import numpy as np
import pandas as pd
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from backend.gateway.defense.features import advanced_extractor

def generate_session(scenario_type: str) -> dict:
    payloads = []
    confidences = []
    
    first_seen = time.time()
    
    if scenario_type == "normal":
        # 3-15 queries, realistic values, high confidence, spread over minutes
        query_count = np.random.randint(3, 15)
        duration = np.random.uniform(60, 3600)  # 1 min to 1 hour
        last_seen = first_seen + duration
        for _ in range(query_count):
            p = {"Time": np.random.uniform(100, 5000), "Amount": np.random.uniform(5, 50)}
            for i in range(1, 29):
                p[f"V{i}"] = float(np.random.randn())
            payloads.append(p)
            confidences.append(np.random.uniform(0.85, 0.99))
            
    elif scenario_type == "knockoff":
        # 15-80 queries uniformly sampled across huge feature space, fast
        query_count = np.random.randint(15, 80)
        duration = query_count * np.random.uniform(0.05, 0.5)  # Very fast, 50ms-500ms per query
        last_seen = first_seen + duration
        for _ in range(query_count):
            p = {"Time": np.random.uniform(0, 172800), "Amount": np.random.uniform(0, 1000)}
            for i in range(1, 29):
                p[f"V{i}"] = float(np.random.uniform(-10, 10))
            payloads.append(p)
            # The REAL victim API returns high confidence even on garbage inputs
            # So we simulate realistic victim responses
            confidences.append(np.random.uniform(0.7, 0.99))
            
    elif scenario_type == "jbda":
        # 10-40 queries, each a tiny perturbation from a seed
        query_count = np.random.randint(10, 40)
        duration = query_count * np.random.uniform(0.05, 0.5)
        last_seen = first_seen + duration
        # Random seed point
        current_p = {"Time": np.random.uniform(0, 5000), "Amount": np.random.uniform(10, 200)}
        for i in range(1, 29): current_p[f"V{i}"] = float(np.random.randn())
            
        for _ in range(query_count):
            new_p = {"Time": current_p["Time"], "Amount": current_p["Amount"]}
            for i in range(1, 29):
                new_p[f"V{i}"] = current_p[f"V{i}"] + np.random.uniform(-0.01, 0.01)
            payloads.append(new_p)
            current_p = new_p
            # Real victim API is confident even on boundary samples
            confidences.append(np.random.uniform(0.7, 0.99))
            
    elif scenario_type == "analytical":
        # Exactly 29-31 queries designed to be linearly independent
        query_count = np.random.choice([29, 30, 31])
        duration = query_count * np.random.uniform(0.05, 0.3)
        last_seen = first_seen + duration
        for idx in range(query_count):
            p = {"Time": 1.0 if idx == 0 else 0.0, "Amount": 0.0}
            for i in range(1, 29):
                p[f"V{i}"] = 1.0 if idx == i else 0.0
            payloads.append(p)
            confidences.append(np.random.uniform(0.7, 0.99))
            
    elif scenario_type == "evolutionary":
        # 15-50 queries, spread across feature space but more clustered than knockoff
        query_count = np.random.randint(15, 50)
        duration = query_count * np.random.uniform(0.1, 1.0)
        last_seen = first_seen + duration
        for _ in range(query_count):
            p = {"Time": np.random.uniform(100, 5000), "Amount": np.random.uniform(5, 50)}
            for i in range(1, 29):
                p[f"V{i}"] = float(np.random.randn() * 2)
            payloads.append(p)
            confidences.append(np.random.uniform(0.7, 0.99))
            
    return {
        "query_count": query_count,
        "first_seen": first_seen,
        "last_seen": last_seen,
        "query_payloads": payloads,
        "confidence_scores": confidences
    }

def main():
    print("Generating training data...")
    scenarios = [
        ("normal", 500, 0),
        ("knockoff", 300, 1),
        ("jbda", 300, 2),
        ("analytical", 200, 3),
        ("evolutionary", 200, 4)
    ]
    
    rows = []
    
    for scenario_type, count, label in scenarios:
        print(f"Simulating {count} {scenario_type} sessions...")
        for _ in range(count):
            session = generate_session(scenario_type)
            features = advanced_extractor.extract(session)
            
            row = features.copy()
            row["label"] = label
            rows.append(row)
            
    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(__file__), "training_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} records to {out_path}")
    
    # Print feature stats for verification
    print("\n=== FEATURE STATS ===")
    for label, name in [(0, "Normal"), (1, "Knockoff"), (2, "JBDA"), (3, "Analytical"), (4, "Evolutionary")]:
        subset = df[df["label"] == label]
        print(f"\n--- {name} ---")
        for col in [c for c in df.columns if c != "label"]:
            print(f"  {col:30s} mean={subset[col].mean():12.4f}  min={subset[col].min():12.4f}  max={subset[col].max():12.4f}")

if __name__ == "__main__":
    main()
