"""
Knockoff Nets Attacker — Generic Gaussian Sampling + Active Learning

Strategy:
1. Load per-feature mean/std from discovery phase (victim_info.json).
2. Generate initial queries by sampling each feature from Normal(mean, std).
3. Active Learning: adaptively oversample the decision boundary zone.
   - Queries that return confidence near 0.5 are the most valuable
     because they sit on the boundary.
   - Mutate boundary samples to generate more queries in that region.
   - This maximizes information gained per query and produces a
     stolen model that closely mirrors the victim's decision surface.
"""
import httpx
import numpy as np
import pandas as pd
import sys
import os
import time
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def run_knockoff(url, output_file):
    print(f"Starting Knockoff Nets Attacker against {url}...")
    ip = f"10.1.{np.random.randint(1,255)}.{np.random.randint(1,255)}"
    rows = []

    # Load dynamically discovered intelligence
    info_path = os.path.join(os.path.dirname(__file__), "victim_info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    feature_names = info["feature_names"]
    stats = info["feature_stats"]

    # Build per-feature mean and std vectors
    means = np.array([stats[f]["mean"] for f in feature_names])
    stds = np.array([stats[f]["std"] for f in feature_names])

    n_total = 500
    n_initial = 250  # Phase 1: broad exploration
    n_active = n_total - n_initial  # Phase 2: active learning near boundary

    # ─── Phase 1: Broad Gaussian Exploration ───
    # Sample each feature independently from Normal(mean, std)
    X_initial = np.random.randn(n_initial, len(feature_names)) * stds + means
    queries = pd.DataFrame(X_initial, columns=feature_names).to_dict(orient='records')

    boundary_samples = []  # Samples near the decision boundary (conf ~0.5)

    with httpx.Client() as client:
        # Phase 1: Explore the full distribution
        for i, p in enumerate(queries):
            try:
                response = client.post(url, json=p, headers={"X-Forwarded-For": ip}, timeout=5.0)
                data = response.json()

                row = p.copy()
                row["stolen_label"] = data.get("prediction", "")
                conf = data.get("confidence", 0.5)
                row["stolen_confidence"] = conf
                rows.append(row)

                # Track boundary samples (confidence close to 0.5 = near decision boundary)
                if abs(conf - 0.5) < 0.2:
                    boundary_samples.append(np.array([p[f] for f in feature_names]))
            except Exception as e:
                print(f"Error on query {i}: {e}")

            if (i + 1) % 50 == 0:
                print(f"  Phase 1 — Exploration: {i+1}/{n_initial} | "
                      f"Boundary samples found: {len(boundary_samples)}")

        # ─── Phase 2: Active Learning — Oversample Decision Boundary ───
        # Mutate boundary samples to explore the boundary region more densely.
        # If no boundary samples found, fall back to more Gaussian exploration.
        print(f"\n  Phase 2 — Active Learning ({n_active} queries)")
        print(f"  Boundary seeds available: {len(boundary_samples)}")

        for i in range(n_active):
            if len(boundary_samples) > 0:
                # Pick a random boundary sample and mutate it slightly
                seed_idx = np.random.randint(len(boundary_samples))
                seed = boundary_samples[seed_idx].copy()
                # Small mutation: 10% of each feature's std
                mutation = np.random.randn(len(feature_names)) * (stds * 0.10)
                query_vec = seed + mutation
            else:
                # Fallback: sample from the discovered distribution
                query_vec = np.random.randn(len(feature_names)) * stds + means

            p = dict(zip(feature_names, query_vec.tolist()))
            try:
                response = client.post(url, json=p, headers={"X-Forwarded-For": ip}, timeout=5.0)
                data = response.json()

                row = p.copy()
                row["stolen_label"] = data.get("prediction", "")
                conf = data.get("confidence", 0.5)
                row["stolen_confidence"] = conf
                rows.append(row)

                # Expand boundary pool
                if abs(conf - 0.5) < 0.2:
                    boundary_samples.append(query_vec)
            except Exception as e:
                print(f"Error on active query {i}: {e}")

            if (i + 1) % 50 == 0:
                print(f"  Phase 2 — Active Learning: {i+1}/{n_active} | "
                      f"Boundary pool: {len(boundary_samples)}")

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} records to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python knockoff_attacker.py <target_url> <output_file>")
        sys.exit(1)
    run_knockoff(sys.argv[1], sys.argv[2])
