"""
Evolutionary Boundary Sampler — Feature-Scaled Genetic Algorithm

Strategy:
1. Load per-feature mean/std from discovery phase.
2. Initialize population by sampling each feature from Normal(mean, std).
3. Fitness function: minimize |confidence - 0.5| (find the decision boundary).
4. Selection: keep the top-K fittest individuals (closest to boundary).
5. Reproduction: mutate parents with per-feature noise scaled to 0.05 * std_i.
6. Adaptive mutation: as generations progress, shrink mutation rate
   to fine-tune boundary location (simulated annealing effect).
7. Diversity injection: if population variance collapses, inject fresh
   in-distribution samples to prevent premature convergence.
"""
import httpx
import numpy as np
import pandas as pd
import sys
import os
import time
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def run_evolutionary(url, output_file):
    print(f"Starting Evolutionary Boundary Attacker against {url}...")
    ip = f"10.4.{np.random.randint(1,255)}.{np.random.randint(1,255)}"
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

    # ─── Exploration: Find initial boundary seeds ───
    boundary_seeds = []
    print("  Phase 1 — Exploration: Finding initial boundary seeds")
    
    n_explore = 200
    explore_matrix = np.random.randn(n_explore, n_features) * stds + means
    
    with httpx.Client() as client:
        for i in range(n_explore):
            payload = dict(zip(feature_names, explore_matrix[i].tolist()))
            try:
                response = client.post(url, json=payload, headers={"X-Forwarded-For": ip}, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    conf = data.get("confidence", 0.0)
                    
                    # Save EVERY query to the dataset for the final Stolen Model Evaluation
                    rows.append({
                        **payload,
                        "stolen_label": data.get("prediction", ""),
                        "stolen_confidence": conf
                    })
                    
                    if 0.2 < conf < 0.8:
                        boundary_seeds.append((explore_matrix[i], conf))
            except Exception:
                pass
            
            if (i+1) % 50 == 0:
                print(f"  Phase 1 — Exploration: {i+1}/{n_explore} | Boundary samples found: {len(boundary_seeds)}")

    pop_size = 20
    if len(boundary_seeds) < 2:
        print("  Failed to find enough boundary seeds. Falling back to random population.")
        pop = np.random.randn(pop_size, n_features) * stds + means
    else:
        # Seed population with discovered boundary points
        pop = []
        for _ in range(pop_size):
            seed = boundary_seeds[np.random.randint(len(boundary_seeds))][0]
            pop.append(seed + np.random.randn(n_features) * (stds * 0.1))
        pop = np.array(pop)

    # ─── Evolution Loop ───
    n_generations = 25
    n_elite = 5   # Keep top 5
    n_children = 3  # 3 children per parent → 5 + 15 = 20
    base_mutation_scale = 0.05  # Fraction of std used for mutation
    
    with httpx.Client() as client:
        for gen in range(n_generations):
            gen_confidences = []
            fitness_scores = []
            
            # Evaluate fitness of population
            for i in range(pop_size):
                payload = dict(zip(feature_names, pop[i].tolist()))
                try:
                    response = client.post(url, json=payload, headers={"X-Forwarded-For": ip}, timeout=5.0)
                    data = response.json()
                    conf = data.get("confidence", 0.0)
                    
                    # Save EVERY query to the dataset
                    rows.append({
                        **payload,
                        "stolen_label": data.get("prediction", ""),
                        "stolen_confidence": conf
                    })
                    
                    # Fitness is how close confidence is to 0.5 (the decision boundary)
                    fitness = -abs(conf - 0.5) 
                    fitness_scores.append((pop[i], fitness, conf))
                    gen_confidences.append(conf)
                except Exception as e:
                    print(f"Error on gen {gen}: {e}")
                    gen_confidences.append(1.0)  # Penalize failed queries

            avg_conf = np.mean(gen_confidences)
            avg_boundary_dist = np.mean([abs(c - 0.5) for c in gen_confidences])
            print(f"  Gen {gen+1:2d}/{n_generations} | Avg Conf: {avg_conf:.4f} | "
                  f"Avg |conf-0.5|: {avg_boundary_dist:.4f}")

            # ─── Diversity Check ───
            per_feature_std = np.std(pop, axis=0)
            mean_pop_std = np.mean(per_feature_std)
            mean_feature_std = np.mean(stds)
            
            if mean_pop_std < 0.01 * mean_feature_std:
                print("  [!] Population collapsed. Injecting diversity from discovered distribution...")
                pop = np.random.randn(pop_size, n_features) * stds + means
                continue  # Re-evaluate with fresh population

            # ─── Selection ───
            # Sort descending by fitness (which is negative distance to 0.5, so higher is better)
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            elites = [x[0] for x in fitness_scores[:n_elite]]

            # ─── Reproduction & Mutation ───
            new_pop = list(elites)
            
            # Decay mutation scale over generations (Simulated Annealing approach)
            decay_factor = 1.0 - (gen / n_generations)
            current_mutation_scale = base_mutation_scale * decay_factor

            for parent in elites:
                for _ in range(n_children):
                    # Mutate: parent + noise scaled by the discovered feature stds
                    noise = np.random.randn(n_features) * (stds * current_mutation_scale)
                    child = parent + noise
                    new_pop.append(child)

            pop = np.array(new_pop)[:pop_size]

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} records to {output_file}")
    return df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evolutionary_attacker.py <target_url> <output_file>")
        sys.exit(1)
    run_evolutionary(sys.argv[1], sys.argv[2])
