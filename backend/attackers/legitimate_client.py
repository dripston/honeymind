import httpx
import pandas as pd
import sys
import os
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def run_legitimate_client(url):
    print(f"Starting Legitimate Customer Traffic Simulator against {url}...\n")
    
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "..", "dataset.csv")
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print("Error: Could not find dataset.csv.")
        return

    # Simulate 100 legitimate queries
    features = [c for c in df.columns if c not in ["Class", "Time", "Amount"]]
    df_sample = df.sample(n=100, replace=True)
    queries = df_sample[features].to_dict(orient='records')

    success_count = 0
    with httpx.Client() as client:
        for i, p in enumerate(queries):
            ip = f"203.0.113.{random.randint(1, 255)}" # Realistic dynamic IPs
            try:
                start_time = time.time()
                response = client.post(url, json=p, headers={"X-Forwarded-For": ip}, timeout=5.0)
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    success_count += 1
                    status = "200 OK"
                else:
                    status = f"{response.status_code} ERR"

                print(f"[{time.strftime('%H:%M:%S')}] GET /api/v1/predict | IP: {ip:15s} | Latency: {latency:4.0f}ms | {status}")
                
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] GET /api/v1/predict | IP: {ip:15s} | ERROR: {str(e)}")

            time.sleep(random.uniform(0.05, 0.2)) # Simulate human latency

    print("\n" + "=" * 60)
    print(f"TRAFFIC SIMULATION COMPLETE")
    print(f"Total Requests: 100")
    print(f"Successful Responses: {success_count} ({success_count/100*100:.0f}%)")
    if success_count >= 95:
        print("\n[VERIFIED] HoneyMind Gateway DOES NOT impact legitimate traffic!")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python legitimate_client.py <target_url>")
        sys.exit(1)
    run_legitimate_client(sys.argv[1])
