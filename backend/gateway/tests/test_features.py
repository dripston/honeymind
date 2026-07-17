import numpy as np
import time

from backend.gateway.defense.features import advanced_extractor

def generate_normal_user():
    # 10 queries, realistic values, high confidence
    payloads = []
    confidences = []
    np.random.seed(10)
    for _ in range(10):
        # Time and Amount
        p = {"Time": np.random.uniform(100, 5000), "Amount": np.random.uniform(5, 50)}
        # V1-V28 standard normal
        for i in range(1, 29):
            p[f"V{i}"] = float(np.random.randn())
        payloads.append(p)
        confidences.append(np.random.uniform(0.9, 0.99))
        
    return {
        "query_count": 10,
        "first_seen": time.time(),
        "last_seen": time.time() + 600, # 10 minutes spread
        "query_payloads": payloads,
        "confidence_scores": confidences
    }

def generate_knockoff_attacker():
    # 50 queries uniformly sampled across huge feature space
    payloads = []
    confidences = []
    np.random.seed(20)
    for _ in range(50):
        p = {"Time": np.random.uniform(0, 172800), "Amount": np.random.uniform(0, 1000)}
        for i in range(1, 29):
            p[f"V{i}"] = float(np.random.uniform(-10, 10)) # Uniformly spread
        payloads.append(p)
        confidences.append(np.random.uniform(0.1, 0.9)) # Random responses
        
    return {
        "query_count": 50,
        "first_seen": time.time(),
        "last_seen": time.time() + 10, # Very fast
        "query_payloads": payloads,
        "confidence_scores": confidences
    }

def generate_jbda_attacker():
    # 30 queries, each a tiny perturbation
    payloads = []
    confidences = []
    np.random.seed(30)
    
    # Starting point
    current_p = {"Time": 0.0, "Amount": 100.0}
    for i in range(1, 29):
        current_p[f"V{i}"] = 0.5
        
    for _ in range(30):
        # Tiny perturbation (epsilon)
        new_p = {"Time": current_p["Time"], "Amount": current_p["Amount"]}
        for i in range(1, 29):
            new_p[f"V{i}"] = current_p[f"V{i}"] + np.random.uniform(-0.01, 0.01)
        
        payloads.append(new_p)
        current_p = new_p
        
        # Searching decision boundary (around 0.5)
        confidences.append(np.random.uniform(0.4, 0.6))
        
    return {
        "query_count": 30,
        "first_seen": time.time(),
        "last_seen": time.time() + 30, # 1 sec per query
        "query_payloads": payloads,
        "confidence_scores": confidences
    }

def print_comparison():
    normal = generate_normal_user()
    knockoff = generate_knockoff_attacker()
    jbda = generate_jbda_attacker()
    
    f_normal = advanced_extractor.extract(normal)
    f_knock = advanced_extractor.extract(knockoff)
    f_jbda = advanced_extractor.extract(jbda)
    
    print(f"{'Feature':<25} | {'Normal User':<15} | {'Knockoff Nets':<15} | {'JBDA Attacker':<15}")
    print("-" * 78)
    
    for key in f_normal.keys():
        print(f"{key:<25} | {f_normal[key]:<15.4f} | {f_knock[key]:<15.4f} | {f_jbda[key]:<15.4f}")

if __name__ == "__main__":
    print_comparison()
