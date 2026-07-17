import httpx
import time
import numpy as np
import asyncio

GATEWAY_URL = "http://127.0.0.1:8001/api/v1/predict"

async def send_queries(client, headers, payloads, label):
    for i, p in enumerate(payloads):
        response = await client.post(GATEWAY_URL, json=p, headers=headers)
        try:
            data = response.json()
            print(f"  Req {i+1:2d} | Attack Type: {data['attack_type']:<20s} | Risk: {data['risk_score']:.4f}")
        except Exception:
            print(f"  Req {i+1:2d} | ERROR: {response.status_code} {response.text[:80]}")
            return
        await asyncio.sleep(0.05)

async def main():
    print("=" * 65)
    print("  FULL ATTACK SIMULATION — ALL 5 THREAT PROFILES")
    print("=" * 65)
    
    async with httpx.AsyncClient() as client:
        
        # 1. NORMAL USER
        print("\n[1/5] Normal User (IP: 192.168.1.1)")
        payloads = []
        for _ in range(10):
            p = {"Time": float(np.random.uniform(100, 5000)), "Amount": float(np.random.uniform(5, 50))}
            for j in range(1, 29): p[f"V{j}"] = float(np.random.randn())
            payloads.append(p)
        await send_queries(client, {"X-Forwarded-For": "192.168.1.1"}, payloads, "Normal")

        # 2. KNOCKOFF NETS
        print("\n[2/5] Knockoff Nets Attacker (IP: 10.0.0.1)")
        payloads = []
        for _ in range(20):
            p = {"Time": float(np.random.uniform(0, 172800)), "Amount": float(np.random.uniform(0, 1000))}
            for j in range(1, 29): p[f"V{j}"] = float(np.random.uniform(-10, 10))
            payloads.append(p)
        await send_queries(client, {"X-Forwarded-For": "10.0.0.1"}, payloads, "Knockoff")

        # 3. JBDA (Jacobian-Based Data Augmentation)
        print("\n[3/5] JBDA Attacker (IP: 10.0.0.2)")
        seed = {"Time": 500.0, "Amount": 100.0}
        for j in range(1, 29): seed[f"V{j}"] = float(np.random.randn())
        payloads = []
        for _ in range(20):
            p = {"Time": seed["Time"], "Amount": seed["Amount"]}
            for j in range(1, 29): p[f"V{j}"] = seed[f"V{j}"] + float(np.random.uniform(-0.01, 0.01))
            payloads.append(p)
            seed = p
        await send_queries(client, {"X-Forwarded-For": "10.0.0.2"}, payloads, "JBDA")

        # 4. ANALYTICAL SOLVER
        print("\n[4/5] Analytical Solver (IP: 10.0.0.3)")
        payloads = []
        for idx in range(30):
            p = {"Time": 1.0 if idx == 0 else 0.0, "Amount": 0.0}
            for j in range(1, 29): p[f"V{j}"] = 1.0 if idx == j else 0.0
            payloads.append(p)
        await send_queries(client, {"X-Forwarded-For": "10.0.0.3"}, payloads, "Analytical")

        # 5. EVOLUTIONARY BOUNDARY SAMPLER
        print("\n[5/5] Evolutionary Boundary Sampler (IP: 10.0.0.4)")
        payloads = []
        for _ in range(20):
            p = {"Time": float(np.random.uniform(100, 5000)), "Amount": float(np.random.uniform(5, 50))}
            for j in range(1, 29): p[f"V{j}"] = float(np.random.randn() * 2)
            payloads.append(p)
        await send_queries(client, {"X-Forwarded-For": "10.0.0.4"}, payloads, "Evolutionary")

    print("\n" + "=" * 65)

if __name__ == "__main__":
    asyncio.run(main())
