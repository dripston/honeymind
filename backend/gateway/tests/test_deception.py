import httpx
import asyncio
import numpy as np
import uuid

GATEWAY_URL = "http://127.0.0.1:8001/api/v1/predict"
STATS_URL = "http://127.0.0.1:8001/api/v1/stats"

def get_random_ip():
    return f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"

async def send_normal(client):
    ip = get_random_ip()
    print(f"\n[+] Baseline: 10 Normal Queries from {ip}")
    for i in range(10):
        p = {"Time": float(np.random.uniform(100, 5000)), "Amount": float(np.random.uniform(5, 50))}
        for j in range(1, 29): p[f"V{j}"] = float(np.random.randn())
        
        response = await client.post(GATEWAY_URL, json=p, headers={"X-Forwarded-For": ip})
        data = response.json()
        print(f"  Req {i+1:2d} | Attack: {data['attack_type']:<15s} | Risk: {data['risk_score']:.2f} | Pred: {data['prediction']:<10s} | Conf: {data['confidence']:.4f}")
        await asyncio.sleep(0.05)

async def test_knockoff(client):
    await send_normal(client)
    
    ip = get_random_ip()
    print(f"\n[!] Test 1: 30 Knockoff Nets Queries from {ip} (Strategy: Confidence Perturbation)")
    for i in range(30):
        p = {"Time": float(np.random.uniform(0, 172800)), "Amount": float(np.random.uniform(0, 1000))}
        for j in range(1, 29): p[f"V{j}"] = float(np.random.uniform(-10, 10))
        
        response = await client.post(GATEWAY_URL, json=p, headers={"X-Forwarded-For": ip})
        data = response.json()
        print(f"  Req {i+1:2d} | Attack: {data['attack_type']:<15s} | Risk: {data['risk_score']:.2f} | Conf: {data['confidence']:.4f}  <-- Check for perturbation [0.51 - 0.75]")
        await asyncio.sleep(0.05)

async def test_jbda(client):
    await send_normal(client)
    
    ip = get_random_ip()
    print(f"\n[!] Test 2: 30 JBDA Queries from {ip} (Strategy: Label Flipping)")
    seed = {"Time": 500.0, "Amount": 100.0}
    for j in range(1, 29): seed[f"V{j}"] = float(np.random.randn())
    
    for i in range(30):
        p = {"Time": seed["Time"], "Amount": seed["Amount"]}
        for j in range(1, 29): p[f"V{j}"] = seed[f"V{j}"] + float(np.random.uniform(-0.01, 0.01))
        seed = p
        
        response = await client.post(GATEWAY_URL, json=p, headers={"X-Forwarded-For": ip})
        data = response.json()
        print(f"  Req {i+1:2d} | Attack: {data['attack_type']:<15s} | Risk: {data['risk_score']:.2f} | Pred: {data['prediction']:<10s}  <-- Check for flips (fraud vs legitimate)")
        await asyncio.sleep(0.05)

async def test_evolutionary(client):
    await send_normal(client)
    
    ip = get_random_ip()
    print(f"\n[!] Test 3: 50 Evolutionary Queries from {ip} (Strategy: Output Drift)")
    for i in range(50):
        p = {"Time": float(np.random.uniform(100, 5000)), "Amount": float(np.random.uniform(5, 50))}
        for j in range(1, 29): p[f"V{j}"] = float(np.random.randn() * 2)
        
        response = await client.post(GATEWAY_URL, json=p, headers={"X-Forwarded-For": ip})
        data = response.json()
        print(f"  Req {i+1:2d} | Attack: {data['attack_type']:<15s} | Risk: {data['risk_score']:.2f} | Conf: {data['confidence']:.4f}  <-- Check for smooth drift toward 0.5")
        await asyncio.sleep(0.05)

async def main():
    print("=================================================================")
    print("  PHASE 6: DECEPTION ENGINE TEST")
    print("=================================================================")
    
    async with httpx.AsyncClient() as client:
        await test_knockoff(client)
        await test_jbda(client)
        await test_evolutionary(client)
        
        print("\n=================================================================")
        print("  FINAL GATEWAY STATS")
        print("=================================================================")
        stats = await client.get(STATS_URL)
        import json
        print(json.dumps(stats.json(), indent=2))

if __name__ == "__main__":
    asyncio.run(main())
