<div align="center">
  <h1>🍯 HoneyMind</h1>
  <h3>Active Deception & Dynamic Target Moving for LLM/ML API Security</h3>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" alt="Status" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License" />
</p>

---

## 🚀 Overview

**HoneyMind** is an advanced active-defense gateway designed to protect Machine Learning APIs from **Model Extraction**, **Data Poisoning**, and **Adversarial Exploitation** attacks. 

Unlike traditional passive firewalls that simply block suspicious IP addresses (which attackers easily bypass by rotating proxies), HoneyMind uses **Deep Reinforcement Learning (DRL)** and **Spatial Hashing** to detect Out-Of-Distribution (OOD) attack patterns across distributed sessions. Once an attacker is identified, HoneyMind actively deceives them by feeding them mathematically poisoned responses (the "Chaos Matrix"), rendering their stolen models utterly useless.

### 🛡️ Supported Defenses
- **Knockoff Nets** (High-volume querying)
- **Jacobian-Based Dataset Augmentation (JBDA)**
- **Analytical Solvers** (Covariance/Boundary probing)
- **Evolutionary/Genetic Algorithms**

---

## 🏗️ Architecture

The system is split into two primary components, designed to run in a microservice architecture:

### 1. HoneyMind Gateway (`backend/gateway`)
The intelligent reverse-proxy firewall. It intercepts all incoming requests to the ML API.
- **OOD Detector:** Uses XGBoost and Locality Sensitive Hashing to cluster distributed requests and identify malicious probing.
- **Chaos Matrix (DRL Agent):** Dynamically perturbs the probabilities returned to identified attackers based on their risk score.

### 2. Victim API (`backend/victim_api`)
The underlying target API serving the legitimate Machine Learning model (e.g., Credit Card Fraud Detection). It is fully isolated behind the HoneyMind Gateway.

---

## 💻 Tech Stack
- **Frontend**: React + Vite (Custom Neo-Brutalist OS UI Architecture)
- **Backend**: FastAPI, Uvicorn, Scikit-Learn, Pandas, XGBoost
- **Deployment**: Vercel (Frontend), Render (Backend APIs)

---

## ⚙️ Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dripston/honeymind.git
   cd honeymind
   ```

2. **Backend Setup:**
   ```bash
   cd backend
   pip install -r requirements.txt
   
   # Start the Victim API
   python -m uvicorn victim_api.main:app --host 127.0.0.1 --port 8000
   
   # Start the HoneyMind Gateway
   python -m uvicorn gateway.main:app --host 127.0.0.1 --port 8001
   ```

3. **Frontend Setup:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. Open your browser to `http://localhost:5173` to access the Active Defense Control Center.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
