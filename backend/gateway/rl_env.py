import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
import warnings
from sklearn.cluster import DBSCAN

warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "victim_api", "ml")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
DATASET_PATH = os.path.join(MODEL_DIR, "..", "..", "..", "dataset.csv")

# -----------------------------
# 1. RL Policy Network (DRL)
# -----------------------------
class DeceptionPolicy(nn.Module):
    def __init__(self, state_dim=5):
        super(DeceptionPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # Action Type Head (Discrete: Passthrough, Flip, Flatline)
        self.type_head = nn.Linear(32, 3)
        # Action Magnitude Head (Continuous: 0.0 to 1.0)
        self.mag_head = nn.Linear(32, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        action_probs = torch.softmax(self.type_head(x), dim=-1)
        action_mag = torch.sigmoid(self.mag_head(x))
        
        return action_probs, action_mag

# -----------------------------
# 2. Environment Simulator
# -----------------------------
class HoneyMindEnv:
    def __init__(self):
        import joblib
        print("[*] Loading Victim Model and Dataset for Simulation...")
        self.model = joblib.load(MODEL_PATH)
            
        df = pd.read_csv(DATASET_PATH)
        self.X_real = df.drop(columns=["Class"]).values
        self.feature_names = df.drop(columns=["Class"]).columns.tolist()
        
        # Calculate real data distribution for OOD detection
        self.real_mean = np.mean(self.X_real, axis=0)
        self.real_std = np.std(self.X_real, axis=0)
        
        # Global Reservoir for State Tracking (Defeating IP Rotation)
        self.query_reservoir = []
        self.max_reservoir_size = 500

    def get_state(self, query, force_campaign=False):
        """
        State formulation based on Global Feature-Space Density.
        """
        # 1. OOD Score (Mahalanobis-like distance)
        z_scores = np.abs((query - self.real_mean) / (self.real_std + 1e-6))
        ood_score = np.mean(z_scores)
        
        # 2. Model Confidence
        probs = self.model.predict_proba([query])[0]
        confidence = float(np.max(probs))
        
        # 3. Global Clustering (Detecting coordinated extraction campaigns)
        cluster_density = 0.0
        dist_to_nearest = 10.0
        campaign_active = 0.0
        
        if len(self.query_reservoir) > 10:
            res_arr = np.array(self.query_reservoir)
            # Find Euclidean distance to all recent queries
            dists = np.linalg.norm(res_arr - query, axis=1)
            dist_to_nearest = np.min(dists)
            
            # Density: How many recent queries are within a small radius of this query?
            # If high, it indicates an attacker is actively mapping this specific region of the boundary.
            radius = 15.0
            neighbors = np.sum(dists < radius)
            cluster_density = neighbors / len(self.query_reservoir)
            
            if cluster_density > 0.05 or force_campaign:
                campaign_active = 1.0  # Coordinated attack detected
                
        # Update reservoir
        self.query_reservoir.append(query)
        if len(self.query_reservoir) > self.max_reservoir_size:
            self.query_reservoir.pop(0)
            
        state = np.array([ood_score, confidence, dist_to_nearest, cluster_density, campaign_active], dtype=np.float32)
        return torch.tensor(state)

    def calculate_reward(self, state, action_type, action_mag):
        """
        Reward is heavily tied to the structural state.
        """
        ood_score, conf, dist, density, campaign = state.numpy()
        
        reward = 0.0
        
        # Case 1: In-Distribution (Normal User)
        if ood_score < 2.0:
            if action_type == 0:  # Passthrough
                reward += 1.0 - action_mag  # Penalize high magnitude noise on real users
            else:
                reward -= 5.0  # Huge penalty for lying to real users
                
        # Case 2: Coordinated Attack Campaign (Dense OOD Clusters)
        elif campaign == 1.0 or ood_score > 3.0:
            if action_type == 1:  # Flip label
                reward += 2.0 + action_mag  # Highly rewarded for aggressive flips during campaign
            elif action_type == 2:  # Flatline
                reward += 1.5 + action_mag
            else:
                reward -= 5.0  # Huge penalty for letting attackers extract the boundary
                
        # Case 3: Random Noise / Isolated Probes (Not part of a campaign)
        else:
            if action_type == 0:  # Passthrough is stealthier if it's just random noise
                reward += 1.0
            else:
                reward -= 1.0  # Don't waste deception on random noise
                
        return reward

# -----------------------------
# 3. Training Loop (REINFORCE)
# -----------------------------
def train_agent():
    print("\n[+] Initializing Deep Reinforcement Learning Agent...")
    env = HoneyMindEnv()
    policy = DeceptionPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=0.005)
    
    episodes = 2000
    
    print("[*] Beginning Simulation & Training Phase (Supervised RL)...")
    criterion = nn.CrossEntropyLoss()
    mag_criterion = nn.MSELoss()
    
    for episode in range(episodes):
        is_attack = False
        if np.random.rand() < 0.3:
            # 30% chance: Real user query
            idx = np.random.randint(0, len(env.X_real))
            query = env.X_real[idx]
        elif np.random.rand() < 0.6:
            is_attack = True
            if len(env.query_reservoir) > 5 and np.random.rand() < 0.8:
                base = env.query_reservoir[-1]
                query = base + np.random.randn(len(base)) * 0.5
            else:
                query = np.random.randn(30) * env.real_std * 5.0 + env.real_mean
        else:
            query = np.random.randn(30) * env.real_std * 10.0
            
        state = env.get_state(query, force_campaign=is_attack)
        
        # Determine optimal action (Oracle)
        ood_score = state[0].item()
        campaign = state[4].item()
        
        if campaign == 1.0 or ood_score > 3.0:
            optimal_action = 1 if np.random.rand() < 0.5 else 2  # 50/50 Flip or Flatline
            optimal_mag = 0.8
        else:
            optimal_action = 0  # Passthrough
            optimal_mag = 0.1
            
        # Forward pass
        action_probs, action_mag = policy(state)
        
        # Supervised Loss
        target_action = torch.tensor(optimal_action)
        target_mag = torch.tensor([optimal_mag], dtype=torch.float32)
        
        # We need logits for CrossEntropyLoss, but our network outputs probs.
        # It's fine, we can just use NLLLoss on log(probs+eps)
        log_probs = torch.log(action_probs + 1e-8)
        loss = nn.NLLLoss()(log_probs.unsqueeze(0), target_action.unsqueeze(0))
        mag_loss = mag_criterion(action_mag, target_mag)
        
        total_loss = loss + mag_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if episode % 500 == 0:
            print(f"  [Episode {episode:4d}] Loss: {total_loss.item():.4f} | Optimal Action: {optimal_action}")

    print("\n[+] Training Complete.")
    
    # Save the model
    save_path = os.path.join(os.path.dirname(__file__), "rl_agent.pth")
    torch.save(policy.state_dict(), save_path)
    print(f"[+] DRL Model saved to {save_path}")

if __name__ == "__main__":
    train_agent()
