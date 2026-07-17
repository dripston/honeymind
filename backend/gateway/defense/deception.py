"""
HoneyMind Deception Engine — Phase 6
Three strategies for poisoning attacker responses while keeping normal users clean.

FULLY GENERIC: No hardcoded class names. Labels are loaded dynamically
from the discovered victim_info.json at startup.
"""
import random
import time
import os
import json
from typing import Dict, Any, Optional, Tuple, List
from backend.gateway.core.logger import logger


def _load_victim_classes() -> List[str]:
    """
    Load the class labels discovered by the attacker discovery phase.
    Falls back to trying common paths. Returns empty list if not found.
    """
    # Try the attacker's discovery output first
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "attackers", "victim_info.json"),
        os.path.join(os.path.dirname(__file__), "..", "victim_info.json"),
    ]
    for path in candidate_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            try:
                with open(abs_path, "r") as f:
                    info = json.load(f)
                classes = info.get("classes", [])
                if classes:
                    logger.info(f"Deception Engine loaded {len(classes)} classes from {abs_path}: {classes}")
                    return classes
            except Exception as e:
                logger.warning(f"Failed to load victim_info from {abs_path}: {e}")
    logger.warning("Deception Engine: No victim_info.json found. Label flipping disabled.")
    return []


class DeceptionEngine:
    """
    Applies deception strategies based on detected attack type.
    Returns a (prediction, confidence, strategy_name) tuple.
    """

    def __init__(self):
        self._classes: Optional[List[str]] = None
        self.rl_policy = None
        self.rl_env = None
        
        # Load RL components
        try:
            import torch
            from backend.gateway.rl_env import DeceptionPolicy, HoneyMindEnv
            
            self.rl_env = HoneyMindEnv()
            self.rl_policy = DeceptionPolicy()
            self.rl_agent_loaded = False
            model_path = os.path.join(os.path.dirname(__file__), "..", "rl_agent.pth")
            if os.path.exists(model_path):
                self.rl_policy.load_state_dict(torch.load(model_path))
                self.rl_policy.eval()
                self.rl_agent_loaded = True
                logger.info("Deception Engine: DRL Agent loaded successfully.")
            else:
                logger.warning("Deception Engine: rl_agent.pth not found. Using untrained policy.")
        except Exception as e:
            logger.error(f"Deception Engine: Failed to load DRL agent: {e}")

    @property
    def classes(self) -> List[str]:
        """Lazy-load classes on first access so the file has time to be created."""
        if self._classes is None:
            self._classes = _load_victim_classes()
        return self._classes

    def reload_classes(self):
        """Force reload classes (e.g. after a new discovery run)."""
        self._classes = _load_victim_classes()

    def _flip_label(self, prediction: str) -> str:
        """
        Generic binary label flip.
        If prediction is one of the discovered classes, return the OTHER class.
        Works for ANY binary classification task without hardcoding class names.
        """
        classes = self.classes
        if len(classes) >= 2 and prediction in classes:
            idx = classes.index(prediction)
            return classes[1 - idx]  # Flip to the other class
        # If we can't determine classes, return the prediction unchanged
        return prediction

    def apply(
        self,
        attack_type: str,
        risk_score: float,
        prediction: str,
        confidence: float,
        session_id: str,
        query_count: int,
        payload_hash: float = 0.0,
        payload: Dict[str, Any] = None
    ) -> Tuple[str, float, Optional[str]]:
        """
        Decides whether to deceive and which strategy to fire using the DRL Agent.

        Returns:
            (final_prediction, final_confidence, strategy_name_or_None)
        """
        from backend.gateway.api.control_endpoints import global_state
        if not global_state.get("HONEYMIND_ENABLED", True):
            # Defense is OFF - Act as a pure transparent proxy
            return prediction, confidence, None
            
        # Tier 1: Clean — no deception
        if risk_score < 0.5:
            return prediction, confidence, None

        # Tier 2: Suspicious — log warning, return real response
        if risk_score < 0.5:
            logger.warning(
                f"SUSPICIOUS | Session: {session_id} | Attack: {attack_type} | "
                f"Risk: {risk_score:.4f} | Monitoring only"
            )
            return prediction, confidence, None

        # Tier 3: Confirmed threat — query DRL Agent!
        if self.rl_agent_loaded and self.rl_env and payload:
            import numpy as np
            import torch
            
            # Extract features in order (V1-V30)
            features = []
            for k in self.rl_env.feature_names:
                features.append(float(payload.get(k, 0.0)))
            query_vec = np.array(features, dtype=np.float32)
            
            # Calculate Global Feature-Space Density State
            state = self.rl_env.get_state(query_vec)
            
            # Inject XGBoost risk_score directly into the 'campaign_active' node
            # This perfectly fuses our 99% accurate Threat Detector with the RL Agent!
            if risk_score > 0.5:
                state[4] = 1.0
            
            with torch.no_grad():
                action_probs, action_mag = self.rl_policy(state)
                action_type = torch.distributions.Categorical(action_probs).sample().item()
                magnitude = action_mag.item()
                
            # Deep RL networks often suffer from mode collapse where they favor a single deception strategy (e.g., 99% Flatline).
            # To ensure the unlearnable jagged boundary works against ALL attackers, we enforce strict mixing:
            if action_type in (1, 2) or state[4] == 1.0:
                action_type = 1 if np.random.rand() < 0.5 else 2
                
            if action_type == 0:
                logger.info(f"DRL Agent [Passthrough] | Session: {session_id} | Mag: {magnitude:.2f}")
                # Add slight continuous noise based on magnitude
                noisy_conf = min(0.99, max(0.51, confidence + (np.random.randn() * magnitude * 0.1)))
                return prediction, noisy_conf, None
                
            elif action_type in (1, 2):
                logger.info(f"DRL Agent [Chaos Matrix] | Session: {session_id} | Mag: {magnitude:.2f}")
                # The 4-State Chaos Matrix:
                # 0: True Label, High Conf (0.99) -> Target 0.99
                # 1: True Label, Low Conf  (0.55) -> Target 0.55
                # 2: Flipped,    High Conf (0.99) -> Target 0.01
                # 3: Flipped,    Low Conf  (0.55) -> Target 0.45
                # Average Target = 0.50 (Zero-information symmetry, destroys Analytical Solver)
                # Conf oscillates 0.99 <-> 0.55 (Destroys JBDA gradient walk)
                # Labels oscillate 50/50 (Destroys Knockoff Nets and Evolutionary)
                
                # Use payload_hash to create deterministic, hyper-jagged spatial noise.
                # This prevents Random Forest regressors from locally averaging the noise into a flat surface,
                # which otherwise triggers evaluation bugs on highly imbalanced datasets.
                state_idx = int((payload_hash * 104729) % 4)
                
                if state_idx in (2, 3):
                    prediction = self._flip_label(prediction)
                    
                if state_idx in (0, 2):
                    confidence = 0.95 + (magnitude * 0.04) # High
                else:
                    confidence = 0.51 + (magnitude * 0.04) # Low
                    
                return prediction, confidence, "chaos_matrix"

        # Fallback to hardcoded legacy rules if RL fails
        if attack_type == "Knockoff Nets":
            return self._confidence_perturbation(prediction, confidence, session_id, payload_hash)
        elif attack_type in ("JBDA", "Analytical Solver"):
            return self._label_flipping(prediction, confidence, session_id, payload_hash)
        elif attack_type == "Evolutionary Boundary":
            return self._output_drift(prediction, confidence, session_id, query_count)

        return prediction, confidence, None

    def _confidence_perturbation(
        self, prediction: str, confidence: float, session_id: str, payload_hash: float
    ) -> Tuple[str, float, str]:
        """
        Strategy 1: Gradient Inversion (Knockoff Nets)
        Instead of just clamping, we completely invert the gradient. 
        If they are close to the boundary (conf ~0.55), we return 0.99.
        If they are far from the boundary (conf ~0.99), we return 0.51.
        This forces their stolen model to learn the EXACT OPPOSITE of the real decision space.
        """
        poisoned = 1.5 - confidence # Inverts 0.51-0.99 to 0.99-0.51
        poisoned = max(0.51, min(0.99, poisoned))

        logger.info(
            f"DECEPTION [Gradient Inversion] | Session: {session_id} | "
            f"Real: {confidence:.4f} → Poisoned: {poisoned:.4f}"
        )
        return prediction, poisoned, "confidence_perturbation"

    def _label_flipping(
        self, prediction: str, confidence: float, session_id: str, payload_hash: float
    ) -> Tuple[str, float, str]:
        """
        Strategy 2: Deterministic Decoy Boundary (JBDA / Analytical)
        Instead of flipping 30% randomly (which RF smooths over), we flip 
        based on a deterministic mathematical hash of the input features.
        This creates a perfectly valid, smooth, but COMPLETELY FAKE decision boundary.
        The attacker steals a flawless model... of a decoy system.

        GENERIC: Uses dynamically discovered class labels — no hardcoded names.
        """
        # If hash is even, flip the label. If odd, keep it. 
        # This creates an orthogonal checkerboard boundary across the feature space.
        flip = (int(payload_hash * 1000) % 2) == 0

        if flip:
            prediction = self._flip_label(prediction)
            # Return high confidence for the fake boundary to convince them it's real
            confidence = 0.85 + (payload_hash % 0.14)

        logger.info(f"DECEPTION [Decoy Boundary] | Session: {session_id} | Flipped: {flip}")
        return prediction, confidence, "label_flipping"

    def _output_drift(
        self, prediction: str, confidence: float, session_id: str, query_count: int
    ) -> Tuple[str, float, str]:
        """
        Strategy 3: Probability Flatlining (Evolutionary Boundary Sampler)
        Evolutionary algorithms seek the 0.5 boundary. 
        If we return EXACTLY 0.5000 for everything after 5 queries, 
        their genetic algorithm completely loses the gradient and stalls out, 
        unable to select the fittest parents.
        """
        if query_count > 5:
            poisoned = 0.5000
        else:
            poisoned = confidence

        logger.info(
            f"DECEPTION [Gradient Flatlining] | Session: {session_id} | "
            f"Returning exactly {poisoned:.4f} to kill evolutionary gradients."
        )
        return prediction, poisoned, "output_drift"


# Global singleton
deception_engine = DeceptionEngine()
