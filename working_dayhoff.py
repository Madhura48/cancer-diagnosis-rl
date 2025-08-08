"""
Working Dayhoff Framework: Multi-Agent Reinforcement Learning for Cancer Genomics Analysis
Fixed version with proper error handling and verbose output
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time
import sys

# Set up verbose logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class CancerGenomicsEnvironment:
    """
    Environment for cancer genomics analysis with verbose output
    """
    
    def __init__(self, data_path: str):
        logger.info("Initializing Cancer Genomics Environment...")
        
        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Data loaded successfully from {data_path}")
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            raise
        
        self.features = [col for col in self.data.columns if col not in ['diagnosis', 'patient_id', 'id']]
        self.target_col = 'diagnosis'
        
        # Encode diagnosis: M (Malignant) = 1, B (Benign) = 0
        self.data['diagnosis_encoded'] = (self.data['diagnosis'] == 'M').astype(int)
        
        # Normalize features
        logger.info("Normalizing features...")
        self.scaler = StandardScaler()
        feature_data = self.data[self.features].values
        self.X = self.scaler.fit_transform(feature_data)
        self.y = self.data['diagnosis_encoded'].values
        
        self.n_samples, self.n_features = self.X.shape
        self.current_sample = 0
        
        logger.info(f"Environment initialized successfully:")
        logger.info(f"  - Samples: {self.n_samples}")
        logger.info(f"  - Features: {self.n_features}")
        logger.info(f"  - Malignant cases: {np.sum(self.y)} ({np.sum(self.y)/len(self.y)*100:.1f}%)")
        logger.info(f"  - Benign cases: {self.n_samples - np.sum(self.y)} ({(1-np.sum(self.y)/len(self.y))*100:.1f}%)")
    
    def reset(self, sample_idx: int = None):
        """Reset environment to a new sample"""
        if sample_idx is None:
            self.current_sample = random.randint(0, self.n_samples - 1)
        else:
            self.current_sample = sample_idx % self.n_samples
            
        state = {
            'features': self.X[self.current_sample],
            'sample_id': self.current_sample,
            'true_diagnosis': self.y[self.current_sample]
        }
        
        return state
    
    def get_reward(self, prediction: int, confidence: float) -> float:
        """Calculate reward based on prediction accuracy and confidence"""
        true_label = self.y[self.current_sample]
        
        # Base reward for correct prediction
        if prediction == true_label:
            base_reward = 1.0
            confidence_bonus = confidence * 0.5
            return base_reward + confidence_bonus
        else:
            base_penalty = -1.0
            confidence_penalty = confidence * 0.5
            return base_penalty - confidence_penalty

class FeatureAnalysisAgent:
    """
    Agent specialized in analyzing individual genomic features
    Uses DQN to learn which features to focus on
    """
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001):
        logger.info("Initializing Feature Analysis Agent...")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # DQN Network
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Update target network
        self.update_target_network()
        
        logger.info(f"FeatureAnalysisAgent initialized:")
        logger.info(f"  - State dim: {self.state_dim}")
        logger.info(f"  - Action dim: {self.action_dim}")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Network parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
    
    def _build_network(self) -> nn.Module:
        """Build DQN neural network"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim)
        )
    
    def select_features(self, state: np.ndarray) -> Tuple[List[int], float]:
        """Select important features using epsilon-greedy strategy"""
        if random.random() < self.epsilon:
            # Exploration: random feature selection
            n_features = random.randint(5, min(15, len(state)))
            selected_indices = random.sample(range(len(state)), n_features)
            confidence = 0.5
        else:
            # Exploitation: use Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                
                # Select top features based on Q-values
                k = min(10, self.action_dim)
                top_indices = torch.topk(q_values, k=k)[1].cpu().numpy().flatten()
                selected_indices = top_indices.tolist()
                confidence = torch.softmax(q_values, dim=1).max().item()
        
        return selected_indices, confidence
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None
        
        try:
            batch = random.sample(self.memory, self.batch_size)
            states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
            actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
            next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
            dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
            
            # Ensure actions are valid indices
            actions = torch.clamp(actions, 0, self.action_dim - 1)
            
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            return loss.item()
            
        except Exception as e:
            logger.warning(f"Training step failed: {e}")
            return None
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

class DiagnosisAgent:
    """
    Agent specialized in making cancer diagnosis predictions
    Uses Policy Gradient methods
    """
    
    def __init__(self, state_dim: int, lr: float = 0.001):
        logger.info("Initializing Diagnosis Agent...")
        
        self.state_dim = state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy Network
        self.policy_network = self._build_policy_network().to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        
        # Store trajectories for policy gradient
        self.log_probs = []
        self.rewards = []
        self.gamma = 0.99
        
        logger.info(f"DiagnosisAgent initialized:")
        logger.info(f"  - State dim: {self.state_dim}")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Network parameters: {sum(p.numel() for p in self.policy_network.parameters()):,}")
    
    def _build_policy_network(self) -> nn.Module:
        """Build policy network for diagnosis prediction"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # Binary classification: malignant vs benign
            nn.Softmax(dim=-1)
        )
    
    def predict_diagnosis(self, features: np.ndarray) -> Tuple[int, float]:
        """Make diagnosis prediction with confidence"""
        if len(features) < self.state_dim:
            # Pad features if needed
            padded_features = np.zeros(self.state_dim)
            padded_features[:len(features)] = features[:self.state_dim]
            features = padded_features
        elif len(features) > self.state_dim:
            features = features[:self.state_dim]
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            probabilities = self.policy_network(features_tensor)
            prediction = torch.argmax(probabilities).item()
            confidence = torch.max(probabilities).item()
        
        return prediction, confidence
    
    def select_action(self, features: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Select action and return log probability for policy gradient"""
        if len(features) < self.state_dim:
            padded_features = np.zeros(self.state_dim)
            padded_features[:len(features)] = features[:self.state_dim]
            features = padded_features
        elif len(features) > self.state_dim:
            features = features[:self.state_dim]
            
        features_tensor = torch.FloatTensor(features).to(self.device)
        probabilities = self.policy_network(features_tensor)
        
        # Sample from policy
        action_dist = torch.distributions.Categorical(probabilities)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob
    
    def store_trajectory(self, log_prob: torch.Tensor, reward: float):
        """Store trajectory for policy gradient update"""
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
    
    def train_step(self):
        """Perform policy gradient update"""
        if not self.rewards:
            return None
        
        try:
            # Calculate discounted rewards
            discounted_rewards = []
            cumulative = 0
            for reward in reversed(self.rewards):
                cumulative = reward + self.gamma * cumulative
                discounted_rewards.insert(0, cumulative)
            
            # Normalize rewards
            if len(discounted_rewards) > 1:
                discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
            else:
                discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
            
            # Calculate policy loss
            policy_loss = []
            for log_prob, reward in zip(self.log_probs, discounted_rewards):
                policy_loss.append(-log_prob * reward)
            
            policy_loss = torch.stack(policy_loss).sum()
            
            # Update policy
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            # Clear trajectories
            loss_value = policy_loss.item()
            self.log_probs.clear()
            self.rewards.clear()
            
            return loss_value
            
        except Exception as e:
            logger.warning(f"Policy gradient update failed: {e}")
            self.log_probs.clear()
            self.rewards.clear()
            return None

class CoordinationAgent:
    """
    Meta-agent that coordinates between FeatureAnalysisAgent and DiagnosisAgent
    """
    
    def __init__(self):
        self.feature_weights = {}
        self.diagnosis_history = []
        self.coordination_rewards = []
        
        logger.info("CoordinationAgent initialized")
    
    def coordinate_analysis(self, feature_agent: FeatureAnalysisAgent, 
                          diagnosis_agent: DiagnosisAgent, 
                          state: Dict) -> Dict:
        """Coordinate between agents for optimal analysis"""
        features = state['features']
        
        # Get feature selection from feature agent
        selected_features, feature_confidence = feature_agent.select_features(features)
        
        # Use selected features for diagnosis
        if selected_features:
            selected_feature_values = features[selected_features]
        else:
            selected_feature_values = features[:diagnosis_agent.state_dim]
        
        prediction, log_prob = diagnosis_agent.select_action(selected_feature_values)
        
        # Calculate coordination reward
        coordination_reward = self._calculate_coordination_reward(
            feature_confidence, len(selected_features)
        )
        
        result = {
            'selected_features': selected_features,
            'feature_confidence': feature_confidence,
            'prediction': prediction,
            'log_prob': log_prob,
            'coordination_reward': coordination_reward
        }
        
        return result
    
    def _calculate_coordination_reward(self, feature_conf: float, n_features: int) -> float:
        """Calculate reward for coordination quality"""
        efficiency_bonus = max(0, (30 - n_features) / 30.0)  # Reward fewer features
        confidence_reward = feature_conf
        
        return (confidence_reward + efficiency_bonus) * 0.1

class DayhoffFramework:
    """
    Main Dayhoff Framework with comprehensive logging and error handling
    """
    
    def __init__(self, data_path: str):
        logger.info("Initializing Dayhoff Framework...")
        
        # Initialize environment
        self.env = CancerGenomicsEnvironment(data_path)
        
        # Initialize agents
        self.feature_agent = FeatureAnalysisAgent(
            state_dim=self.env.n_features, 
            action_dim=self.env.n_features
        )
        self.diagnosis_agent = DiagnosisAgent(state_dim=20)
        self.coordination_agent = CoordinationAgent()
        
        # Training metrics
        self.training_history = {
            'episode_rewards': [],
            'accuracy_history': [],
            'loss_history': {'feature': [], 'diagnosis': []},
            'feature_usage': [],
            'confidence_history': []
        }
        
        logger.info("Dayhoff Framework initialized successfully!")
    
    def train_episode(self, episode: int) -> Dict:
        """Train for one episode with detailed logging"""
        state = self.env.reset()
        
        # Run coordination analysis
        try:
            result = self.coordination_agent.coordinate_analysis(
                self.feature_agent, self.diagnosis_agent, state
            )
            
            # Get environment reward
            prediction_confidence = torch.softmax(
                self.diagnosis_agent.policy_network(
                    torch.FloatTensor(np.zeros(self.diagnosis_agent.state_dim)).to(self.diagnosis_agent.device)
                ), dim=-1
            ).max().item()
            
            env_reward = self.env.get_reward(result['prediction'], prediction_confidence)
            total_reward = env_reward + result['coordination_reward']
            
            # Calculate accuracy
            accuracy = 1.0 if result['prediction'] == state['true_diagnosis'] else 0.0
            
            # Store experiences and train agents
            feature_state = state['features']
            
            # Train feature agent
            next_state = self.env.reset()
            action_idx = result['selected_features'][0] if result['selected_features'] else 0
            self.feature_agent.store_experience(
                feature_state, action_idx, env_reward, next_state['features'], True
            )
            feature_loss = self.feature_agent.train_step()
            
            # Train diagnosis agent
            self.diagnosis_agent.store_trajectory(result['log_prob'], total_reward)
            diagnosis_loss = self.diagnosis_agent.train_step()
            
            # Store metrics
            self.training_history['episode_rewards'].append(total_reward)
            self.training_history['accuracy_history'].append(accuracy)
            self.training_history['feature_usage'].append(len(result['selected_features']))
            self.training_history['confidence_history'].append(result['feature_confidence'])
            
            if feature_loss is not None:
                self.training_history['loss_history']['feature'].append(feature_loss)
            if diagnosis_loss is not None:
                self.training_history['loss_history']['diagnosis'].append(diagnosis_loss)
            
            return {
                'episode': episode,
                'reward': total_reward,
                'accuracy': accuracy,
                'selected_features': len(result['selected_features']),
                'prediction': result['prediction'],
                'true_label': state['true_diagnosis'],
                'feature_loss': feature_loss,
                'diagnosis_loss': diagnosis_loss
            }
            
        except Exception as e:
            logger.error(f"Episode {episode} failed: {e}")
            return {
                'episode': episode,
                'reward': -1.0,
                'accuracy': 0.0,
                'selected_features': 0,
                'prediction': 0,
                'true_label': state['true_diagnosis'],
                'feature_loss': None,
                'diagnosis_loss': None
            }
    
    def train(self, episodes: int = 1000):
        """Train the multi-agent system with progress tracking"""
        logger.info(f"Starting Dayhoff Framework training for {episodes} episodes")
        start_time = time.time()
        
        best_accuracy = 0.0
        recent_accuracies = deque(maxlen=100)
        
        try:
            for episode in range(episodes):
                episode_result = self.train_episode(episode)
                recent_accuracies.append(episode_result['accuracy'])
                
                # Update target network periodically
                if episode % 100 == 0:
                    self.feature_agent.update_target_network()
                
                # Progress logging
                if episode % 100 == 0 or episode < 10:
                    avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                    avg_accuracy = np.mean(list(recent_accuracies))
                    current_accuracy = avg_accuracy
                    
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        logger.info(f"🎉 New best accuracy: {best_accuracy:.3f}")
                    
                    elapsed_time = time.time() - start_time
                    eps_per_sec = (episode + 1) / elapsed_time
                    eta = (episodes - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0
                    
                    logger.info(f"Episode {episode:4d}/{episodes}")
                    logger.info(f"  Reward: {episode_result['reward']:6.3f} | Accuracy: {current_accuracy:.3f}")
                    logger.info(f"  Features: {episode_result['selected_features']:2d} | Epsilon: {self.feature_agent.epsilon:.3f}")
                    logger.info(f"  Speed: {eps_per_sec:.1f} eps/sec | ETA: {eta/60:.1f}min")
                    logger.info("  " + "-" * 50)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time:.2f} seconds")
            logger.info(f"Best accuracy achieved: {best_accuracy:.3f}")
        
        return self.training_history
    
    def evaluate(self, n_samples: int = 100) -> Dict:
        """Evaluate the trained system"""
        logger.info(f"Evaluating system on {n_samples} samples...")
        
        self.feature_agent.epsilon = 0.01  # Minimal exploration
        
        predictions = []
        true_labels = []
        confidences = []
        
        for i in range(n_samples):
            state = self.env.reset()
            result = self.coordination_agent.coordinate_analysis(
                self.feature_agent, self.diagnosis_agent, state
            )
            
            predictions.append(result['prediction'])
            true_labels.append(state['true_diagnosis'])
            confidences.append(result['feature_confidence'])
        
        accuracy = accuracy_score(true_labels, predictions)
        avg_confidence = np.mean(confidences)
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Confidence: {avg_confidence:.3f}")
        
        return {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'predictions': predictions,
            'true_labels': true_labels,
            'classification_report': classification_report(true_labels, predictions, output_dict=True)
        }
    
    def save_model(self, path: str):
        """Save trained models"""
        save_path = Path(path)
        save_path.mkdir(exist_ok=True, parents=True)
        
        torch.save(self.feature_agent.q_network.state_dict(), 
                  save_path / "feature_agent.pth")
        torch.save(self.diagnosis_agent.policy_network.state_dict(), 
                  save_path / "diagnosis_agent.pth")
        
        logger.info(f"Models saved to {save_path}")

def main():
    """Main training function"""
    logger.info("=== Dayhoff Cancer Genomics RL Framework ===")
    
    # Check data file
    data_file = "data/processed/primary_cancer_data.csv"
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        logger.error("Please run inspect_data.py first to prepare the data")
        return
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Initialize framework
        framework = DayhoffFramework(data_file)
        
        # Train the system
        logger.info("Starting training phase...")
        training_history = framework.train(episodes=2000)
        
        # Evaluate performance
        logger.info("Starting evaluation phase...")
        results = framework.evaluate(n_samples=200)
        
        logger.info("\n" + "="*60)
        logger.info("🎯 FINAL RESULTS")
        logger.info("="*60)
        logger.info(f"Final Accuracy: {results['accuracy']:.3f}")
        logger.info(f"Average Confidence: {results['avg_confidence']:.3f}")
        logger.info("="*60)
        
        # Save models
        framework.save_model("models/dayhoff_agents")
        
        logger.info("✅ Training completed successfully!")
        logger.info("Check 'models/' folder for saved agents")
        logger.info("Check 'training.log' for detailed logs")
        
    except Exception as e:
        logger.error(f"Framework execution failed: {e}")
        raise

if __name__ == "__main__":
    main()