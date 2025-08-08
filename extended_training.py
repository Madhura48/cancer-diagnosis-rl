"""
Extended Training Version - Based on the proven working framework
Scales up the working version for 3-4 hour training
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
import json
import sys
from tqdm import tqdm

# Simple logging without emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('extended_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Import the working classes from our previous successful run
class CancerGenomicsEnvironment:
    def __init__(self, data_path: str):
        logger.info("Initializing Cancer Genomics Environment...")
        
        self.data = pd.read_csv(data_path)
        self.features = [col for col in self.data.columns if col not in ['diagnosis', 'patient_id', 'id']]
        self.target_col = 'diagnosis'
        
        self.data['diagnosis_encoded'] = (self.data['diagnosis'] == 'M').astype(int)
        
        logger.info("Normalizing features...")
        self.scaler = StandardScaler()
        feature_data = self.data[self.features].values
        self.X = self.scaler.fit_transform(feature_data)
        self.y = self.data['diagnosis_encoded'].values
        
        # Create augmented dataset for longer training
        self.create_extended_dataset()
        
        logger.info(f"Environment initialized:")
        logger.info(f"  - Original samples: {len(self.data)}")
        logger.info(f"  - Extended samples: {self.n_samples}")
        logger.info(f"  - Features: {self.n_features}")
    
    def create_extended_dataset(self):
        """Create extended dataset for longer training"""
        logger.info("Creating extended dataset...")
        
        augmented_X = []
        augmented_y = []
        
        # Original data multiple times
        for _ in range(20):  # 20x original data
            augmented_X.append(self.X)
            augmented_y.append(self.y)
        
        # Add noise variations
        for noise_level in [0.05, 0.1, 0.15, 0.2]:
            for _ in range(10):
                noisy_X = self.X + np.random.normal(0, noise_level, self.X.shape)
                augmented_X.append(noisy_X)
                augmented_y.append(self.y)
        
        self.X_extended = np.vstack(augmented_X)
        self.y_extended = np.hstack(augmented_y)
        self.n_samples, self.n_features = self.X_extended.shape
        
        logger.info(f"Extended dataset created: {self.n_samples} samples")
    
    def reset(self, sample_idx: int = None):
        if sample_idx is None:
            self.current_sample = random.randint(0, self.n_samples - 1)
        else:
            self.current_sample = sample_idx % self.n_samples
            
        state = {
            'features': self.X_extended[self.current_sample],
            'sample_id': self.current_sample,
            'true_diagnosis': self.y_extended[self.current_sample]
        }
        
        return state
    
    def get_reward(self, prediction: int, confidence: float) -> float:
        true_label = self.y_extended[self.current_sample]
        
        if prediction == true_label:
            base_reward = 1.0
            confidence_bonus = confidence * 0.5
            return base_reward + confidence_bonus
        else:
            base_penalty = -1.0
            confidence_penalty = confidence * 0.5
            return base_penalty - confidence_penalty

class FeatureAnalysisAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.0005):
        logger.info("Initializing Feature Analysis Agent...")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # Larger network for extended training
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        
        # Enhanced replay buffer
        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99998  # Slower decay for extended training
        self.epsilon_min = 0.02
        
        self.update_target_network()
        
        logger.info(f"FeatureAnalysisAgent initialized:")
        logger.info(f"  - Network parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
    
    def _build_network(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.action_dim)
        )
    
    def select_features(self, state: np.ndarray) -> Tuple[List[int], float]:
        if random.random() < self.epsilon:
            n_features = random.randint(8, 18)
            selected_indices = random.sample(range(len(state)), n_features)
            confidence = 0.5
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                k = min(12, self.action_dim)
                top_indices = torch.topk(q_values, k=k)[1].cpu().numpy().flatten()
                selected_indices = top_indices.tolist()
                confidence = torch.softmax(q_values, dim=1).max().item()
        
        return selected_indices, confidence
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        
        try:
            batch = random.sample(self.memory, self.batch_size)
            states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
            actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
            next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
            dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
            
            actions = torch.clamp(actions, 0, self.action_dim - 1)
            
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
            self.optimizer.step()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            return loss.item()
            
        except Exception as e:
            logger.warning(f"Training step failed: {e}")
            return None
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

class DiagnosisAgent:
    def __init__(self, state_dim: int, lr: float = 0.0005):
        logger.info("Initializing Diagnosis Agent...")
        
        self.state_dim = state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Larger policy network
        self.policy_network = self._build_policy_network().to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr, weight_decay=1e-5)
        
        self.log_probs = []
        self.rewards = []
        self.gamma = 0.99
        
        logger.info(f"DiagnosisAgent initialized:")
        logger.info(f"  - Network parameters: {sum(p.numel() for p in self.policy_network.parameters()):,}")
    
    def _build_policy_network(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )
    
    def predict_diagnosis(self, features: np.ndarray) -> Tuple[int, float]:
        if len(features) < self.state_dim:
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
        if len(features) < self.state_dim:
            padded_features = np.zeros(self.state_dim)
            padded_features[:len(features)] = features[:self.state_dim]
            features = padded_features
        elif len(features) > self.state_dim:
            features = features[:self.state_dim]
            
        features_tensor = torch.FloatTensor(features).to(self.device)
        probabilities = self.policy_network(features_tensor)
        
        action_dist = torch.distributions.Categorical(probabilities)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob
    
    def store_trajectory(self, log_prob: torch.Tensor, reward: float):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
    
    def train_step(self):
        if not self.rewards:
            return None
        
        try:
            discounted_rewards = []
            cumulative = 0
            for reward in reversed(self.rewards):
                cumulative = reward + self.gamma * cumulative
                discounted_rewards.insert(0, cumulative)
            
            if len(discounted_rewards) > 1:
                discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
            else:
                discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
            
            policy_loss = []
            for log_prob, reward in zip(self.log_probs, discounted_rewards):
                policy_loss.append(-log_prob * reward)
            
            policy_loss = torch.stack(policy_loss).sum()
            
            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.optimizer.step()
            
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
    def __init__(self):
        self.feature_weights = {}
        self.diagnosis_history = []
        self.coordination_rewards = []
        
        logger.info("CoordinationAgent initialized")
    
    def coordinate_analysis(self, feature_agent, diagnosis_agent, state):
        features = state['features']
        
        selected_features, feature_confidence = feature_agent.select_features(features)
        
        if selected_features:
            selected_feature_values = features[selected_features]
        else:
            selected_feature_values = features[:diagnosis_agent.state_dim]
        
        prediction, log_prob = diagnosis_agent.select_action(selected_feature_values)
        
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
        efficiency_bonus = max(0, (30 - n_features) / 30.0)
        confidence_reward = feature_conf
        
        return (confidence_reward + efficiency_bonus) * 0.1

class ExtendedDayhoffFramework:
    def __init__(self, data_path: str):
        logger.info("Initializing Extended Dayhoff Framework...")
        
        # Initialize environment
        self.env = CancerGenomicsEnvironment(data_path)
        
        # Initialize agents
        self.feature_agent = FeatureAnalysisAgent(
            state_dim=self.env.n_features, 
            action_dim=self.env.n_features
        )
        self.diagnosis_agent = DiagnosisAgent(state_dim=20)
        self.coordination_agent = CoordinationAgent()
        
        # Training configuration
        self.config = {
            'total_episodes': 100000,  # Adjusted for extended training
            'evaluation_interval': 2500,
            'checkpoint_interval': 10000,
            'target_update_interval': 500
        }
        
        # Training metrics
        self.training_history = {
            'episode_rewards': [],
            'accuracy_history': [],
            'loss_history': {'feature': [], 'diagnosis': []},
            'feature_usage': [],
            'confidence_history': [],
            'evaluation_results': []
        }
        
        # Create results directory
        self.results_dir = Path("extended_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("Extended Dayhoff Framework initialized successfully!")
        logger.info(f"Planning {self.config['total_episodes']:,} episodes")
    
    def train_episode(self, episode: int):
        state = self.env.reset()
        
        try:
            result = self.coordination_agent.coordinate_analysis(
                self.feature_agent, self.diagnosis_agent, state
            )
            
            prediction_confidence = torch.softmax(
                self.diagnosis_agent.policy_network(
                    torch.FloatTensor(np.zeros(self.diagnosis_agent.state_dim)).to(self.diagnosis_agent.device)
                ), dim=-1
            ).max().item()
            
            env_reward = self.env.get_reward(result['prediction'], prediction_confidence)
            total_reward = env_reward + result['coordination_reward']
            
            accuracy = 1.0 if result['prediction'] == state['true_diagnosis'] else 0.0
            
            # Store experiences and train
            next_state = self.env.reset()
            action_idx = result['selected_features'][0] if result['selected_features'] else 0
            self.feature_agent.store_experience(
                state['features'], action_idx, env_reward, next_state['features'], True
            )
            
            self.diagnosis_agent.store_trajectory(result['log_prob'], total_reward)
            
            feature_loss = self.feature_agent.train_step()
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
    
    def evaluate(self, n_samples: int = 1000):
        logger.info(f"Evaluating system on {n_samples} samples...")
        
        original_epsilon = self.feature_agent.epsilon
        self.feature_agent.epsilon = 0.02
        
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
        
        self.feature_agent.epsilon = original_epsilon
        
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
    
    def save_checkpoint(self, episode, is_best=False):
        checkpoint_name = f"checkpoint_ep{episode}_best.pth" if is_best else f"checkpoint_ep{episode}.pth"
        
        checkpoint = {
            'episode': episode,
            'feature_agent_state': self.feature_agent.q_network.state_dict(),
            'diagnosis_agent_state': self.diagnosis_agent.policy_network.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }
        
        torch.save(checkpoint, self.results_dir / checkpoint_name)
        logger.info(f"Checkpoint saved: {checkpoint_name}")
    
    def train(self):
        logger.info("=== STARTING EXTENDED 3-4 HOUR TRAINING ===")
        start_time = time.time()
        
        best_accuracy = 0.0
        recent_accuracies = deque(maxlen=500)
        
        try:
            with tqdm(total=self.config['total_episodes'], desc="Extended Training") as pbar:
                for episode in range(self.config['total_episodes']):
                    episode_result = self.train_episode(episode)
                    recent_accuracies.append(episode_result['accuracy'])
                    
                    # Update target network
                    if episode % self.config['target_update_interval'] == 0:
                        self.feature_agent.update_target_network()
                    
                    # Evaluation
                    if episode % self.config['evaluation_interval'] == 0 and episode > 0:
                        eval_results = self.evaluate(n_samples=1000)
                        self.training_history['evaluation_results'].append({
                            'episode': episode,
                            'results': eval_results
                        })
                        
                        current_accuracy = eval_results['accuracy']
                        if current_accuracy > best_accuracy:
                            best_accuracy = current_accuracy
                            self.save_checkpoint(episode, is_best=True)
                            logger.info(f"NEW BEST ACCURACY: {best_accuracy:.3f} at episode {episode}")
                    
                    # Checkpointing
                    if episode % self.config['checkpoint_interval'] == 0 and episode > 0:
                        self.save_checkpoint(episode)
                        
                        elapsed_hours = (time.time() - start_time) / 3600
                        logger.info(f"CHECKPOINT - Episode {episode:,}")
                        logger.info(f"  Elapsed: {elapsed_hours:.2f} hours")
                        logger.info(f"  Recent accuracy: {np.mean(list(recent_accuracies)):.3f}")
                        logger.info(f"  Best accuracy: {best_accuracy:.3f}")
                        logger.info(f"  Epsilon: {self.feature_agent.epsilon:.4f}")
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Accuracy': f"{np.mean(list(recent_accuracies)):.3f}",
                        'Best': f"{best_accuracy:.3f}",
                        'Epsilon': f"{self.feature_agent.epsilon:.3f}"
                    })
                    pbar.update(1)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            total_time = time.time() - start_time
            logger.info("=== EXTENDED TRAINING COMPLETED ===")
            logger.info(f"Total time: {total_time/3600:.2f} hours")
            logger.info(f"Episodes completed: {len(self.training_history['episode_rewards']):,}")
            logger.info(f"Best accuracy achieved: {best_accuracy:.3f}")
            
            # Final evaluation
            final_results = self.evaluate(n_samples=2000)
            
            # Save final results
            self.save_final_results(final_results, total_time)
            
            return final_results, total_time
    
    def save_final_results(self, final_results, training_time):
        # Save models
        torch.save(self.feature_agent.q_network.state_dict(), 
                  self.results_dir / "final_feature_agent.pth")
        torch.save(self.diagnosis_agent.policy_network.state_dict(), 
                  self.results_dir / "final_diagnosis_agent.pth")
        
        # Save comprehensive results
        summary = {
            'training_summary': {
                'total_time_hours': training_time / 3600,
                'total_episodes': len(self.training_history['episode_rewards']),
                'final_epsilon': self.feature_agent.epsilon,
                'final_accuracy': final_results['accuracy'],
                'average_reward': np.mean(self.training_history['episode_rewards'][-2000:])
            },
            'final_evaluation': final_results,
            'model_info': {
                'feature_agent_params': sum(p.numel() for p in self.feature_agent.q_network.parameters()),
                'diagnosis_agent_params': sum(p.numel() for p in self.diagnosis_agent.policy_network.parameters())
            }
        }
        
        with open(self.results_dir / 'extended_training_results.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Final results saved to {self.results_dir}")

def main():
    logger.info("=== EXTENDED DAYHOFF TRAINING SESSION ===")
    
    # Check setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU Memory: {mem_gb:.1f} GB")
    
    # Initialize and run
    framework = ExtendedDayhoffFramework("data/processed/primary_cancer_data.csv")
    final_results, training_time = framework.train()
    
    logger.info("=== FINAL SUMMARY ===")
    logger.info(f"Training completed in {training_time/3600:.2f} hours")
    logger.info(f"Final accuracy: {final_results['accuracy']:.3f}")
    logger.info("Check 'extended_results/' folder for all outputs")

if __name__ == "__main__":
    print("Starting Extended Training...")
    print("This will run for 3-4 hours with comprehensive logging")
    print("Press Ctrl+C to stop early if needed")
    print("-" * 60)
    main()