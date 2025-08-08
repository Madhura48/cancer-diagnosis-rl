"""
Final Working Dayhoff Framework - 3-4 Hour Training
Fixed version that resolves all tensor dimension and BatchNorm issues
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import random
from collections import deque
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time
import json
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class CancerGenomicsEnvironment:
    def __init__(self, data_path: str):
        logger.info("Initializing Cancer Genomics Environment...")
        
        self.data = pd.read_csv(data_path)
        self.features = [col for col in self.data.columns if col not in ['diagnosis', 'patient_id', 'id']]
        
        self.data['diagnosis_encoded'] = (self.data['diagnosis'] == 'M').astype(int)
        
        # Normalize features
        self.scaler = StandardScaler()
        feature_data = self.data[self.features].values
        self.X = self.scaler.fit_transform(feature_data)
        self.y = self.data['diagnosis_encoded'].values
        
        # Create extended dataset
        self.create_extended_dataset()
        
        logger.info(f"Environment ready: {self.n_samples} samples, {self.n_features} features")
    
    def create_extended_dataset(self):
        """Create extended dataset for long training"""
        augmented_X = []
        augmented_y = []
        
        # Original data 50x
        for _ in range(50):
            augmented_X.append(self.X)
            augmented_y.append(self.y)
        
        # Add noise versions
        for noise in [0.05, 0.1, 0.15, 0.2]:
            for _ in range(25):
                noisy_X = self.X + np.random.normal(0, noise, self.X.shape)
                augmented_X.append(noisy_X)
                augmented_y.append(self.y)
        
        self.X_extended = np.vstack(augmented_X)
        self.y_extended = np.hstack(augmented_y)
        self.n_samples, self.n_features = self.X_extended.shape
        
        logger.info(f"Extended dataset: {self.n_samples} samples for intensive training")
    
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
    
    def get_reward(self, prediction: int, confidence: float, n_features_used: int) -> float:
        true_label = self.y_extended[self.current_sample]
        
        if prediction == true_label:
            accuracy_reward = 1.0 + confidence * 0.5
        else:
            accuracy_reward = -1.0 - confidence * 0.5
        
        efficiency_bonus = max(0, (self.n_features - n_features_used) / self.n_features) * 0.3
        
        return accuracy_reward + efficiency_bonus

class FeatureAnalysisAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Fixed network without BatchNorm issues
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0003)
        
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.02
        
        self.update_target_network()
        logger.info(f"FeatureAgent: {sum(p.numel() for p in self.q_network.parameters()):,} parameters")
    
    def _build_network(self):
        """Build network without BatchNorm for single-sample compatibility"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.action_dim)
        )
    
    def select_features(self, state: np.ndarray) -> Tuple[List[int], float]:
        if random.random() < self.epsilon:
            n_features = random.randint(8, 20)
            selected_indices = random.sample(range(len(state)), n_features)
            confidence = self.epsilon
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                k = min(15, self.action_dim)
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
            logger.warning(f"Feature agent training failed: {e}")
            return None
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

class DiagnosisAgent:
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Fixed policy network
        self.policy_network = self._build_policy_network().to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.0005)
        
        self.log_probs = []
        self.rewards = []
        self.gamma = 0.99
        
        logger.info(f"DiagnosisAgent: {sum(p.numel() for p in self.policy_network.parameters()):,} parameters")
    
    def _build_policy_network(self):
        """Build policy network without BatchNorm"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
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
        
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probabilities = self.policy_network(features_tensor)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities, dim=1)[0].item()
        
        return prediction, confidence
    
    def select_action(self, features: np.ndarray) -> Tuple[int, torch.Tensor]:
        if len(features) < self.state_dim:
            padded_features = np.zeros(self.state_dim)
            padded_features[:len(features)] = features[:self.state_dim]
            features = padded_features
        elif len(features) > self.state_dim:
            features = features[:self.state_dim]
            
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
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
            logger.warning(f"Policy gradient failed: {e}")
            self.log_probs.clear()
            self.rewards.clear()
            return None

class CoordinationAgent:
    def __init__(self):
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
        
        return {
            'selected_features': selected_features,
            'feature_confidence': feature_confidence,
            'prediction': prediction,
            'log_prob': log_prob,
            'coordination_reward': coordination_reward
        }
    
    def _calculate_coordination_reward(self, feature_conf: float, n_features: int) -> float:
        efficiency_bonus = max(0, (30 - n_features) / 30.0)
        return (feature_conf + efficiency_bonus) * 0.1

class FinalDayhoffFramework:
    def __init__(self, data_path: str):
        logger.info("Initializing Final Dayhoff Framework...")
        
        self.env = CancerGenomicsEnvironment(data_path)
        
        self.feature_agent = FeatureAnalysisAgent(
            state_dim=self.env.n_features,
            action_dim=self.env.n_features
        )
        self.diagnosis_agent = DiagnosisAgent(state_dim=25)
        self.coordination_agent = CoordinationAgent()
        
        # Training configuration for 3-4 hours
        self.config = {
            'total_episodes': 150000,  # Large number for 3-4 hour training
            'evaluation_interval': 5000,
            'checkpoint_interval': 15000,
            'target_update_interval': 1000
        }
        
        self.metrics = {
            'episode_rewards': [],
            'accuracies': [],
            'feature_losses': [],
            'diagnosis_losses': [],
            'feature_usage': [],
            'confidences': []
        }
        
        # Results directory
        self.results_dir = Path("final_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("Final Dayhoff Framework ready for intensive training!")
        logger.info(f"Target: {self.config['total_episodes']:,} episodes over 3-4 hours")
    
    def train_episode(self, episode: int):
        state = self.env.reset()
        
        try:
            result = self.coordination_agent.coordinate_analysis(
                self.feature_agent, self.diagnosis_agent, state
            )
            
            # Calculate reward
            env_reward = self.env.get_reward(
                result['prediction'], 
                result['feature_confidence'], 
                len(result['selected_features'])
            )
            total_reward = env_reward + result['coordination_reward']
            
            # Calculate accuracy
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
            self.metrics['episode_rewards'].append(total_reward)
            self.metrics['accuracies'].append(accuracy)
            self.metrics['feature_usage'].append(len(result['selected_features']))
            self.metrics['confidences'].append(result['feature_confidence'])
            
            if feature_loss:
                self.metrics['feature_losses'].append(feature_loss)
            if diagnosis_loss:
                self.metrics['diagnosis_losses'].append(diagnosis_loss)
            
            return {
                'episode': episode,
                'reward': total_reward,
                'accuracy': accuracy,
                'features_used': len(result['selected_features']),
                'prediction': result['prediction'],
                'true_label': state['true_diagnosis']
            }
            
        except Exception as e:
            logger.error(f"Episode {episode} failed: {e}")
            return {
                'episode': episode,
                'reward': -1.0,
                'accuracy': 0.0,
                'features_used': 0,
                'prediction': 0,
                'true_label': state['true_diagnosis']
            }
    
    def evaluate(self, n_episodes=1000):
        logger.info(f"Evaluating on {n_episodes} episodes...")
        
        original_epsilon = self.feature_agent.epsilon
        self.feature_agent.epsilon = 0.02  # Minimal exploration
        
        predictions = []
        true_labels = []
        confidences = []
        
        for _ in range(n_episodes):
            state = self.env.reset()
            
            # Feature selection
            selected_features, confidence = self.feature_agent.select_features(state['features'])
            
            # Diagnosis prediction
            if selected_features:
                selected_values = state['features'][selected_features[:25]]
            else:
                selected_values = state['features'][:25]
            
            prediction, _ = self.diagnosis_agent.predict_diagnosis(selected_values)
            
            predictions.append(prediction)
            true_labels.append(state['true_diagnosis'])
            confidences.append(confidence)
        
        accuracy = accuracy_score(true_labels, predictions)
        avg_confidence = np.mean(confidences)
        
        # Restore epsilon
        self.feature_agent.epsilon = original_epsilon
        
        logger.info(f"Evaluation: Accuracy={accuracy:.3f}, Confidence={avg_confidence:.3f}")
        
        return {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'total_episodes': n_episodes,
            'classification_report': classification_report(true_labels, predictions, output_dict=True)
        }
    
    def save_checkpoint(self, episode, is_best=False):
        name = f"checkpoint_ep{episode}_best.pth" if is_best else f"checkpoint_ep{episode}.pth"
        
        checkpoint = {
            'episode': episode,
            'feature_agent_state': self.feature_agent.q_network.state_dict(),
            'diagnosis_agent_state': self.diagnosis_agent.policy_network.state_dict(),
            'metrics': self.metrics
        }
        
        torch.save(checkpoint, self.results_dir / name)
        logger.info(f"Saved: {name}")
    
    def train(self):
        logger.info("=== STARTING FINAL 3-4 HOUR TRAINING ===")
        start_time = time.time()
        
        best_accuracy = 0.0
        
        try:
            with tqdm(total=self.config['total_episodes'], desc="Final Training") as pbar:
                for episode in range(self.config['total_episodes']):
                    episode_result = self.train_episode(episode)
                    
                    # Target network updates
                    if episode % self.config['target_update_interval'] == 0:
                        self.feature_agent.update_target_network()
                    
                    # Evaluation
                    if episode % self.config['evaluation_interval'] == 0 and episode > 0:
                        eval_results = self.evaluate(n_episodes=1000)
                        current_accuracy = eval_results['accuracy']
                        
                        if current_accuracy > best_accuracy:
                            best_accuracy = current_accuracy
                            self.save_checkpoint(episode, is_best=True)
                            logger.info(f"NEW BEST: {best_accuracy:.3f} at episode {episode}")
                    
                    # Regular checkpointing
                    if episode % self.config['checkpoint_interval'] == 0 and episode > 0:
                        self.save_checkpoint(episode)
                        
                        elapsed_hours = (time.time() - start_time) / 3600
                        recent_acc = np.mean(self.metrics['accuracies'][-1000:])
                        logger.info(f"Progress - Episode {episode:,}")
                        logger.info(f"  Time: {elapsed_hours:.2f}h | Accuracy: {recent_acc:.3f}")
                        logger.info(f"  Best: {best_accuracy:.3f} | Epsilon: {self.feature_agent.epsilon:.4f}")
                    
                    # Update progress bar
                    if episode % 100 == 0:
                        recent_acc = np.mean(self.metrics['accuracies'][-100:]) if self.metrics['accuracies'] else 0.0
                        pbar.set_postfix({
                            'Acc': f"{recent_acc:.3f}",
                            'Best': f"{best_accuracy:.3f}",
                            'Eps': f"{self.feature_agent.epsilon:.3f}"
                        })
                    pbar.update(1)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            total_time = time.time() - start_time
            logger.info("=== TRAINING COMPLETED ===")
            logger.info(f"Total time: {total_time/3600:.2f} hours")
            logger.info(f"Episodes: {len(self.metrics['episode_rewards']):,}")
            logger.info(f"Best accuracy: {best_accuracy:.3f}")
            
            # Final evaluation
            final_results = self.evaluate(n_episodes=2000)
            
            # Save everything
            self.save_final_results(final_results, total_time)
            
            return final_results, total_time
    
    def save_final_results(self, final_results, training_time):
        # Save models
        torch.save(self.feature_agent.q_network.state_dict(), 
                  self.results_dir / "final_feature_agent.pth")
        torch.save(self.diagnosis_agent.policy_network.state_dict(), 
                  self.results_dir / "final_diagnosis_agent.pth")
        
        # Save comprehensive summary
        summary = {
            'training_summary': {
                'total_time_hours': training_time / 3600,
                'total_episodes': len(self.metrics['episode_rewards']),
                'final_accuracy': final_results['accuracy'],
                'avg_confidence': final_results['avg_confidence'],
                'final_epsilon': self.feature_agent.epsilon
            },
            'final_evaluation': final_results,
            'architecture': {
                'feature_agent_params': sum(p.numel() for p in self.feature_agent.q_network.parameters()),
                'diagnosis_agent_params': sum(p.numel() for p in self.diagnosis_agent.policy_network.parameters())
            }
        }
        
        with open(self.results_dir / 'final_training_results.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"All results saved to {self.results_dir}")

def main():
    logger.info("=== FINAL DAYHOFF TRAINING SESSION ===")
    
    # System check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize and run
    framework = FinalDayhoffFramework("data/processed/primary_cancer_data.csv")
    final_results, training_time = framework.train()
    
    logger.info("=== FINAL SUMMARY ===")
    logger.info(f"Training time: {training_time/3600:.2f} hours")
    logger.info(f"Final accuracy: {final_results['accuracy']:.3f}")
    logger.info("Success! Check 'final_results/' for all outputs")

if __name__ == "__main__":
    print("Starting Final 3-4 Hour Dayhoff Training")
    print("This is the corrected version that will work properly")
    print("Press Ctrl+C to stop early if needed")
    print("-" * 60)
    main()