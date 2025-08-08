"""
Intensive 3-4 Hour Training for Assignment Requirements
Scaled Dayhoff Framework with comprehensive evaluation and logging
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
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import time
import json
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Configure logging without emoji characters for Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('intensive_training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class IntensiveCancerEnvironment:
    """
    Enhanced environment for intensive 3-4 hour training
    """
    
    def __init__(self, data_path: str):
        logger.info("Initializing Intensive Cancer Environment...")
        
        self.data = pd.read_csv(data_path)
        self.features = [col for col in self.data.columns if col not in ['diagnosis', 'patient_id', 'id']]
        self.target_col = 'diagnosis'
        
        # Create encoded diagnosis
        self.data['diagnosis_encoded'] = (self.data['diagnosis'] == 'M').astype(int)
        
        # Create multiple augmented versions for extensive training
        self.create_augmented_dataset()
        
        # Initialize training state
        self.current_sample = 0
        self.episode_count = 0
        self.difficulty_level = 0  # 0=easy, 1=medium, 2=hard, 3=expert
        
        logger.info(f"Intensive Environment initialized:")
        logger.info(f"  - Original samples: {len(self.data)}")
        logger.info(f"  - Augmented samples: {self.n_samples}")
        logger.info(f"  - Features: {self.n_features}")
        logger.info(f"  - Training hours planned: 3-4 hours")
    
    def create_augmented_dataset(self):
        """Create large augmented dataset for 3-4 hour training"""
        logger.info("Creating augmented dataset for intensive training...")
        
        # Normalize original features
        feature_data = self.data[self.features].values
        scaler = StandardScaler()
        X_original = scaler.fit_transform(feature_data)
        y_original = self.data['diagnosis_encoded'].values
        
        augmented_X = []
        augmented_y = []
        
        # 1. Original data (multiple times)
        for _ in range(10):
            augmented_X.append(X_original)
            augmented_y.append(y_original)
        
        # 2. Add various noise levels
        noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25]
        for noise_std in noise_levels:
            for _ in range(5):  # Multiple versions of each noise level
                noisy_X = X_original + np.random.normal(0, noise_std, X_original.shape)
                augmented_X.append(noisy_X)
                augmented_y.append(y_original)
        
        # 3. Feature dropout versions
        dropout_rates = [0.1, 0.2, 0.3]
        for dropout_rate in dropout_rates:
            for _ in range(5):
                dropout_X = X_original.copy()
                n_dropout = int(X_original.shape[1] * dropout_rate)
                for i in range(len(dropout_X)):
                    dropout_indices = np.random.choice(X_original.shape[1], n_dropout, replace=False)
                    dropout_X[i, dropout_indices] = 0
                augmented_X.append(dropout_X)
                augmented_y.append(y_original)
        
        # 4. Scaled versions
        scale_factors = [0.8, 1.2, 1.5, 0.6]
        for scale in scale_factors:
            for _ in range(3):
                scaled_X = X_original * scale
                augmented_X.append(scaled_X)
                augmented_y.append(y_original)
        
        # Combine all augmented data
        self.X = np.vstack(augmented_X)
        self.y = np.hstack(augmented_y)
        self.n_samples, self.n_features = self.X.shape
        
        logger.info(f"Augmented dataset created: {self.n_samples} samples for intensive training")
    
    def reset(self, difficulty=None):
        """Reset with difficulty progression"""
        if difficulty is not None:
            self.difficulty_level = difficulty
        
        self.current_sample = random.randint(0, self.n_samples - 1)
        features = self.X[self.current_sample].copy()
        
        # Apply difficulty modifications
        if self.difficulty_level == 1:  # Medium
            features += np.random.normal(0, 0.1, len(features))
        elif self.difficulty_level == 2:  # Hard
            features += np.random.normal(0, 0.2, len(features))
            # Mask some features
            mask_indices = np.random.choice(len(features), len(features)//10, replace=False)
            features[mask_indices] = 0
        elif self.difficulty_level == 3:  # Expert
            features += np.random.normal(0, 0.3, len(features))
            # Mask more features
            mask_indices = np.random.choice(len(features), len(features)//5, replace=False)
            features[mask_indices] = 0
        
        state = {
            'features': features,
            'sample_id': self.current_sample,
            'true_diagnosis': self.y[self.current_sample],
            'difficulty': self.difficulty_level
        }
        
        self.episode_count += 1
        return state
    
    def get_reward(self, prediction: int, confidence: float, n_features_used: int) -> float:
        """Enhanced reward function"""
        true_label = self.y[self.current_sample]
        
        # Base accuracy reward
        if prediction == true_label:
            accuracy_reward = 1.0 + confidence * 0.5
        else:
            accuracy_reward = -1.0 - confidence * 0.5
        
        # Efficiency bonus (reward using fewer features)
        efficiency_bonus = max(0, (self.n_features - n_features_used) / self.n_features) * 0.3
        
        # Difficulty multiplier
        difficulty_multipliers = [1.0, 1.2, 1.5, 2.0]
        difficulty_bonus = difficulty_multipliers[self.difficulty_level]
        
        total_reward = (accuracy_reward + efficiency_bonus) * difficulty_bonus
        return total_reward

class LargeFeatureAgent:
    """
    Large-scale Feature Analysis Agent for intensive training
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Larger networks for intensive training
        self.q_network = self._build_large_network().to(self.device)
        self.target_network = self._build_large_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0003, weight_decay=1e-5)
        
        # Enhanced experience replay
        self.memory = deque(maxlen=100000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99995  # Slower decay for longer training
        self.epsilon_min = 0.02
        
        # Training metrics
        self.loss_history = []
        self.q_value_history = []
        
        self.update_target_network()
        logger.info(f"Large Feature Agent initialized: {sum(p.numel() for p in self.q_network.parameters()):,} parameters")
    
    def _build_large_network(self):
        """Build larger network for intensive training"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
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
        """Enhanced feature selection"""
        if random.random() < self.epsilon:
            # Advanced exploration
            n_features = random.randint(8, 20)
            if random.random() < 0.3:
                # Random selection
                selected_indices = random.sample(range(len(state)), n_features)
            else:
                # Variance-based exploration
                variances = np.abs(state - np.mean(state))
                top_var_indices = np.argsort(variances)[-25:]
                selected_indices = np.random.choice(top_var_indices, n_features, replace=False).tolist()
            confidence = self.epsilon
        else:
            # Exploitation with Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                # Select top features
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
            
            self.loss_history.append(loss.item())
            return loss.item()
        
        except Exception as e:
            logger.warning(f"Feature agent training failed: {e}")
            return None
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

class LargeDiagnosisAgent:
    """
    Large-scale Diagnosis Agent for intensive training
    """
    
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Larger networks
        self.policy_network = self._build_large_policy().to(self.device)
        self.value_network = self._build_large_value().to(self.device)
        
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=0.0003, weight_decay=1e-5)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.0005, weight_decay=1e-5)
        
        # A2C parameters
        self.trajectories = []
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        logger.info(f"Large Diagnosis Agent initialized: {sum(p.numel() for p in self.policy_network.parameters()):,} parameters")
    
    def _build_large_policy(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
    
    def _build_large_value(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def select_action(self, features: np.ndarray):
        """Select action with A2C"""
        # Ensure correct input size
        if len(features) < self.state_dim:
            padded_features = np.zeros(self.state_dim)
            padded_features[:len(features)] = features[:self.state_dim]
            features = padded_features
        elif len(features) > self.state_dim:
            features = features[:self.state_dim]
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        probabilities = self.policy_network(features_tensor)
        value = self.value_network(features_tensor)
        
        action_dist = torch.distributions.Categorical(probabilities)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        return action.item(), log_prob, value, entropy
    
    def store_trajectory(self, state, action, reward, log_prob, value, entropy, done):
        self.trajectories.append({
            'state': state,
            'action': action,
            'reward': reward,
            'log_prob': log_prob,
            'value': value,
            'entropy': entropy,
            'done': done
        })
    
    def train_step(self):
        if not self.trajectories:
            return None, None
        
        try:
            # Calculate returns and advantages
            returns = []
            advantages = []
            
            # Discounted returns
            discounted_return = 0
            for i in reversed(range(len(self.trajectories))):
                discounted_return = self.trajectories[i]['reward'] + self.gamma * discounted_return * (1 - self.trajectories[i]['done'])
                returns.insert(0, discounted_return)
            
            # GAE advantages
            gae_advantage = 0
            for i in reversed(range(len(self.trajectories))):
                delta = self.trajectories[i]['reward'] + self.gamma * (
                    self.trajectories[i+1]['value'] if i < len(self.trajectories)-1 else 0
                ) * (1 - self.trajectories[i]['done']) - self.trajectories[i]['value']
                
                gae_advantage = delta + self.gamma * self.gae_lambda * gae_advantage * (1 - self.trajectories[i]['done'])
                advantages.insert(0, gae_advantage)
            
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            log_probs = torch.stack([traj['log_prob'] for traj in self.trajectories])
            values = torch.stack([traj['value'] for traj in self.trajectories])
            entropies = torch.stack([traj['entropy'] for traj in self.trajectories])
            
            # Policy loss
            policy_loss = -(log_probs * advantages.detach()).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values, returns.detach())
            
            # Entropy bonus
            entropy_loss = -entropies.mean()
            
            # Update networks
            total_policy_loss = policy_loss + 0.01 * entropy_loss
            
            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
            self.value_optimizer.step()
            
            policy_loss_val = policy_loss.item()
            value_loss_val = value_loss.item()
            
            self.trajectories.clear()
            
            return policy_loss_val, value_loss_val
        
        except Exception as e:
            logger.warning(f"Diagnosis agent training failed: {e}")
            self.trajectories.clear()
            return None, None

class IntensiveDayhoffFramework:
    """
    Intensive Dayhoff Framework for 3-4 hour training
    """
    
    def __init__(self, data_path: str):
        logger.info("Initializing Intensive Dayhoff Framework...")
        
        # Initialize environment
        self.env = IntensiveCancerEnvironment(data_path)
        
        # Initialize large agents
        self.feature_agent = LargeFeatureAgent(
            state_dim=self.env.n_features,
            action_dim=self.env.n_features
        )
        self.diagnosis_agent = LargeDiagnosisAgent(state_dim=25)
        
        # Training configuration for 3-4 hours
        self.config = {
            'total_episodes': 200000,  # Large number for 3-4 hour training
            'evaluation_interval': 5000,
            'checkpoint_interval': 20000,
            'target_update_interval': 1000,
            'difficulty_schedule': {
                0: 0,      # Easy
                50000: 1,  # Medium
                100000: 2, # Hard
                150000: 3  # Expert
            }
        }
        
        # Comprehensive metrics
        self.metrics = {
            'episode_rewards': [],
            'accuracies': [],
            'feature_losses': [],
            'policy_losses': [],
            'value_losses': [],
            'feature_usage': [],
            'confidences': [],
            'evaluation_results': []
        }
        
        # Create results directory
        self.results_dir = Path("intensive_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Intensive Framework initialized for {self.config['total_episodes']:,} episodes")
        logger.info("Expected training time: 3-4 hours")
    
    def train_episode(self, episode: int):
        """Train one episode with comprehensive logging"""
        # Determine difficulty
        current_difficulty = 0
        for threshold, difficulty in self.config['difficulty_schedule'].items():
            if episode >= threshold:
                current_difficulty = difficulty
        
        # Reset environment
        state = self.env.reset(difficulty=current_difficulty)
        
        # Feature selection
        selected_features, feature_confidence = self.feature_agent.select_features(state['features'])
        
        # Prepare features for diagnosis
        if not selected_features:
            selected_features = [0]
        
        max_features = min(len(selected_features), 25)
        selected_values = state['features'][selected_features[:max_features]]
        
        # Diagnosis prediction
        prediction, log_prob, value, entropy = self.diagnosis_agent.select_action(selected_values)
        
        # Calculate rewards
        env_reward = self.env.get_reward(prediction, feature_confidence, len(selected_features))
        
        # Store experiences
        next_state = self.env.reset()
        action_idx = selected_features[0] if selected_features else 0
        
        self.feature_agent.store_experience(
            state['features'], action_idx, env_reward, next_state['features'], True
        )
        
        self.diagnosis_agent.store_trajectory(
            selected_values, prediction, env_reward, log_prob, value, entropy, True
        )
        
        # Train agents
        feature_loss = self.feature_agent.train_step()
        policy_loss, value_loss = self.diagnosis_agent.train_step()
        
        # Calculate metrics
        accuracy = 1.0 if prediction == state['true_diagnosis'] else 0.0
        
        # Store metrics
        self.metrics['episode_rewards'].append(env_reward)
        self.metrics['accuracies'].append(accuracy)
        self.metrics['feature_usage'].append(len(selected_features))
        self.metrics['confidences'].append(feature_confidence)
        
        if feature_loss:
            self.metrics['feature_losses'].append(feature_loss)
        if policy_loss:
            self.metrics['policy_losses'].append(policy_loss)
        if value_loss:
            self.metrics['value_losses'].append(value_loss)
        
        return {
            'episode': episode,
            'reward': env_reward,
            'accuracy': accuracy,
            'features_used': len(selected_features),
            'difficulty': current_difficulty,
            'prediction': prediction,
            'true_label': state['true_diagnosis']
        }
    
    def evaluate_comprehensive(self, n_episodes=2000):
        """Comprehensive evaluation"""
        logger.info(f"Starting comprehensive evaluation over {n_episodes} episodes...")
        
        # Minimal exploration during evaluation
        original_epsilon = self.feature_agent.epsilon
        self.feature_agent.epsilon = 0.01
        
        results = {
            'overall': {'predictions': [], 'true_labels': [], 'confidences': []},
            'by_difficulty': {i: {'predictions': [], 'true_labels': [], 'confidences': []} 
                            for i in range(4)}
        }
        
        for i in tqdm(range(n_episodes), desc="Comprehensive Evaluation"):
            difficulty = i % 4  # Cycle through all difficulties
            
            state = self.env.reset(difficulty=difficulty)
            selected_features, confidence = self.feature_agent.select_features(state['features'])
            
            if not selected_features:
                selected_features = [0]
            
            selected_values = state['features'][selected_features[:25]]
            prediction, _, _, _ = self.diagnosis_agent.select_action(selected_values)
            
            # Store results
            results['overall']['predictions'].append(prediction)
            results['overall']['true_labels'].append(state['true_diagnosis'])
            results['overall']['confidences'].append(confidence)
            
            results['by_difficulty'][difficulty]['predictions'].append(prediction)
            results['by_difficulty'][difficulty]['true_labels'].append(state['true_diagnosis'])
            results['by_difficulty'][difficulty]['confidences'].append(confidence)
        
        # Calculate metrics
        overall_accuracy = accuracy_score(results['overall']['true_labels'], results['overall']['predictions'])
        
        difficulty_accuracies = {}
        for diff in range(4):
            if results['by_difficulty'][diff]['predictions']:
                diff_acc = accuracy_score(
                    results['by_difficulty'][diff]['true_labels'], 
                    results['by_difficulty'][diff]['predictions']
                )
                difficulty_accuracies[f'difficulty_{diff}'] = diff_acc
        
        # Restore epsilon
        self.feature_agent.epsilon = original_epsilon
        
        evaluation_summary = {
            'overall_accuracy': overall_accuracy,
            'difficulty_accuracies': difficulty_accuracies,
            'average_confidence': np.mean(results['overall']['confidences']),
            'total_episodes': n_episodes,
            'classification_report': classification_report(
                results['overall']['true_labels'], 
                results['overall']['predictions'], 
                output_dict=True
            )
        }
        
        return evaluation_summary
    
    def save_checkpoint(self, episode, is_best=False):
        """Save training checkpoint"""
        checkpoint_name = f"checkpoint_ep{episode}_best.pth" if is_best else f"checkpoint_ep{episode}.pth"
        
        checkpoint = {
            'episode': episode,
            'feature_agent_state': self.feature_agent.q_network.state_dict(),
            'diagnosis_policy_state': self.diagnosis_agent.policy_network.state_dict(),
            'diagnosis_value_state': self.diagnosis_agent.value_network.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, self.results_dir / checkpoint_name)
        logger.info(f"Checkpoint saved: {checkpoint_name}")
    
    def intensive_train(self):
        """Main intensive training loop"""
        logger.info("=== STARTING INTENSIVE 3-4 HOUR TRAINING ===")
        start_time = time.time()
        
        best_accuracy = 0.0
        
        try:
            for episode in tqdm(range(self.config['total_episodes']), desc="Intensive Training"):
                # Train episode
                episode_result = self.train_episode(episode)
                
                # Update target networks
                if episode % self.config['target_update_interval'] == 0:
                    self.feature_agent.update_target_network()
                
                # Periodic evaluation
                if episode % self.config['evaluation_interval'] == 0 and episode > 0:
                    eval_results = self.evaluate_comprehensive(n_episodes=1000)
                    self.metrics['evaluation_results'].append({
                        'episode': episode,
                        'results': eval_results
                    })
                    
                    current_accuracy = eval_results['overall_accuracy']
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        self.save_checkpoint(episode, is_best=True)
                        logger.info(f"NEW BEST ACCURACY: {best_accuracy:.3f} at episode {episode}")
                
                # Regular checkpointing
                if episode % self.config['checkpoint_interval'] == 0 and episode > 0:
                    self.save_checkpoint(episode)
                    
                    elapsed_hours = (time.time() - start_time) / 3600
                    logger.info(f"CHECKPOINT - Episode {episode:,} | Elapsed: {elapsed_hours:.2f}h")
                    logger.info(f"Recent accuracy: {np.mean(self.metrics['accuracies'][-1000:]):.3f}")
                    logger.info(f"Best accuracy: {best_accuracy:.3f}")
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            total_time = time.time() - start_time
            logger.info(f"=== INTENSIVE TRAINING COMPLETED ===")
            logger.info(f"Total time: {total_time/3600:.2f} hours")
            logger.info(f"Episodes completed: {len(self.metrics['episode_rewards']):,}")
            logger.info(f"Best accuracy achieved: {best_accuracy:.3f}")
            
            # Final comprehensive evaluation
            logger.info("Performing final evaluation...")
            final_results = self.evaluate_comprehensive(n_episodes=3000)
            
            # Save final results
            self.save_final_results(final_results, total_time)
            
            return final_results, total_time
    
    def save_final_results(self, final_results, training_time):
        """Save comprehensive final results"""
        # Save models
        torch.save(self.feature_agent.q_network.state_dict(), 
                  self.results_dir / "final_feature_agent.pth")
        torch.save(self.diagnosis_agent.policy_network.state_dict(), 
                  self.results_dir / "final_diagnosis_policy.pth")
        torch.save(self.diagnosis_agent.value_network.state_dict(), 
                  self.results_dir / "final_diagnosis_value.pth")
        
        # Comprehensive results summary
        summary = {
            'training_summary': {
                'total_time_hours': training_time / 3600,
                'total_episodes': len(self.metrics['episode_rewards']),
                'final_epsilon': self.feature_agent.epsilon,
                'average_reward': np.mean(self.metrics['episode_rewards'][-5000:]),
                'average_accuracy': np.mean(self.metrics['accuracies'][-5000:])
            },
            'final_evaluation': final_results,
            'training_metrics': {
                'episode_rewards': self.metrics['episode_rewards'][-1000:],  # Last 1000
                'accuracies': self.metrics['accuracies'][-1000:],
                'feature_usage': self.metrics['feature_usage'][-1000:]
            },
            'model_details': {
                'feature_agent_params': sum(p.numel() for p in self.feature_agent.q_network.parameters()),
                'diagnosis_agent_params': sum(p.numel() for p in self.diagnosis_agent.policy_network.parameters()),
                'total_parameters': sum(p.numel() for p in self.feature_agent.q_network.parameters()) + 
                                  sum(p.numel() for p in self.diagnosis_agent.policy_network.parameters())
            }
        }
        
        # Save to JSON
        with open(self.results_dir / 'intensive_training_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Final results saved to {self.results_dir}")
        logger.info("=== INTENSIVE TRAINING COMPLETE ===")

def main():
    """Run intensive 3-4 hour training"""
    logger.info("=== DAYHOFF INTENSIVE TRAINING SESSION ===")
    
    # Check setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")