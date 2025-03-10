#!/usr/bin/env python3

"""
DQN (Deep Q-Network) Algorithm Implementation
for Drone Path Planning in Webots Environment

This implementation provides a Deep Q-Network for drone path planning
with discrete action space control.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple, deque

# Define Experience tuple for replay buffer
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

# Hyperparameters
class DQNConfig:
    def __init__(self, config_file=None):
        # Network architecture
        self.hidden_sizes = [256, 256]
        self.activation = nn.ReLU
        
        # Training parameters
        self.learning_rate = 1e-4
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005  # For soft update of target network
        
        # Replay buffer
        self.buffer_size = 100000
        self.min_buffer_size = 1000  # Min experiences before training
        
        # Exploration parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 0.995
        
        # Training duration
        self.total_timesteps = 500000
        self.update_freq = 4
        self.target_update_freq = 1000
        
        # Checkpointing and logging
        self.save_freq = 10000
        self.log_freq = 1000
        self.eval_freq = 10000
        
        # Discrete actions
        self.action_set = self._default_action_set()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def _default_action_set(self):
        """Default discrete action set for drone control
        
        Each action is a tuple of (roll, pitch, yaw_rate, thrust) values 
        in the range [-1, 1]
        """
        actions = []
        
        # Hover action
        actions.append((0.0, 0.0, 0.0, 0.5))  # (roll, pitch, yaw, thrust)
        
        # Forward/backward actions with different speeds
        actions.append((0.0, 0.5, 0.0, 0.5))   # Forward medium
        actions.append((0.0, 0.8, 0.0, 0.6))   # Forward fast
        actions.append((0.0, -0.5, 0.0, 0.5))  # Backward medium
        actions.append((0.0, -0.8, 0.0, 0.6))  # Backward fast
        
        # Left/right actions with different speeds
        actions.append((0.5, 0.0, 0.0, 0.5))   # Right medium
        actions.append((0.8, 0.0, 0.0, 0.6))   # Right fast
        actions.append((-0.5, 0.0, 0.0, 0.5))  # Left medium
        actions.append((-0.8, 0.0, 0.0, 0.6))  # Left fast
        
        # Up/down actions
        actions.append((0.0, 0.0, 0.0, 0.7))   # Up medium
        actions.append((0.0, 0.0, 0.0, 0.9))   # Up fast
        actions.append((0.0, 0.0, 0.0, 0.3))   # Down medium
        actions.append((0.0, 0.0, 0.0, 0.1))   # Down fast
        
        # Combined actions (diagonal movements)
        actions.append((0.5, 0.5, 0.0, 0.6))   # Forward-right
        actions.append((-0.5, 0.5, 0.0, 0.6))  # Forward-left
        actions.append((0.5, -0.5, 0.0, 0.6))  # Backward-right
        actions.append((-0.5, -0.5, 0.0, 0.6)) # Backward-left
        
        # Rotation actions
        actions.append((0.0, 0.0, 0.5, 0.5))   # Rotate clockwise
        actions.append((0.0, 0.0, -0.5, 0.5))  # Rotate counter-clockwise
        
        return actions
    
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            
            for key, value in config_data.items():
                if key != 'action_set' and hasattr(self, key):
                    setattr(self, key, value)
                    
            # Handle action_set separately (convert lists to tuples)
            if 'action_set' in config_data:
                self.action_set = [tuple(action) for action in config_data['action_set']]
    
    def save_config(self, save_path):
        """Save configuration to JSON file"""
        config_dict = {key: value for key, value in self.__dict__.items()
                      if key != 'action_set' and key != 'activation'}
        
        # Convert action_set tuples to lists for JSON serialization
        config_dict['action_set'] = [list(action) for action in self.action_set]
        
        # Save activation as string
        config_dict['activation'] = self.activation.__name__
        
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @property
    def num_actions(self):
        """Get number of discrete actions"""
        return len(self.action_set)


# Deep Q-Network model
class DQNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_sizes, activation=nn.ReLU):
        super(DQNetwork, self).__init__()
        
        # Build the Q-network
        layers = []
        layers.append(nn.Linear(state_dim, hidden_sizes[0]))
        layers.append(activation())
        
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(activation())
        
        # Output layer (Q-values for each action)
        layers.append(nn.Linear(hidden_sizes[-1], num_actions))
        
        self.q_network = nn.Sequential(*layers)
        
    def forward(self, state):
        """Forward pass to get Q-values"""
        return self.q_network(state)
    
    def select_action(self, state, epsilon=0.0):
        """Select action using epsilon-greedy policy"""
        if random.random() < epsilon:
            # Random action
            return random.randint(0, self.q_network[-1].out_features - 1)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax().item()


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Store experience in buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_dim = state_dim
        self.device = device
        
        # Load configuration or use defaults
        self.config = config if config is not None else DQNConfig()
        
        # Number of actions from action set
        self.num_actions = self.config.num_actions
        
        # Initialize Q-networks
        self.policy_net = DQNetwork(
            state_dim, 
            self.num_actions, 
            self.config.hidden_sizes, 
            activation=self.config.activation
        ).to(device)
        
        self.target_net = DQNetwork(
            state_dim, 
            self.num_actions, 
            self.config.hidden_sizes, 
            activation=self.config.activation
        ).to(device)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in evaluation mode
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.config.learning_rate
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        
        # Exploration rate (epsilon)
        self.epsilon = self.config.epsilon_start
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        self.best_return = -float('inf')
        
        # Metrics tracking
        self.metrics = {
            'episode_returns': [],
            'episode_lengths': [],
            'q_values': [],
            'losses': [],
            'epsilons': []
        }
    
    def select_action(self, state, epsilon=None):
        """Select action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.epsilon
            
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action index using policy network
        action_idx = self.policy_net.select_action(state_tensor, epsilon)
        
        # Convert discrete action index to continuous action
        continuous_action = self.config.action_set[action_idx]
        
        return action_idx, continuous_action
    
    def store_experience(self, state, action_idx, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action_idx, reward, next_state, done)
    
    def update_policy(self):
        """Update policy network using sampled batch"""
        # Skip update if buffer doesn't have enough experiences
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Unzip batch into separate arrays and convert to tensors
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([e.done for e in batch]).astype(np.float32)).to(self.device)
        
        # Compute Q-values for current states and actions
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values
        
        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(q_values, target_q_values)
        
        # Optimize policy network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients (optional, for stability)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        
        self.optimizer.step()
        
        # Update training step
        self.training_step += 1
        
        # Update target network periodically
        if self.training_step % self.config.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, 
                           self.epsilon * self.config.epsilon_decay)
        
        # Track metrics
        self.metrics['q_values'].append(q_values.mean().item())
        self.metrics['losses'].append(loss.item())
        self.metrics['epsilons'].append(self.epsilon)
        
        return {
            'loss': loss.item(),
            'q_value': q_values.mean().item(),
            'epsilon': self.epsilon
        }
    
    def update_target_network(self):
        """Update target network with policy network weights (soft update)"""
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                             self.policy_net.parameters()):
            target_param.data.copy_(
                self.config.tau * policy_param.data + 
                (1 - self.config.tau) * target_param.data
            )
    
    def save_model(self, save_dir, prefix='dqn'):
        """Save model checkpoints"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join(save_dir, f"{prefix}_step{self.training_step}_{timestamp}.pt")
        
        # Save models, optimizer, and training state
        torch.save({
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'best_return': self.best_return,
            'epsilon': self.epsilon,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {key: value for key, value in self.config.__dict__.items() 
                      if key != 'action_set' and key != 'activation'},
            'action_set': [list(action) for action in self.config.action_set],
            'metrics': self.metrics
        }, model_path)
        
        print(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """Load model checkpoint"""
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load models and optimizer
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training state
            self.training_step = checkpoint['training_step']
            self.episode_count = checkpoint['episode_count']
            self.best_return = checkpoint['best_return']
            self.epsilon = checkpoint['epsilon']
            
            # Load action set if available
            if 'action_set' in checkpoint:
                self.config.action_set = [tuple(action) for action in checkpoint['action_set']]
            
            # Load metrics if available
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
            
            print(f"Model loaded from {model_path}")
            print(f"Resuming from training step {self.training_step}, episode {self.episode_count}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def plot_training_metrics(self, save_dir=None):
        """Plot training metrics"""
        # Create subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        
        # Episode returns
        if len(self.metrics['episode_returns']) > 0:
            axs[0, 0].plot(self.metrics['episode_returns'])
            axs[0, 0].set_title('Episode Returns')
            axs[0, 0].set_xlabel('Episode')
            axs[0, 0].set_ylabel('Return')
            
            # Add running average
            if len(self.metrics['episode_returns']) > 10:
                returns = np.array(self.metrics['episode_returns'])
                running_avg = np.convolve(returns, np.ones(10)/10, mode='valid')
                axs[0, 0].plot(range(9, len(returns)), running_avg, 'r--', label='10-episode avg')
                axs[0, 0].legend()
        
        # Episode lengths
        if len(self.metrics['episode_lengths']) > 0:
            axs[0, 1].plot(self.metrics['episode_lengths'])
            axs[0, 1].set_title('Episode Lengths')
            axs[0, 1].set_xlabel('Episode')
            axs[0, 1].set_ylabel('Steps')
        
        # Q-values and losses
        if len(self.metrics['q_values']) > 0:
            axs[1, 0].plot(self.metrics['q_values'])
            axs[1, 0].set_title('Average Q-Values')
            axs[1, 0].set_xlabel('Update')
            axs[1, 0].set_ylabel('Q-Value')
            
            axs[1, 1].plot(self.metrics['losses'])
            axs[1, 1].set_title('Loss')
            axs[1, 1].set_xlabel('Update')
            axs[1, 1].set_ylabel('Loss')
        
        # Epsilon
        if len(self.metrics['epsilons']) > 0:
            axs[2, 0].plot(self.metrics['epsilons'])
            axs[2, 0].set_title('Exploration Rate (Epsilon)')
            axs[2, 0].set_xlabel('Update')
            axs[2, 0].set_ylabel('Epsilon')
            
            # Empty plot for symmetry
            axs[2, 1].axis('off')
        
        plt.tight_layout()
        
        # Save figure if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(save_dir, f'training_metrics_{timestamp}.png'))
            
            # Save metrics as CSV
            metrics_df = pd.DataFrame({
                'training_step': range(len(self.metrics['losses'])),
                'loss': self.metrics['losses'],
                'q_value': self.metrics['q_values'],
                'epsilon': self.metrics['epsilons']
            })
            metrics_df.to_csv(os.path.join(save_dir, f'training_metrics_{timestamp}.csv'), index=False)
            
            # Save episode metrics
            episode_df = pd.DataFrame({
                'episode': range(len(self.metrics['episode_returns'])),
                'return': self.metrics['episode_returns'],
                'length': self.metrics['episode_lengths'],
            })
            episode_df.to_csv(os.path.join(save_dir, f'episode_metrics_{timestamp}.csv'), index=False)
        
        return fig


# DQN Trainer
class DQNTrainer:
    def __init__(self, env, config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = device
        self.config = config if config is not None else DQNConfig()
        
        # Get state dimension
        self.state_dim = env.flat_observation_space.shape[0]
        
        # Initialize agent
        self.agent = DQNAgent(self.state_dim, self.config, device)
        
        # Set up logging and checkpoints
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = f"dqn_training_{timestamp}"
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.log_dir = os.path.join(self.run_dir, "logs")
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save configuration
        self.config.save_config(os.path.join(self.run_dir, "config.json"))
        
        # For tracking progress
        self.total_timesteps = 0
        self.total_episodes = 0
        
        # For calculating running statistics
        self.episode_returns = []
        self.episode_lengths = []
        self.recent_returns = deque(maxlen=10)
    
    def train(self, total_timesteps=None):
        """Train the agent using DQN algorithm"""
        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps
            
        # Training loop
        print(f"Starting training for {total_timesteps} timesteps...")
        
        # Initialize environment
        state = self.env.reset()
        episode_return = 0
        episode_length = 0
        
        # Start timer
        start_time = time.time()
        last_print_time = start_time
        
        # Main training loop
        for t in range(1, total_timesteps + 1):
            # Select action
            action_idx, continuous_action = self.agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(continuous_action)
            
            # Store experience
            self.agent.store_experience(state, action_idx, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Track episode stats
            episode_return += reward
            episode_length += 1
            self.total_timesteps += 1
            
            # Update policy periodically
            if t % self.config.update_freq == 0:
                update_result = self.agent.update_policy()
                
                # Print update result if available
                if update_result and t % self.config.log_freq == 0:
                    print(f"\nUpdate {self.agent.training_step}:")
                    print(f"  Loss: {update_result['loss']:.4f}")
                    print(f"  Q-Value: {update_result['q_value']:.4f}")
                    print(f"  Epsilon: {update_result['epsilon']:.4f}\n")
            
            # If episode ended
            if done:
                self.total_episodes += 1
                self.episode_returns.append(episode_return)
                self.episode_lengths.append(episode_length)
                self.recent_returns.append(episode_return)
                
                # Update agent's episode count
                self.agent.episode_count = self.total_episodes
                
                # Track best performance
                if episode_return > self.agent.best_return:
                    self.agent.best_return = episode_return
                    self.agent.save_model(self.checkpoint_dir, prefix='dqn_best')
                
                # Reset episode stats
                state = self.env.reset()
                episode_return = 0
                episode_length = 0
                
                # Store episode metrics
                self.agent.metrics['episode_returns'].append(float(self.episode_returns[-1]))
                self.agent.metrics['episode_lengths'].append(int(self.episode_lengths[-1]))
                
                # Print progress every few episodes
                current_time = time.time()
                if current_time - last_print_time > 10:  # Print every 10 seconds
                    avg_return = np.mean(self.recent_returns) if self.recent_returns else 0
                    fps = self.total_timesteps / (current_time - start_time)
                    print(f"Episode {self.total_episodes} | "
                          f"Timestep {self.total_timesteps}/{total_timesteps} | "
                          f"Return: {self.episode_returns[-1]:.2f} | "
                          f"Avg Return: {avg_return:.2f} | "
                          f"FPS: {fps:.2f}")
                    last_print_time = current_time
            
            # Save checkpoints and plots periodically
            if t % self.config.save_freq == 0:
                self.agent.save_model(self.checkpoint_dir)
                self.agent.plot_training_metrics(self.log_dir)
            
            # Evaluate periodically
            if t % self.config.eval_freq == 0:
                self.evaluate(n_episodes=5, deterministic=True)
        
        # Final save
        self.agent.save_model(self.checkpoint_dir, prefix='dqn_final')
        
        # Final plots
        self.agent.plot_training_metrics(self.log_dir)
        
        # Calculate training time
        total_time = time.time() - start_time
        hours = int(total_time / 3600)
        minutes = int((total_time % 3600) / 60)
        seconds = int(total_time % 60)
        
        print(f"\nTraining completed in {hours}h {minutes}m {seconds}s")
        print(f"Total episodes: {self.total_episodes}")
        print(f"Best episode return: {self.agent.best_return:.2f}")
        print(f"Final average return (last 10): {np.mean(self.recent_returns):.2f}")
        
        return self.agent
    
    def evaluate(self, n_episodes=10, render=False, deterministic=True):
        """Evaluate the agent's performance"""
        print(f"\nEvaluating agent for {n_episodes} episodes...")
        
        eval_returns = []
        eval_lengths = []
        
        for i in range(n_episodes):
            state = self.env.reset()
            done = False
            episode_return = 0
            episode_length = 0
            
            while not done:
                # Get action
                epsilon = 0.0 if deterministic else self.agent.epsilon
                _, continuous_action = self.agent.select_action(state, epsilon=epsilon)
                
                # Take step in environment
                next_state, reward, done, _ = self.env.step(continuous_action)
                
                # Update state
                state = next_state
                
                # Update stats
                episode_return += reward
                episode_length += 1
                
                # Render if requested
                if render:
                    self.env.render()
            
            # Store episode results
            eval_returns.append(episode_return)
            eval_lengths.append(episode_length)
            
            print(f"Evaluation episode {i+1}/{n_episodes}: Return = {episode_return:.2f}, Length = {episode_length}")
        
        # Calculate statistics
        mean_return = np.mean(eval_returns)
        std_return = np.std(eval_returns)
        mean_length = np.mean(eval_lengths)
        
        print(f"\nEvaluation Results:")
        print(f"  Mean Return: {mean_return:.2f} ± {std_return:.2f}")
        print(f"  Mean Episode Length: {mean_length:.2f}")
        
        return {
            'mean_return': mean_return,
            'std_return': std_return,
            'mean_length': mean_length,
            'returns': eval_returns,
            'lengths': eval_lengths
        }


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from env_wrapper import create_env, WebotsMAVDroneEnv
    
    # Create environment
    env = create_env(headless=True, flat_observation=True)
    
    # Create DQN trainer
    trainer = DQNTrainer(env)
    
    # Train agent
    agent = trainer.train(total_timesteps=100000)
    
    # Evaluate agent
    eval_results = trainer.evaluate(n_episodes=5, deterministic=True)
    
    # Close environment
    env.close() 