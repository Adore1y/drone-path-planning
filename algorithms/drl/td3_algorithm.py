#!/usr/bin/env python3

"""
TD3 (Twin Delayed DDPG) Algorithm Implementation
for Drone Path Planning in Webots Environment
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import copy
import gym

# TD3 Configuration class
class TD3Config:
    def __init__(self, config_file=None):
        # Network architecture
        self.hidden_sizes = [256, 256]
        
        # Actor parameters
        self.actor_lr = 3e-4
        self.policy_noise = 0.2  # Noise added to target policy during critic update
        self.noise_clip = 0.5    # Range to clip target policy noise
        
        # Critic parameters
        self.critic_lr = 3e-4
        
        # Training parameters
        self.buffer_size = 1000000
        self.min_buffer_size = 1000
        self.batch_size = 100
        self.gamma = 0.99        # Discount factor
        self.tau = 0.005         # Target network update rate
        
        # TD3-specific parameters
        self.policy_freq = 2     # Delayed policy updates parameter (update actor every n critic updates)
        
        # Exploration noise
        self.action_noise = 0.1
        self.noise_decay = 0.995
        self.min_noise = 0.05
        
        # Training schedule
        self.total_timesteps = 1000000
        self.update_freq = 1     # Update networks every n steps
        self.save_freq = 10000   # Save model every n steps
        self.log_freq = 1000     # Log metrics every n steps
        self.eval_freq = 10000   # Evaluate model every n steps
        
        # Load from file if provided
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
                
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
            print(f"Loaded TD3 configuration from {config_file}")
        except Exception as e:
            print(f"Error loading TD3 configuration: {e}")
    
    def save_config(self, save_path):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        
        try:
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"Saved TD3 configuration to {save_path}")
        except Exception as e:
            print(f"Error saving TD3 configuration: {e}")


# Actor Network for continuous action spaces
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes=[256, 256]):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        # Build the network
        layers = []
        layers.append(nn.Linear(state_dim, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_sizes[-1], action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        # Return scaled actions between -max_action and max_action
        return torch.tanh(self.network(state)) * self.max_action
    
    
# Critic Network (Q-function) with twin critics to reduce overestimation bias
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super(Critic, self).__init__()
        
        # First Q-network
        self.q1_layers = nn.ModuleList()
        self.q1_layers.append(nn.Linear(state_dim + action_dim, hidden_sizes[0]))
        
        for i in range(len(hidden_sizes)-1):
            self.q1_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            
        self.q1_output = nn.Linear(hidden_sizes[-1], 1)
        
        # Second Q-network
        self.q2_layers = nn.ModuleList()
        self.q2_layers.append(nn.Linear(state_dim + action_dim, hidden_sizes[0]))
        
        for i in range(len(hidden_sizes)-1):
            self.q2_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            
        self.q2_output = nn.Linear(hidden_sizes[-1], 1)
        
    def forward(self, state, action):
        # Concatenate state and action
        sa = torch.cat([state, action], 1)
        
        # First Q-network
        q1 = F.relu(self.q1_layers[0](sa))
        for i in range(1, len(self.q1_layers)):
            q1 = F.relu(self.q1_layers[i](q1))
        q1 = self.q1_output(q1)
        
        # Second Q-network
        q2 = F.relu(self.q2_layers[0](sa))
        for i in range(1, len(self.q2_layers)):
            q2 = F.relu(self.q2_layers[i](q2))
        q2 = self.q2_output(q2)
        
        return q1, q2
    
    def q1(self, state, action):
        # Only use the first Q-network for action selection
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.q1_layers[0](sa))
        for i in range(1, len(self.q1_layers)):
            q1 = F.relu(self.q1_layers[i](q1))
        q1 = self.q1_output(q1)
        
        return q1


# Experience replay buffer
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )
        
    def __len__(self):
        return self.size


# TD3 Agent Implementation
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action=1.0, config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        
        # Load configuration
        if config is None:
            self.config = TD3Config()
        elif isinstance(config, str):
            self.config = TD3Config(config)
        else:
            self.config = config
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim, max_action, self.config.hidden_sizes).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        
        self.critic = Critic(state_dim, action_dim, self.config.hidden_sizes).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, self.config.buffer_size)
        
        # Training metrics
        self.current_action_noise = self.config.action_noise
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.training_metrics = {
            'q_values': [],
            'actor_loss': [],
            'critic_loss': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'exploration_noise': [],
            'training_time': [],
            'timestamp': []
        }
        self.episode_length = 0
        
    def select_action(self, state, eval_mode=False):
        """Select action with noise for exploration during training or without for evaluation"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        # Get action from policy
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
            
        # Add noise if not in evaluation mode
        if not eval_mode:
            noise = np.random.normal(0, self.current_action_noise, size=self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def store_experience(self, state, action, next_state, reward, done):
        """Store transition in replay buffer"""
        self.replay_buffer.add(state, action, next_state, reward, done)
        
        # Track episode statistics
        self.episode_reward += reward
        self.episode_length += 1
        
    def update_policy(self):
        """Update actor and critic networks"""
        self.total_steps += 1
        
        # Delay training until we have enough samples
        if self.replay_buffer.size < self.config.min_buffer_size:
            return
        
        # Sample from replay buffer
        state, action, next_state, reward, done = self.replay_buffer.sample(self.config.batch_size)
        
        # Update critic
        with torch.no_grad():
            # Select action according to target policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.config.policy_noise
            ).clamp(-self.config.noise_clip, self.config.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)
            
            # Get target Q-values
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.config.gamma * target_q
        
        # Get current Q-values
        current_q1, current_q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        actor_loss = 0
        if self.total_steps % self.config.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            self._soft_update(self.critic, self.critic_target, self.config.tau)
            self._soft_update(self.actor, self.actor_target, self.config.tau)
            
            # Decay exploration noise
            self.current_action_noise = max(
                self.config.min_noise,
                self.current_action_noise * self.config.noise_decay
            )
        
        # Track metrics
        if self.total_steps % self.config.log_freq == 0:
            with torch.no_grad():
                avg_q = torch.mean(current_q1).item()
            
            self.training_metrics['q_values'].append(avg_q)
            self.training_metrics['critic_loss'].append(critic_loss.item())
            self.training_metrics['actor_loss'].append(actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss)
            self.training_metrics['exploration_noise'].append(self.current_action_noise)
            self.training_metrics['training_time'].append(time.time())
            self.training_metrics['timestamp'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def _soft_update(self, source, target, tau):
        """Soft update: target = tau * source + (1 - tau) * target"""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )
    
    def end_episode(self):
        """Track end of episode metrics"""
        if self.episode_length > 0:
            self.episode_rewards.append(self.episode_reward)
            self.training_metrics['episode_rewards'].append(self.episode_reward)
            self.training_metrics['episode_lengths'].append(self.episode_length)
            
            # Reset episode tracking
            self.episode_reward = 0
            self.episode_length = 0
    
    def save_model(self, save_dir, prefix='td3'):
        """Save models to disk"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Save actor and critic
        torch.save(self.actor.state_dict(), os.path.join(save_dir, f"{prefix}_actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, f"{prefix}_critic.pt"))
        
        # Save optimizer states
        torch.save(self.actor_optimizer.state_dict(), os.path.join(save_dir, f"{prefix}_actor_optimizer.pt"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(save_dir, f"{prefix}_critic_optimizer.pt"))
        
        # Save configuration and training state
        training_state = {
            'total_steps': self.total_steps,
            'current_action_noise': self.current_action_noise,
            'config': self.config.__dict__
        }
        
        with open(os.path.join(save_dir, f"{prefix}_training_state.json"), 'w') as f:
            json.dump(training_state, f, indent=2)
            
        print(f"Model saved to {save_dir}")
    
    def load_model(self, model_path):
        """Load models from disk"""
        # Load actor
        actor_path = os.path.join(model_path, "td3_actor.pt")
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.actor_target = copy.deepcopy(self.actor)
            print(f"Loaded actor from {actor_path}")
        
        # Load critic
        critic_path = os.path.join(model_path, "td3_critic.pt")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            self.critic_target = copy.deepcopy(self.critic)
            print(f"Loaded critic from {critic_path}")
            
        # Load optimizer states
        actor_opt_path = os.path.join(model_path, "td3_actor_optimizer.pt")
        if os.path.exists(actor_opt_path):
            self.actor_optimizer.load_state_dict(torch.load(actor_opt_path, map_location=self.device))
            print(f"Loaded actor optimizer from {actor_opt_path}")
            
        critic_opt_path = os.path.join(model_path, "td3_critic_optimizer.pt")
        if os.path.exists(critic_opt_path):
            self.critic_optimizer.load_state_dict(torch.load(critic_opt_path, map_location=self.device))
            print(f"Loaded critic optimizer from {critic_opt_path}")
            
        # Load training state
        state_path = os.path.join(model_path, "td3_training_state.json")
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                training_state = json.load(f)
                
            self.total_steps = training_state.get('total_steps', 0)
            self.current_action_noise = training_state.get('current_action_noise', self.config.action_noise)
            
            # Update config from saved state if available
            if 'config' in training_state:
                for key, value in training_state['config'].items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            
            print(f"Loaded training state from {state_path}")

    def plot_training_metrics(self, save_dir=None):
        """Plot training metrics"""
        if len(self.training_metrics['episode_rewards']) == 0:
            print("No training metrics to plot yet.")
            return
            
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('TD3 Training Metrics', fontsize=16)
        
        # Plot episode rewards
        axes[0, 0].plot(self.training_metrics['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        # Plot episode lengths
        axes[0, 1].plot(self.training_metrics['episode_lengths'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Plot Q-values
        if len(self.training_metrics['q_values']) > 0:
            steps = np.arange(0, len(self.training_metrics['q_values'])) * self.config.log_freq
            axes[1, 0].plot(steps, self.training_metrics['q_values'])
            axes[1, 0].set_title('Average Q-Values')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Q-Value')
        
        # Plot critic loss
        if len(self.training_metrics['critic_loss']) > 0:
            steps = np.arange(0, len(self.training_metrics['critic_loss'])) * self.config.log_freq
            axes[1, 1].plot(steps, self.training_metrics['critic_loss'])
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Loss')
        
        # Plot actor loss
        if len(self.training_metrics['actor_loss']) > 0:
            steps = np.arange(0, len(self.training_metrics['actor_loss'])) * self.config.log_freq
            non_zero_indices = [i for i, x in enumerate(self.training_metrics['actor_loss']) if x != 0]
            non_zero_steps = [steps[i] for i in non_zero_indices]
            non_zero_losses = [self.training_metrics['actor_loss'][i] for i in non_zero_indices]
            
            axes[2, 0].plot(non_zero_steps, non_zero_losses)
            axes[2, 0].set_title('Actor Loss')
            axes[2, 0].set_xlabel('Steps')
            axes[2, 0].set_ylabel('Loss')
        
        # Plot exploration noise
        if len(self.training_metrics['exploration_noise']) > 0:
            steps = np.arange(0, len(self.training_metrics['exploration_noise'])) * self.config.log_freq
            axes[2, 1].plot(steps, self.training_metrics['exploration_noise'])
            axes[2, 1].set_title('Exploration Noise')
            axes[2, 1].set_xlabel('Steps')
            axes[2, 1].set_ylabel('Noise Scale')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure if directory is provided
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            plt.savefig(os.path.join(save_dir, f"td3_training_metrics_{int(time.time())}.png"))
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame({
                'step': np.arange(0, len(self.training_metrics['q_values'])) * self.config.log_freq,
                'q_value': self.training_metrics['q_values'],
                'critic_loss': self.training_metrics['critic_loss'],
                'actor_loss': self.training_metrics['actor_loss'],
                'exploration_noise': self.training_metrics['exploration_noise'],
                'timestamp': self.training_metrics['timestamp']
            })
            metrics_df.to_csv(os.path.join(save_dir, f"td3_training_metrics_{int(time.time())}.csv"), index=False)
            
            # Save episode metrics to CSV
            episode_df = pd.DataFrame({
                'episode': np.arange(0, len(self.training_metrics['episode_rewards'])),
                'reward': self.training_metrics['episode_rewards'],
                'length': self.training_metrics['episode_lengths']
            })
            episode_df.to_csv(os.path.join(save_dir, f"td3_episode_metrics_{int(time.time())}.csv"), index=False)
            
        else:
            plt.show()


# TD3 Trainer class
class TD3Trainer:
    def __init__(self, env, config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = torch.device(device)
        
        # Load configuration
        if config is None:
            self.config = TD3Config()
        elif isinstance(config, str):
            self.config = TD3Config(config)
        else:
            self.config = config
            
        # Get state and action dimensions
        self.state_dim = env.flat_observation_space.shape[0]
        
        # Check if action space is continuous (Box)
        assert isinstance(env.action_space, gym.spaces.Box), "TD3 requires continuous action space"
        
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        
        # Initialize agent
        self.agent = TD3Agent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            config=self.config,
            device=device
        )
        
        # Track training progress
        self.best_eval_reward = -float('inf')
        self.start_time = None
        
    def train(self, total_timesteps=None):
        """Train the agent"""
        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps
            
        print(f"Starting TD3 training for {total_timesteps} timesteps...")
        
        # Setup timing and tracking
        self.start_time = time.time()
        timesteps_elapsed = 0
        episode_num = 0
        
        # Main training loop
        state, _ = self.env.reset()
        
        while timesteps_elapsed < total_timesteps:
            episode_num += 1
            done = False
            episode_step = 0
            episode_reward = 0
            
            # Episode loop
            while not done and timesteps_elapsed < total_timesteps:
                # Select action
                action = self.agent.select_action(state)
                
                # Take step in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store experience
                self.agent.store_experience(state, action, next_state, reward, float(done))
                
                # Update policy
                if timesteps_elapsed % self.config.update_freq == 0:
                    self.agent.update_policy()
                
                # Update state and counters
                state = next_state
                timesteps_elapsed += 1
                episode_step += 1
                episode_reward += reward
                
                # Evaluation
                if timesteps_elapsed % self.config.eval_freq == 0:
                    eval_reward = self.evaluate(n_episodes=5)
                    print(f"Timestep {timesteps_elapsed}/{total_timesteps} | Episode {episode_num} | Evaluation reward: {eval_reward:.2f}")
                    
                    # Save best model
                    if eval_reward > self.best_eval_reward:
                        self.best_eval_reward = eval_reward
                        self.agent.save_model(os.path.join("training_results", "td3_best_model"))
                
                # Save model periodically
                if timesteps_elapsed % self.config.save_freq == 0:
                    self.agent.save_model(os.path.join("training_results", f"td3_step_{timesteps_elapsed}"))
                    
                    # Plot training metrics
                    self.agent.plot_training_metrics(os.path.join("training_results", "logs"))
                    
            # End of episode
            self.agent.end_episode()
            
            # Log episode stats
            if episode_num % 10 == 0:
                elapsed_time = time.time() - self.start_time
                steps_per_sec = timesteps_elapsed / elapsed_time
                print(f"Episode {episode_num} | Steps: {episode_step} | Reward: {episode_reward:.2f} | Steps/sec: {steps_per_sec:.2f}")
                
            # Reset for next episode
            state, _ = self.env.reset()
                
        # Training completed
        total_time = time.time() - self.start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model
        self.agent.save_model(os.path.join("training_results", "td3_final"))
        
        # Final evaluation
        final_reward = self.evaluate(n_episodes=10)
        print(f"Final evaluation reward: {final_reward:.2f}")
        
        # Plot final training metrics
        self.agent.plot_training_metrics(os.path.join("training_results", "logs"))
        
        return self.agent
        
    def evaluate(self, n_episodes=10, render=False, deterministic=True):
        """Evaluate current policy"""
        total_reward = 0
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Select action without exploration noise
                action = self.agent.select_action(state, eval_mode=deterministic)
                
                # Take step in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Render if requested
                if render:
                    self.env.render()
                    
                # Update state and reward
                state = next_state
                episode_reward += reward
                
            total_reward += episode_reward
            
        # Return average reward
        avg_reward = total_reward / n_episodes
        return avg_reward 