#!/usr/bin/env python3

"""
PPO (Proximal Policy Optimization) Algorithm Implementation
for Drone Path Planning in Webots Environment
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

# Hyperparameters
class PPOConfig:
    def __init__(self, config_file=None):
        # Default hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.policy_learning_rate = 3e-4
        self.value_learning_rate = 1e-3
        self.target_kl = 0.01
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.update_epochs = 10
        self.batch_size = 64
        self.hidden_sizes = [256, 256]
        self.activation = nn.Tanh
        
        # Training parameters
        self.total_timesteps = 1_000_000
        self.steps_per_epoch = 2048
        self.save_freq = 10
        self.eval_freq = 5
        self.log_freq = 1
        
        # Exploration parameters
        self.action_std = 0.6
        self.action_std_decay_rate = 0.05
        self.min_action_std = 0.1
        self.action_std_decay_freq = int(self.total_timesteps / 10)
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
            
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
    def save_config(self, save_path):
        config_dict = {key: value for key, value in self.__dict__.items()}
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=4)


# Neural Network Models
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, activation=nn.Tanh):
        super(ActorNetwork, self).__init__()
        
        # Build the actor network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        layers.append(activation())
        
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(activation())
            
        # Output layer (mean of action distribution)
        layers.append(nn.Linear(hidden_sizes[-1], output_dim))
        layers.append(nn.Tanh())  # Actions in range [-1, 1]
        
        self.actor = nn.Sequential(*layers)
        
        # Log standard deviation (learnable)
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, state):
        # Actor outputs mean of action distribution
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        
        return mean, std
    
    def get_action(self, state, deterministic=False):
        mean, std = self.forward(state)
        
        if deterministic:
            return mean
        
        # Create normal distribution
        dist = Normal(mean, std)
        
        # Sample action from distribution
        action = dist.sample()
        
        # Calculate log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clamp action to [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)
        
        return action, log_prob
    
    def evaluate_action(self, state, action):
        mean, std = self.forward(state)
        
        # Create normal distribution
        dist = Normal(mean, std)
        
        # Calculate log probability and entropy
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation=nn.Tanh):
        super(CriticNetwork, self).__init__()
        
        # Build the critic network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        layers.append(activation())
        
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(activation())
            
        # Output layer (value function)
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.critic = nn.Sequential(*layers)
        
    def forward(self, state):
        # Critic outputs state value
        value = self.critic(state)
        return value


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Load configuration or use defaults
        self.config = config if config is not None else PPOConfig()
        
        # Initialize networks
        self.actor = ActorNetwork(
            state_dim, 
            self.config.hidden_sizes, 
            action_dim, 
            activation=self.config.activation
        ).to(device)
        
        self.critic = CriticNetwork(
            state_dim, 
            self.config.hidden_sizes, 
            activation=self.config.activation
        ).to(device)
        
        # Setup optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=self.config.policy_learning_rate
        )
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=self.config.value_learning_rate
        )
        
        # Initialize statistics
        self.training_step = 0
        self.episode_count = 0
        self.best_return = -float('inf')
        
        # For exploration decay
        self.action_std = self.config.action_std
        
        # Experience buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
        
        # Training metrics
        self.metrics = {
            'episode_returns': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'kl_divs': [],
            'learning_rates': []
        }
        
    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from policy
            action, log_prob = self.actor.get_action(state_tensor, deterministic)
            
            # Get value estimation
            value = self.critic(state_tensor)
            
        # Return numpy arrays
        return (action.cpu().numpy()[0], 
                log_prob.cpu().numpy(), 
                value.cpu().numpy()[0])
    
    def store_experience(self, state, action, reward, done, log_prob, value):
        """Store experience in buffer"""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['values'].append(value)
        
    def compute_advantages(self, last_value, rewards, dones, values, gamma, gae_lambda):
        """Compute Generalized Advantage Estimation (GAE)"""
        # Convert to numpy arrays
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values + [last_value])
        
        returns = []
        advantages = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            # If episode ended, reset advantage calculation
            next_non_terminal = 1.0 - dones[step]
            
            # Calculate TD error (δ)
            delta = rewards[step] + gamma * values[step + 1] * next_non_terminal - values[step]
            
            # Calculate GAE
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            
            # Insert at beginning of list
            advantages.insert(0, gae)
            
            # Calculate returns for value loss
            returns.insert(0, gae + values[step])
            
        return np.array(advantages), np.array(returns)
    
    def update_policy(self):
        """Update policy using PPO algorithm"""
        # Get last observation for bootstrap value
        with torch.no_grad():
            last_state = self.buffer['states'][-1]
            last_state_tensor = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            last_value = self.critic(last_state_tensor).cpu().numpy()[0][0]
        
        # Calculate advantages and returns
        advantages, returns = self.compute_advantages(
            last_value,
            self.buffer['rewards'],
            self.buffer['dones'],
            self.buffer['values'],
            self.config.gamma,
            self.config.gae_lambda
        )
        
        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer['log_probs'])).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Policy update loop
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy = 0
        avg_kl = 0
        
        # Mini-batch training
        batch_size = self.config.batch_size
        buffer_size = len(self.buffer['states'])
        
        for epoch in range(self.config.update_epochs):
            # Generate random indices
            indices = np.random.permutation(buffer_size)
            
            # Mini-batch training
            for start_idx in range(0, buffer_size, batch_size):
                end_idx = min(start_idx + batch_size, buffer_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get mini-batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Get new log probs and entropy
                new_log_probs, entropy = self.actor.evaluate_action(batch_states, batch_actions)
                
                # Calculate policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Add entropy term for exploration
                policy_loss = policy_loss - self.config.entropy_coef * entropy.mean()
                
                # Calculate value loss
                value_pred = self.critic(batch_states)
                value_loss = nn.MSELoss()(value_pred, batch_returns.unsqueeze(1))
                value_loss = value_loss * self.config.value_loss_coef
                
                # Calculate KL divergence
                with torch.no_grad():
                    kl = (batch_old_log_probs - new_log_probs).mean().item()
                    avg_kl += kl
                
                # Update policy if KL divergence is within threshold
                if kl < self.config.target_kl * 1.5:
                    # Optimize policy
                    self.actor_optimizer.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                    self.actor_optimizer.step()
                
                # Optimize value function
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()
                
                # Track losses
                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss.item()
                avg_entropy += entropy.mean().item()
        
        # Calculate averages
        n_batches = buffer_size // batch_size + (1 if buffer_size % batch_size > 0 else 0)
        total_batches = self.config.update_epochs * n_batches
        avg_policy_loss /= total_batches
        avg_value_loss /= total_batches
        avg_entropy /= total_batches
        avg_kl /= total_batches
        
        # Store metrics
        self.metrics['policy_losses'].append(avg_policy_loss)
        self.metrics['value_losses'].append(avg_value_loss)
        self.metrics['entropies'].append(avg_entropy)
        self.metrics['kl_divs'].append(avg_kl)
        self.metrics['learning_rates'].append(self.config.policy_learning_rate)
        
        # Clear buffer
        for key in self.buffer.keys():
            self.buffer[key] = []
            
        # Increment counter
        self.training_step += 1
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'kl': avg_kl
        }
    
    def decay_action_std(self):
        """Decay the action standard deviation for exploration"""
        if self.action_std > self.config.min_action_std:
            self.action_std = max(
                self.config.min_action_std, 
                self.action_std - self.config.action_std_decay_rate
            )
            
            # Update actor log_std parameter
            with torch.no_grad():
                self.actor.log_std.copy_(torch.ones_like(self.actor.log_std) * np.log(self.action_std))
                
            print(f"Action std decayed to {self.action_std:.4f}")
    
    def save_model(self, save_dir, prefix='ppo'):
        """Save model checkpoints"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join(save_dir, f"{prefix}_step{self.training_step}_{timestamp}.pt")
        
        # Save models, optimizers, and training state
        torch.save({
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'best_return': self.best_return,
            'action_std': self.action_std,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config.__dict__,
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
            
            # Load models and optimizers
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            # Load training state
            self.training_step = checkpoint['training_step']
            self.episode_count = checkpoint['episode_count']
            self.best_return = checkpoint['best_return']
            self.action_std = checkpoint['action_std']
            
            # Update log_std parameter
            with torch.no_grad():
                self.actor.log_std.copy_(torch.ones_like(self.actor.log_std) * np.log(self.action_std))
            
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
        
        # Policy and value losses
        if len(self.metrics['policy_losses']) > 0:
            axs[1, 0].plot(self.metrics['policy_losses'])
            axs[1, 0].set_title('Policy Loss')
            axs[1, 0].set_xlabel('Update')
            axs[1, 0].set_ylabel('Loss')
            
            axs[1, 1].plot(self.metrics['value_losses'])
            axs[1, 1].set_title('Value Loss')
            axs[1, 1].set_xlabel('Update')
            axs[1, 1].set_ylabel('Loss')
        
        # Entropy and KL divergence
        if len(self.metrics['entropies']) > 0:
            axs[2, 0].plot(self.metrics['entropies'])
            axs[2, 0].set_title('Entropy')
            axs[2, 0].set_xlabel('Update')
            axs[2, 0].set_ylabel('Entropy')
            
            axs[2, 1].plot(self.metrics['kl_divs'])
            axs[2, 1].set_title('KL Divergence')
            axs[2, 1].set_xlabel('Update')
            axs[2, 1].set_ylabel('KL')
            axs[2, 1].axhline(y=self.config.target_kl, color='r', linestyle='--', label='Target KL')
            axs[2, 1].legend()
        
        plt.tight_layout()
        
        # Save figure if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(save_dir, f'training_metrics_{timestamp}.png'))
            
            # Save metrics as CSV
            metrics_df = pd.DataFrame({
                'training_step': range(len(self.metrics['policy_losses'])),
                'policy_loss': self.metrics['policy_losses'],
                'value_loss': self.metrics['value_losses'],
                'entropy': self.metrics['entropies'],
                'kl_div': self.metrics['kl_divs']
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


# PPO Trainer
class PPOTrainer:
    def __init__(self, env, config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = device
        self.config = config if config is not None else PPOConfig()
        
        # Get state and action dimensions
        self.state_dim = env.flat_observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Initialize agent
        self.agent = PPOAgent(self.state_dim, self.action_dim, self.config, device)
        
        # Set up logging and checkpoints
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = f"ppo_training_{timestamp}"
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
        """Train the agent using PPO algorithm"""
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
            # Get action
            action, log_prob, value = self.agent.get_action(state)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.agent.store_experience(state, action, reward, done, log_prob, value)
            
            # Update state
            state = next_state
            
            # Track episode stats
            episode_return += reward
            episode_length += 1
            self.total_timesteps += 1
            
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
                    self.agent.save_model(self.checkpoint_dir, prefix='ppo_best')
                
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
            
            # Update policy if enough steps have been collected
            if t % self.config.steps_per_epoch == 0:
                update_results = self.agent.update_policy()
                
                print(f"\nPolicy Update {self.agent.training_step}:")
                print(f"  Policy Loss: {update_results['policy_loss']:.4f}")
                print(f"  Value Loss: {update_results['value_loss']:.4f}")
                print(f"  Entropy: {update_results['entropy']:.4f}")
                print(f"  KL Divergence: {update_results['kl']:.4f}\n")
                
                # Check if time to decay action std
                if self.total_timesteps % self.config.action_std_decay_freq == 0:
                    self.agent.decay_action_std()
                
                # Save model checkpoint
                if self.agent.training_step % self.config.save_freq == 0:
                    self.agent.save_model(self.checkpoint_dir)
                
                # Plot and save training metrics
                if self.agent.training_step % self.config.log_freq == 0:
                    self.agent.plot_training_metrics(self.log_dir)
        
        # Final save
        self.agent.save_model(self.checkpoint_dir, prefix='ppo_final')
        
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
                # Get action (deterministic for evaluation)
                action, _, _ = self.agent.get_action(state, deterministic=deterministic)
                
                # Take step in environment
                next_state, reward, done, _ = self.env.step(action)
                
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
    
    # Create PPO trainer
    trainer = PPOTrainer(env)
    
    # Train agent
    agent = trainer.train(total_timesteps=100000)
    
    # Evaluate agent
    eval_results = trainer.evaluate(n_episodes=5, deterministic=True)
    
    # Close environment
    env.close() 