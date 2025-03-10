"""
GAT-TD3 Algorithm: Twin Delayed Deep Deterministic Policy Gradient with Graph Attention Networks
This implementation extends the TD3 algorithm with graph attention networks for more efficient state representation.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import gym

# GAT layer implementation for batched inputs
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    Modified to handle batched inputs
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Define trainable weights and attention parameters
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        h: Node features [batch_size, N, in_features]
        adj: Adjacency matrix [batch_size, N, N]
        """
        batch_size, N, _ = h.size()
        
        # Apply linear transformation to node features
        # [batch_size, N, in_features] -> [batch_size, N, out_features]
        Wh = torch.matmul(h, self.W)
        
        # Prepare for attention mechanism
        # We need to get all pairs of nodes for attention computation
        
        # Repeat Wh for each node (source nodes)
        # [batch_size, N, 1, out_features] -> [batch_size, N, N, out_features]
        Wh_repeated_source = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        
        # Repeat Wh for each node (target nodes)
        # [batch_size, 1, N, out_features] -> [batch_size, N, N, out_features]
        Wh_repeated_target = Wh.unsqueeze(1).repeat(1, N, 1, 1)
        
        # Concatenate source and target node features
        # [batch_size, N, N, 2*out_features]
        a_input = torch.cat([Wh_repeated_source, Wh_repeated_target], dim=3)
        
        # Apply attention mechanism
        # [batch_size, N, N, 2*out_features] -> [batch_size, N, N, 1] -> [batch_size, N, N]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        
        # Masked attention (set attention to -inf for absent edges)
        # [batch_size, N, N]
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Apply softmax to get attention coefficients
        # [batch_size, N, N]
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention to node features
        # [batch_size, N, N] x [batch_size, N, out_features] -> [batch_size, N, out_features]
        h_prime = torch.matmul(attention, Wh)
        
        # Apply non-linearity if needed
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

# Multi-head GAT layer
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        # First GAT layer with multiple attention heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
            for _ in range(nheads)
        ])
        
        # Output GAT layer with a single attention head
        self.out_att = GraphAttentionLayer(
            nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False
        )

    def forward(self, x, adj):
        """
        x: Node features [batch_size, N, nfeat]
        adj: Adjacency matrix [batch_size, N, N]
        """
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Apply multiple attention heads and concatenate results
        # Each head: [batch_size, N, nhid]
        # After cat: [batch_size, N, nhid*nheads]
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Apply output attention layer
        # [batch_size, N, nout]
        x = self.out_att(x, adj)
        
        return x

# Configuration for GAT-TD3
class GATTD3Config:
    def __init__(self, config_file=None):
        # Network parameters
        self.hidden_sizes = [256, 256]
        
        # GAT parameters
        self.gat_hidden_dim = 64
        self.gat_output_dim = 32
        self.gat_num_heads = 4
        self.gat_dropout = 0.2
        self.gat_alpha = 0.2  # LeakyReLU negative slope
        
        # Number of nodes in the graph
        self.num_nodes = 12  # This represents environment entities (drone, waypoints, obstacles)
        
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
        self.policy_freq = 2     # Delayed policy updates parameter
        
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
                    
            print(f"Loaded GAT-TD3 configuration from {config_file}")
        except Exception as e:
            print(f"Error loading GAT-TD3 configuration: {e}")
    
    def save_config(self, save_path):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        
        try:
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"Saved GAT-TD3 configuration to {save_path}")
        except Exception as e:
            print(f"Error saving GAT-TD3 configuration: {e}")


# GAT Actor Network for continuous action spaces
class GATActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, config):
        super(GATActor, self).__init__()
        
        self.max_action = max_action
        self.config = config
        self.state_dim = state_dim
        
        # Graph parameters
        self.num_nodes = config.num_nodes
        # Calculate feature dimension based on state size
        # If state_dim is not divisible by num_nodes, we'll handle it in _create_graph_from_state
        self.feature_dim = max(1, state_dim // config.num_nodes)
        
        # GAT Network
        self.gat = GAT(
            nfeat=self.feature_dim,
            nhid=config.gat_hidden_dim,
            nout=config.gat_output_dim,
            dropout=config.gat_dropout,
            alpha=config.gat_alpha,
            nheads=config.gat_num_heads
        )
        
        # MLP head after GAT processing
        gat_output_size = config.gat_output_dim * config.num_nodes
        
        self.fc1 = nn.Linear(gat_output_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
    def forward(self, state):
        # Process graph structure from state
        features, adj = self._create_graph_from_state(state)
        
        # Apply GAT - output shape: [batch_size, num_nodes, output_dim]
        x = self.gat(features, adj)
        
        # Flatten GAT output - reshape to [batch_size, num_nodes * output_dim]
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Apply MLP head
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        return self.max_action * x
    
    def _create_graph_from_state(self, state):
        """
        Create graph features and adjacency matrix from state
        
        For batched input:
        - state shape: (batch_size, state_dim)
        - features shape: (batch_size, num_nodes, feature_dim)
        - adj shape: (batch_size, num_nodes, num_nodes)
        """
        batch_size = state.size(0)
        
        # Create a more flexible approach to handle arbitrary state dimensions
        # Instead of trying to reshape directly, we'll create a new tensor with the right shape
        # and fill it with the state values as much as possible
        
        # Create empty features tensor
        features = torch.zeros(batch_size, self.num_nodes, self.feature_dim, device=state.device)
        
        # Fill features tensor with state values
        # We'll flatten the state and then distribute it across the nodes
        flat_state = state.view(batch_size, -1)
        state_size = flat_state.size(1)
        
        # Calculate how many elements we can fill
        elements_per_node = min(self.feature_dim, (state_size + self.num_nodes - 1) // self.num_nodes)
        
        for i in range(self.num_nodes):
            start_idx = i * elements_per_node
            end_idx = min(start_idx + elements_per_node, state_size)
            
            if start_idx < state_size:
                # Fill as many features as we can for this node
                features[:, i, :end_idx-start_idx] = flat_state[:, start_idx:end_idx]
        
        # Create a fully connected adjacency matrix
        adj = torch.ones(batch_size, self.num_nodes, self.num_nodes, device=state.device)
        
        return features, adj


# GAT Critic Network
class GATCritic(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(GATCritic, self).__init__()
        
        self.config = config
        self.state_dim = state_dim
        
        # Graph parameters
        self.num_nodes = config.num_nodes
        # Calculate feature dimension based on state size
        self.feature_dim = max(1, state_dim // config.num_nodes)
        
        # GAT for state processing
        self.gat = GAT(
            nfeat=self.feature_dim,
            nhid=config.gat_hidden_dim,
            nout=config.gat_output_dim,
            dropout=config.gat_dropout,
            alpha=config.gat_alpha,
            nheads=config.gat_num_heads
        )
        
        # Output size of GAT
        gat_output_size = config.gat_output_dim * config.num_nodes
        
        # First Q-network
        self.q1_fc1 = nn.Linear(gat_output_size + action_dim, 256)
        self.q1_fc2 = nn.Linear(256, 256)
        self.q1_fc3 = nn.Linear(256, 1)
        
        # Second Q-network
        self.q2_fc1 = nn.Linear(gat_output_size + action_dim, 256)
        self.q2_fc2 = nn.Linear(256, 256)
        self.q2_fc3 = nn.Linear(256, 1)
    
    def forward(self, state, action):
        # Process graph structure from state
        features, adj = self._create_graph_from_state(state)
        
        # Apply GAT - output shape: [batch_size, num_nodes, output_dim]
        x = self.gat(features, adj)
        
        # Flatten GAT output - reshape to [batch_size, num_nodes * output_dim]
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Concatenate state representation with action
        xu = torch.cat([x, action], 1)
        
        # First Q-network
        q1 = F.relu(self.q1_fc1(xu))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        
        # Second Q-network
        q2 = F.relu(self.q2_fc1(xu))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)
        
        return q1, q2
    
    def q1(self, state, action):
        # Process graph structure from state
        features, adj = self._create_graph_from_state(state)
        
        # Apply GAT - output shape: [batch_size, num_nodes, output_dim]
        x = self.gat(features, adj)
        
        # Flatten GAT output - reshape to [batch_size, num_nodes * output_dim]
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Concatenate state representation with action
        xu = torch.cat([x, action], 1)
        
        # First Q-network only
        q1 = F.relu(self.q1_fc1(xu))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        
        return q1
    
    def _create_graph_from_state(self, state):
        """
        Create graph features and adjacency matrix from state
        """
        batch_size = state.size(0)
        
        # Create a more flexible approach to handle arbitrary state dimensions
        # Instead of trying to reshape directly, we'll create a new tensor with the right shape
        # and fill it with the state values as much as possible
        
        # Create empty features tensor
        features = torch.zeros(batch_size, self.num_nodes, self.feature_dim, device=state.device)
        
        # Fill features tensor with state values
        # We'll flatten the state and then distribute it across the nodes
        flat_state = state.view(batch_size, -1)
        state_size = flat_state.size(1)
        
        # Calculate how many elements we can fill
        elements_per_node = min(self.feature_dim, (state_size + self.num_nodes - 1) // self.num_nodes)
        
        for i in range(self.num_nodes):
            start_idx = i * elements_per_node
            end_idx = min(start_idx + elements_per_node, state_size)
            
            if start_idx < state_size:
                # Fill as many features as we can for this node
                features[:, i, :end_idx-start_idx] = flat_state[:, start_idx:end_idx]
        
        # Create a fully connected adjacency matrix
        adj = torch.ones(batch_size, self.num_nodes, self.num_nodes, device=state.device)
        
        return features, adj


# Experience replay buffer (reused from TD3)
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


# GAT-TD3 Agent Implementation
class GATTD3Agent:
    def __init__(self, state_dim, action_dim, max_action=1.0, config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        
        # Load configuration
        if config is None:
            self.config = GATTD3Config()
        elif isinstance(config, str):
            self.config = GATTD3Config(config)
        else:
            self.config = config
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Initialize actor and critic networks
        self.actor = GATActor(state_dim, action_dim, max_action, self.config).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        
        self.critic = GATCritic(state_dim, action_dim, self.config).to(self.device)
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
    
    def save_model(self, save_dir, prefix='gat_td3'):
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
            'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_') and not callable(v)}
        }
        
        with open(os.path.join(save_dir, f"{prefix}_training_state.json"), 'w') as f:
            json.dump(training_state, f, indent=2)
            
        print(f"Model saved to {save_dir}")
    
    def load_model(self, model_path, prefix='gat_td3'):
        """Load models from disk"""
        # Load actor
        actor_path = os.path.join(model_path, f"{prefix}_actor.pt")
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.actor_target = copy.deepcopy(self.actor)
            print(f"Loaded actor from {actor_path}")
        
        # Load critic
        critic_path = os.path.join(model_path, f"{prefix}_critic.pt")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            self.critic_target = copy.deepcopy(self.critic)
            print(f"Loaded critic from {critic_path}")
        
        # Load training state
        state_path = os.path.join(model_path, f"{prefix}_training_state.json")
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
                self.total_steps = state.get('total_steps', 0)
                self.current_action_noise = state.get('current_action_noise', self.config.action_noise)
                print(f"Loaded training state from {state_path}")
        
        print(f"Model loading complete")
    
    def plot_metrics(self, save_dir=None):
        """Plot and save training metrics"""
        if len(self.training_metrics['q_values']) == 0:
            print("No metrics to plot yet")
            return
            
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        plt.subplots_adjust(hspace=0.3)
        
        # Plot Q-values
        axes[0, 0].plot(np.arange(len(self.training_metrics['q_values'])) * self.config.log_freq, 
                       self.training_metrics['q_values'])
        axes[0, 0].set_title('Average Q-values')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Q-value')
        
        # Plot losses
        axes[0, 1].plot(np.arange(len(self.training_metrics['critic_loss'])) * self.config.log_freq, 
                       self.training_metrics['critic_loss'], label='Critic Loss')
        
        # Filter out zero actor losses (from delayed updates)
        actor_losses = np.array(self.training_metrics['actor_loss'])
        actor_steps = np.arange(len(actor_losses)) * self.config.log_freq
        non_zero_indices = actor_losses != 0
        axes[0, 1].plot(actor_steps[non_zero_indices], actor_losses[non_zero_indices], label='Actor Loss')
        
        axes[0, 1].set_title('Training Losses')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Plot episode rewards
        axes[1, 0].plot(self.training_metrics['episode_rewards'])
        axes[1, 0].set_title('Episode Rewards')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Reward')
        
        # Plot exploration noise
        axes[1, 1].plot(np.arange(len(self.training_metrics['exploration_noise'])) * self.config.log_freq, 
                       self.training_metrics['exploration_noise'])
        axes[1, 1].set_title('Exploration Noise')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Noise Scale')
        
        # Save figure and metrics
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            plt.savefig(os.path.join(save_dir, f"gat_td3_training_metrics_{int(time.time())}.png"))
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame({
                'step': np.arange(0, len(self.training_metrics['q_values'])) * self.config.log_freq,
                'q_value': self.training_metrics['q_values'],
                'critic_loss': self.training_metrics['critic_loss'],
                'actor_loss': self.training_metrics['actor_loss'],
                'exploration_noise': self.training_metrics['exploration_noise'],
                'timestamp': self.training_metrics['timestamp']
            })
            metrics_df.to_csv(os.path.join(save_dir, f"gat_td3_training_metrics_{int(time.time())}.csv"), index=False)
            
            # Save episode metrics to CSV
            episode_df = pd.DataFrame({
                'episode': np.arange(0, len(self.training_metrics['episode_rewards'])),
                'reward': self.training_metrics['episode_rewards'],
                'length': self.training_metrics['episode_lengths']
            })
            episode_df.to_csv(os.path.join(save_dir, f"gat_td3_episode_metrics_{int(time.time())}.csv"), index=False)
            
        else:
            plt.show()


# GAT-TD3 Trainer class
class GATTD3Trainer:
    def __init__(self, env, config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = torch.device(device)
        
        # Load configuration
        if config is None:
            self.config = GATTD3Config()
        elif isinstance(config, str):
            self.config = GATTD3Config(config)
        else:
            self.config = config
            
        # Get state and action dimensions
        self.state_dim = env.flat_observation_space.shape[0]
        
        # Check if action space is continuous (Box)
        assert isinstance(env.action_space, gym.spaces.Box), "GAT-TD3 requires continuous action space"
        
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        
        # Initialize agent
        self.agent = GATTD3Agent(
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
            
        print(f"Starting GAT-TD3 training for {total_timesteps} timesteps...")
        
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
                
                # Save model
                if timesteps_elapsed % self.config.save_freq == 0:
                    save_dir = os.path.join("models", f"gat_td3_step_{timesteps_elapsed}")
                    self.agent.save_model(save_dir)
                
                # Evaluate policy
                if timesteps_elapsed % self.config.eval_freq == 0:
                    eval_reward = self.evaluate()
                    print(f"Step {timesteps_elapsed}: Eval reward = {eval_reward:.2f}")
                    
                    # Save best model
                    if eval_reward > self.best_eval_reward:
                        self.best_eval_reward = eval_reward
                        save_dir = os.path.join("models", "gat_td3_best")
                        self.agent.save_model(save_dir)
                
                # Print progress
                if timesteps_elapsed % (self.config.log_freq * 10) == 0:
                    time_elapsed = time.time() - self.start_time
                    fps = timesteps_elapsed / time_elapsed
                    avg_reward = np.mean(self.agent.episode_rewards[-10:]) if len(self.agent.episode_rewards) > 0 else 0
                    
                    print(f"Steps: {timesteps_elapsed}/{total_timesteps} | "
                          f"FPS: {fps:.2f} | "
                          f"Avg Reward: {avg_reward:.2f} | "
                          f"Noise: {self.agent.current_action_noise:.3f}")
            
            # End of episode processing
            self.agent.end_episode()
            
            # Reset environment for next episode
            state, _ = self.env.reset()
            
            # Print episode stats
            print(f"Episode {episode_num}: steps = {episode_step}, reward = {episode_reward:.2f}")
        
        # Final save
        save_dir = os.path.join("models", "gat_td3_final")
        self.agent.save_model(save_dir)
        
        # Plot and save metrics
        self.agent.plot_metrics(save_dir="metrics")
        
        # Print final stats
        time_elapsed = time.time() - self.start_time
        avg_reward = np.mean(self.agent.episode_rewards[-10:]) if len(self.agent.episode_rewards) > 0 else 0
        
        print(f"Training complete: {total_timesteps} steps in {time_elapsed:.2f} seconds ({total_timesteps/time_elapsed:.2f} FPS)")
        print(f"Final average reward (last 10 episodes): {avg_reward:.2f}")
        print(f"Best evaluation reward: {self.best_eval_reward:.2f}")
        
        return self.agent
    
    def evaluate(self, n_episodes=5, render=False, deterministic=False):
        """Evaluate the agent"""
        total_rewards = []
        
        for i in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Select action without noise if deterministic, otherwise use eval_mode
                action = self.agent.select_action(state, eval_mode=deterministic)
                
                # Take step in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                # Render if requested
                if render:
                    self.env.render()
            
            total_rewards.append(total_reward)
        
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        print(f"Evaluation over {n_episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        return mean_reward 