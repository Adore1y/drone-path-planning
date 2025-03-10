#!/usr/bin/env python3

"""
Main Training Script for Drone Path Planning DRL Framework
Supports multiple algorithms and environment configurations
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import environment and algorithms
from env_wrapper import create_env, WebotsMAVDroneEnv
from ppo_algorithm import PPOTrainer, PPOConfig
from dqn_algorithm import DQNTrainer, DQNConfig
from td3_algorithm import TD3Trainer, TD3Config
# Import GAT-TD3 algorithm
from gat_td3_algorithm import GATTD3Trainer, GATTD3Config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DRL agents for drone path planning')
    
    # Environment settings
    parser.add_argument('--env_config', type=str, default=None,
                        help='Path to environment configuration file')
    parser.add_argument('--headless', action='store_true',
                        help='Run Webots in headless mode')
    parser.add_argument('--real_time_factor', type=float, default=1.0,
                        help='Simulation speed factor (1.0 = real-time)')
    parser.add_argument('--environment', type=str, default='standard', choices=['standard', 'urban'],
                        help='Type of environment to use: standard or urban')
    
    # Algorithm selection
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'dqn', 'td3', 'gat_td3'],
                        help='DRL algorithm to use: ppo, dqn, td3, or gat_td3')
    parser.add_argument('--algo_config', type=str, default=None,
                        help='Path to algorithm configuration file')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total timesteps to train (default: use algo config)')
    parser.add_argument('--eval_episodes', type=int, default=5,
                        help='Number of episodes for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='training_results',
                        help='Directory to save results')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: algo_timestamp)')
    
    # Model loading/saving
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to model checkpoint to load')
    parser.add_argument('--eval_only', action='store_true',
                        help='Run evaluation only, no training')
    
    return parser.parse_args()

def setup_experiment_dir(args):
    """Setup experiment directory for saving results"""
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.exp_name:
        exp_dir = f"{args.exp_name}_{timestamp}"
    else:
        exp_dir = f"{args.algo}_{args.environment}_{timestamp}"
    
    exp_path = os.path.join(args.output_dir, exp_dir)
    os.makedirs(exp_path, exist_ok=True)
    
    # Create subdirectories
    model_dir = os.path.join(exp_path, "models")
    log_dir = os.path.join(exp_path, "logs")
    config_dir = os.path.join(exp_path, "configs")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    
    # Save command line arguments
    with open(os.path.join(config_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    return {
        "exp_path": exp_path,
        "model_dir": model_dir,
        "log_dir": log_dir,
        "config_dir": config_dir
    }

def customize_env_config(args, env_config):
    """Customize environment configuration based on environment type"""
    if args.environment == 'urban':
        # Update the configuration for urban environment
        env_config["world_file"] = "../../webots/worlds/urban_environment.wbt"
        env_config["waypoints_file"] = "../../webots/worlds/urban_waypoints.txt"
        
        # Adjust reward weights for urban environment
        if "reward_weights" in env_config:
            env_config["reward_weights"]["collision"] = -150.0  # Higher penalty for collisions in urban env
            env_config["reward_weights"]["altitude_control"] = -0.5  # Add penalty for altitude control
    
    return env_config

def create_trainer(args, env, dirs):
    """Create appropriate trainer based on algorithm selection"""
    if args.algo == 'ppo':
        # Load PPO configuration if provided
        if args.algo_config:
            config = PPOConfig(args.algo_config)
        else:
            config = PPOConfig()
            
        # Save configuration
        config.save_config(os.path.join(dirs["config_dir"], "ppo_config.json"))
        
        # Create PPO trainer
        trainer = PPOTrainer(env, config=config)
        
        # Load model if specified
        if args.load_model:
            trainer.agent.load_model(args.load_model)
    
    elif args.algo == 'dqn':
        # Load DQN configuration if provided
        if args.algo_config:
            config = DQNConfig(args.algo_config)
        else:
            config = DQNConfig()
        
        # Save configuration
        config.save_config(os.path.join(dirs["config_dir"], "dqn_config.json"))
        
        # Create DQN trainer
        trainer = DQNTrainer(env, config=config)
        
        # Load model if specified
        if args.load_model:
            trainer.agent.load_model(args.load_model)
            
    elif args.algo == 'td3':
        # Load TD3 configuration if provided
        if args.algo_config:
            config = TD3Config(args.algo_config)
        else:
            config = TD3Config()
        
        # Save configuration
        config.save_config(os.path.join(dirs["config_dir"], "td3_config.json"))
        
        # Create TD3 trainer
        trainer = TD3Trainer(env, config=config)
        
        # Load model if specified
        if args.load_model:
            trainer.agent.load_model(args.load_model)
    
    elif args.algo == 'gat_td3':
        # Load GAT-TD3 configuration if provided
        if args.algo_config:
            config = GATTD3Config(args.algo_config)
        else:
            config = GATTD3Config()
        
        # Save configuration
        config.save_config(os.path.join(dirs["config_dir"], "gat_td3_config.json"))
        
        # Create GAT-TD3 trainer
        trainer = GATTD3Trainer(env, config=config)
        
        # Load model if specified
        if args.load_model:
            trainer.agent.load_model(args.load_model)
    
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")
    
    return trainer

def main():
    """Main function to run training"""
    # Parse arguments
    args = parse_args()
    
    # Setup experiment directories
    dirs = setup_experiment_dir(args)
    
    print(f"Training with algorithm: {args.algo}")
    print(f"Environment type: {args.environment}")
    print(f"Experiment directory: {dirs['exp_path']}")
    
    # Load environment config
    env_config = None
    env_config_path = None
    if args.env_config:
        with open(args.env_config, 'r') as f:
            env_config = json.load(f)
        
        # Customize environment config based on environment type
        env_config = customize_env_config(args, env_config)
        
        # Save the modified config to a new file
        env_config_path = os.path.join(dirs["config_dir"], "env_config.json")
        with open(env_config_path, 'w') as f:
            json.dump(env_config, f, indent=4)
    
    # Create environment
    env = create_env(
        config_file=env_config_path,  # Pass the file path, not the config dict
        headless=args.headless,
        real_time_factor=args.real_time_factor,
        flat_observation=True  # Flatten observation for easier use with DRL algorithms
    )
    
    try:
        # Create trainer
        trainer = create_trainer(args, env, dirs)
        
        # Evaluation only mode
        if args.eval_only:
            print("Running evaluation...")
            eval_reward = trainer.evaluate(
                n_episodes=args.eval_episodes,
                render=not args.headless,
                deterministic=True
            )
            
            print(f"Evaluation average reward: {eval_reward:.2f}")
            
            # Save evaluation results
            eval_results = {
                'mean_reward': float(eval_reward),
                'num_episodes': args.eval_episodes,
                'environment': args.environment,
                'algorithm': args.algo
            }
            
            with open(os.path.join(dirs["log_dir"], "eval_results.json"), 'w') as f:
                json.dump(eval_results, f, indent=4)
            
            print("Evaluation complete.")
            
        else:
            # Training mode
            print("Starting training...")
            start_time = time.time()
            
            # Train agent
            agent = trainer.train(total_timesteps=args.timesteps)
            
            # Calculate training time
            total_time = time.time() - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            # Final evaluation
            print("Running final evaluation...")
            eval_reward = trainer.evaluate(
                n_episodes=args.eval_episodes,
                render=not args.headless,
                deterministic=True
            )
            
            print(f"Final evaluation average reward: {eval_reward:.2f}")
            
            # Save final model
            final_model_path = os.path.join(dirs["model_dir"], f"{args.algo}_final")
            if args.algo == 'ppo':
                trainer.agent.save_model(final_model_path, prefix='ppo')
            elif args.algo == 'dqn':
                trainer.agent.save_model(final_model_path, prefix='dqn')
            elif args.algo == 'td3':
                trainer.agent.save_model(final_model_path, prefix='td3')
            else:  # GAT-TD3
                trainer.agent.save_model(final_model_path, prefix='gat_td3')
            
            print(f"Final model saved to {final_model_path}")
            print("Training and evaluation complete.")
    
    finally:
        # Clean up environment
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main() 