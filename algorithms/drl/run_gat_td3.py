#!/usr/bin/env python3
"""
GAT-TD3训练脚本
用于训练基于图注意力网络的TD3深度强化学习算法进行无人机能量高效路径规划
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import gym
from datetime import datetime

# 修复导入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入算法和环境
from algorithms.drl.gat_td3 import GATTD3Config, GATTD3Trainer
from algorithms.drl.env_wrapper import create_env

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GAT-TD3 无人机训练脚本')
    
    # 训练参数
    parser.add_argument('-t', '--timesteps', type=int, default=100000,
                        help='训练总步数')
    parser.add_argument('-e', '--env', type=str, default='urban',
                        help='环境类型 (urban, forest, etc.)')
    parser.add_argument('-c', '--config', type=str, default='algorithms/drl/configs/gat_td3_config.json',
                        help='GAT-TD3配置文件路径')
    parser.add_argument('-o', '--output', type=str, default='training_results',
                        help='输出目录')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='随机种子')
    
    # 环境参数
    parser.add_argument('--headless', action='store_true',
                        help='使用无头模式运行Webots')
    parser.add_argument('--no-normalize', action='store_true',
                        help='不对观测进行归一化')
    parser.add_argument('--env-config', type=str, default='algorithms/drl/configs/env_config.json',
                        help='环境配置文件路径')
    
    # 模型参数
    parser.add_argument('--no-graph', action='store_true',
                        help='不使用图表示')
    parser.add_argument('--no-energy', action='store_true',
                        help='不使用能量模型')
    parser.add_argument('--load', type=str, default=None,
                        help='加载已训练的模型路径')
    
    # 评估参数
    parser.add_argument('--eval-only', action='store_true',
                        help='仅进行评估，不训练')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='评估时的情节数')
    parser.add_argument('--render', action='store_true',
                        help='渲染评估过程')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"gattd3_{args.env}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 打印训练参数
    print("=" * 40)
    print("无人机GAT-DRL训练启动")
    print("=" * 40)
    print(f"算法: GAT-TD3")
    print(f"环境: {args.env}")
    print(f"训练步数: {args.timesteps}")
    print(f"配置文件: {args.config}")
    print(f"输出目录: {args.output}")
    print(f"工作目录: {os.getcwd()}")
    
    # 加载GAT-TD3配置
    config = GATTD3Config(args.config)
    
    # 根据命令行参数修改配置
    if args.no_graph:
        config.use_graph_representation = False
    if args.no_energy:
        config.use_energy_model = False
    config.total_timesteps = args.timesteps
    
    # 保存修改后的配置
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        config_dict = {k: v for k, v in config.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        json.dump(config_dict, f, indent=2)
    
    # 创建环境
    env = create_env(
        config_file=args.env_config,
        headless=args.headless or args.eval_only,  # 评估时总是使用无头模式
        flat_observation=not args.no_normalize
    )
    
    # 创建训练器
    trainer = GATTD3Trainer(env, config)
    
    # 加载已训练的模型
    if args.load:
        trainer.agent.load(args.load)
        print(f"已加载模型: {args.load}")
    
    # 根据模式选择训练或评估
    if args.eval_only:
        # 仅评估
        print(f"仅评估模式，情节数: {args.eval_episodes}")
        mean_reward, mean_energy, mean_efficiency = trainer.evaluate(
            n_episodes=args.eval_episodes,
            render=args.render
        )
        print(f"评估结果:")
        print(f"平均奖励: {mean_reward:.2f}")
        print(f"平均能量消耗: {mean_energy:.2f} J")
        print(f"平均能量效率: {mean_efficiency:.2f} m/kJ")
    else:
        # 训练
        try:
            agent = trainer.train()
            
            # 训练完成后进行评估
            print("训练完成！进行最终评估...")
            mean_reward, mean_energy, mean_efficiency = trainer.evaluate(
                n_episodes=args.eval_episodes,
                render=args.render
            )
            print(f"评估结果:")
            print(f"平均奖励: {mean_reward:.2f}")
            print(f"平均能量消耗: {mean_energy:.2f} J")
            print(f"平均能量效率: {mean_efficiency:.2f} m/kJ")
            
            # 保存结果到文件
            with open(os.path.join(output_dir, "final_results.json"), 'w') as f:
                json.dump({
                    'reward': mean_reward,
                    'energy': mean_energy,
                    'efficiency': mean_efficiency,
                    'timestamp': time.time()
                }, f, indent=2)
                
            print(f"训练和评估完成。模型保存在 {output_dir}")
            
        except KeyboardInterrupt:
            print("训练被用户中断")
        finally:
            # 关闭环境
            env.close()
    
    print("=" * 40)
    print("训练过程完成！")
    print(f"结果保存到: {output_dir}")
    print("=" * 40)

if __name__ == "__main__":
    main() 