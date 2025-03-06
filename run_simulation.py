#!/usr/bin/env python3
"""
无人机路径规划仿真运行脚本
支持两种模式：
- mock：使用Python直接生成模拟数据
- webots：使用Webots物理仿真
"""

import os
import sys
import argparse
import subprocess
import time
import json

def run_mock_simulation(algorithm="GAT-DRL", scenario="mixed", num_waypoints=None, output_dir="results"):
    """
    运行模拟仿真
    
    参数:
        algorithm: 算法名称 (默认: GAT-DRL)
        scenario: 场景类型 [sparse, mixed, dense] (默认: mixed)
        num_waypoints: 必经点数量 (默认根据场景自动设置)
        output_dir: 输出目录 (默认: results)
    
    返回:
        运行结果状态码
    """
    # 设置默认必经点数量
    if num_waypoints is None:
        if scenario == "sparse":
            num_waypoints = 4
        elif scenario == "dense":
            num_waypoints = 6
        else:  # mixed
            num_waypoints = 5
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建命令
    cmd = [
        sys.executable,
        "generate_mock_data.py",
        "--algorithm", algorithm,
        "--scenario", scenario,
        "--num_waypoints", str(num_waypoints),
        "--output_dir", output_dir
    ]
    
    print(f"运行模拟仿真: {' '.join(cmd)}")
    
    # 执行命令
    process = subprocess.run(cmd)
    
    if process.returncode == 0:
        print(f"模拟仿真完成，结果保存在{output_dir}目录中")
    else:
        print(f"模拟仿真失败，错误码: {process.returncode}")
    
    return process.returncode

def run_webots_simulation(algorithm="GAT-DRL", scenario="mixed", num_waypoints=None, output_dir="results"):
    """
    运行Webots物理仿真
    
    参数:
        algorithm: 算法名称 (默认: GAT-DRL)
        scenario: 场景类型 [sparse, mixed, dense] (默认: mixed)
        num_waypoints: 必经点数量 (默认根据场景自动设置)
        output_dir: 输出目录 (默认: results)
    
    返回:
        运行结果状态码
    """
    # 设置默认必经点数量
    if num_waypoints is None:
        if scenario == "sparse":
            num_waypoints = 4
        elif scenario == "dense":
            num_waypoints = 6
        else:  # mixed
            num_waypoints = 5
    
    # 检查Webots是否已安装
    webots_cmd = "webots"
    
    # 检查目录结构
    webots_world_path = f"webots/worlds/{scenario}_scenario.wbt"
    if not os.path.exists(webots_world_path):
        print(f"未找到Webots世界文件: {webots_world_path}")
        print("运行 'python setup_webots.py' 创建必要的Webots文件")
        return 1
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置环境变量
    env = os.environ.copy()
    env["ALGORITHM"] = algorithm
    env["SCENARIO"] = scenario
    env["NUM_WAYPOINTS"] = str(num_waypoints)
    env["OUTPUT_DIR"] = output_dir
    
    # 构建命令
    cmd = [webots_cmd, webots_world_path]
    
    print(f"运行Webots物理仿真: {' '.join(cmd)}")
    print(f"算法: {algorithm}, 场景: {scenario}, 必经点数量: {num_waypoints}")
    
    try:
        # 执行命令
        process = subprocess.run(cmd, env=env)
        
        if process.returncode == 0:
            print(f"Webots物理仿真完成")
        else:
            print(f"Webots物理仿真失败，错误码: {process.returncode}")
        
        return process.returncode
    except FileNotFoundError:
        print("错误: 找不到Webots可执行文件。请确保Webots已安装且可在命令行中访问。")
        return 1

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="无人机路径规划仿真")
    
    # 基本参数
    parser.add_argument("--mode", type=str, choices=["mock", "webots"], default="mock",
                      help="仿真模式: mock (基于Python快速模拟) 或 webots (物理仿真)")
    parser.add_argument("--algorithm", type=str, default="GAT-DRL",
                      help="路径规划算法 (默认: GAT-DRL)")
    parser.add_argument("--scenario", type=str, choices=["sparse", "mixed", "dense", "all"], default="mixed",
                      help="场景类型 (默认: mixed)")
    parser.add_argument("--num_waypoints", type=int, default=None,
                      help="必经点数量 (默认根据场景自动设置)")
    parser.add_argument("--output_type", type=str, choices=["prelim", "final", "custom"], default="prelim",
                      help="输出类型: prelim (前置模拟结果), final (最终结果), custom (自定义目录)")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="自定义输出目录 (仅当 --output_type=custom 时使用)")
    
    args = parser.parse_args()
    
    # 确定输出目录
    sim_type = "python" if args.mode == "mock" else "webots"
    
    if args.output_type == "prelim":
        output_dir = f"{sim_type}_prelim_results"
    elif args.output_type == "final":
        output_dir = f"{sim_type}_final_results"
    elif args.output_type == "custom" and args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"{sim_type}_results"  # 默认目录
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行相应模式的仿真
    if args.mode == "mock":
        return run_mock_simulation(args.algorithm, args.scenario, args.num_waypoints, output_dir)
    else:  # webots
        return run_webots_simulation(args.algorithm, args.scenario, args.num_waypoints, output_dir)

if __name__ == "__main__":
    sys.exit(main()) 