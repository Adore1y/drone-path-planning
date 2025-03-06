#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估工具函数
用于计算性能指标和可视化结果
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.gridspec import GridSpec

class PerformanceMetrics:
    """性能指标收集和分析类"""
    
    def __init__(self):
        """初始化性能指标收集器"""
        self.results = {}
    
    def add_episode_result(self, algorithm, metrics):
        """
        添加一次实验的结果
        
        参数:
            algorithm: 算法名称
            metrics: 性能指标字典
        """
        if algorithm not in self.results:
            self.results[algorithm] = []
        
        self.results[algorithm].append(metrics)
    
    def compute_summary(self):
        """
        计算所有实验的汇总统计信息
        
        返回:
            summary: 汇总统计信息字典
        """
        summary = {}
        
        for algorithm, episodes in self.results.items():
            summary[algorithm] = {}
            
            # 合并所有实验的指标
            all_metrics = {}
            for episode in episodes:
                for metric, value in episode.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
            
            # 计算每个指标的统计信息
            for metric, values in all_metrics.items():
                values = np.array(values)
                summary[algorithm][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values)
                }
        
        return summary
    
    def save_results(self, filename):
        """
        将结果保存到CSV文件
        
        参数:
            filename: 输出文件名
        """
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 准备数据
        data = []
        
        for algorithm, episodes in self.results.items():
            for i, episode in enumerate(episodes):
                row = {"algorithm": algorithm, "episode": i+1}
                row.update(episode)
                data.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        
        print(f"Results saved to {filename}")
    
    def plot_comparison(self, metrics_to_plot=None, save_fig=None):
        """
        绘制算法比较图
        
        参数:
            metrics_to_plot: 要绘制的指标列表
            save_fig: 保存图像的文件名
        """
        summary = self.compute_summary()
        
        # 如果未指定指标，则使用所有可用指标
        if metrics_to_plot is None:
            # 获取第一个算法的所有指标
            first_alg = list(summary.keys())[0]
            metrics_to_plot = list(summary[first_alg].keys())
        
        # 设置图表
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        # 为每个指标创建柱状图
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # 准备数据
            algorithms = []
            means = []
            stds = []
            
            for algorithm, metrics in summary.items():
                if metric in metrics:
                    algorithms.append(algorithm)
                    means.append(metrics[metric]["mean"])
                    stds.append(metrics[metric]["std"])
            
            # 绘制柱状图
            bars = ax.bar(algorithms, means, yerr=stds, capsize=10)
            
            # 添加数值标签
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1*max(means),
                        f'{mean:.2f}', ha='center', va='bottom')
            
            # 设置标题和标签
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_xlabel('Algorithm')
            
            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图像
        if save_fig:
            os.makedirs(os.path.dirname(save_fig), exist_ok=True)
            plt.savefig(save_fig, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_fig}")
        
        plt.close()

def calculate_energy_consumption(path, hover_power=250, forward_power_factor=1.5, vertical_power_factor=2.0):
    """
    计算路径的能量消耗
    
    参数:
        path: 路径点列表 [[x, y, z], ...]
        hover_power: 悬停功率 (W)
        forward_power_factor: 前进功率因子
        vertical_power_factor: 垂直功率因子
        
    返回:
        energy: 能量消耗 (J)
    """
    if not path or len(path) < 2:
        return 0.0
    
    total_energy = 0.0
    avg_speed = 2.0  # 平均速度 (m/s)
    
    for i in range(1, len(path)):
        x1, y1, z1 = path[i-1]
        x2, y2, z2 = path[i]
        
        # 计算水平和垂直距离
        horizontal_dist = np.sqrt((x2-x1)**2 + (z2-z1)**2)
        vertical_dist = abs(y2-y1)
        
        # 计算时间
        horizontal_time = horizontal_dist / avg_speed if horizontal_dist > 0 else 0
        vertical_time = vertical_dist / (avg_speed/2) if vertical_dist > 0 else 0
        
        # 如果同时有水平和垂直移动，取较大的时间
        segment_time = max(horizontal_time, vertical_time)
        
        if segment_time == 0:
            continue
        
        # 计算功率
        # 水平移动功率
        horizontal_power = hover_power * forward_power_factor if horizontal_dist > 0 else 0
        
        # 垂直移动功率
        vertical_power = 0
        if vertical_dist > 0:
            # 上升比下降消耗更多能量
            if y2 > y1:  # 上升
                vertical_power = hover_power * vertical_power_factor
            else:  # 下降
                vertical_power = hover_power * 0.8
        
        # 总功率
        if horizontal_dist > 0 and vertical_dist > 0:
            # 如果同时有水平和垂直移动，取较大的功率并增加一些额外消耗
            total_power = max(horizontal_power, vertical_power) * 1.1
        else:
            total_power = horizontal_power + vertical_power
        
        # 如果没有移动，则使用悬停功率
        if total_power == 0:
            total_power = hover_power
        
        # 计算能量
        segment_energy = total_power * segment_time
        total_energy += segment_energy
    
    return total_energy

def compare_algorithms_visualization(algorithms, start_pos, goal_pos, buildings, boundaries, title=None, save_fig=None):
    """
    可视化比较不同算法的路径
    
    参数:
        algorithms: 算法名称和路径的字典 {"算法名": 路径点列表, ...}
        start_pos: 起始位置 [x, y, z]
        goal_pos: 目标位置 [x, y, z]
        buildings: 建筑物列表
        boundaries: 环境边界
        title: 图表标题
        save_fig: 保存图像的文件名
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制建筑物
    min_x, min_z, max_x, max_z = boundaries
    
    for building in buildings:
        bx, by, width, height, length = building
        
        # 创建立方体的顶点
        x = [bx - width/2, bx + width/2]
        y = [0, height]  # 假设建筑物从地面开始
        z = [by - length/2, by + length/2]
        
        # 绘制立方体
        xx, yy = np.meshgrid(x, y)
        ax.plot_surface(xx, yy*0+z[0], np.ones_like(xx)*z[0], alpha=0.2, color='gray')
        ax.plot_surface(xx, yy*0+z[1], np.ones_like(xx)*z[1], alpha=0.2, color='gray')
        ax.plot_surface(np.ones_like(xx)*x[0], yy*0+z, xx*0+y[0], alpha=0.2, color='gray')
        ax.plot_surface(np.ones_like(xx)*x[1], yy*0+z, xx*0+y[0], alpha=0.2, color='gray')
        ax.plot_surface(xx, np.ones_like(xx)*y[1], yy*0+z, alpha=0.2, color='gray')
    
    # 绘制起点和终点
    ax.scatter(start_pos[0], start_pos[2], start_pos[1], c='g', marker='o', s=100, label='Start')
    ax.scatter(goal_pos[0], goal_pos[2], goal_pos[1], c='r', marker='*', s=100, label='Goal')
    
    # 绘制不同算法的路径
    colors = ['b', 'orange', 'purple', 'cyan', 'magenta']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (algorithm, path) in enumerate(algorithms.items()):
        if path:
            # 提取路径点的坐标
            xs = [p[0] for p in path]
            ys = [p[2] for p in path]  # z坐标作为y轴
            zs = [p[1] for p in path]  # y坐标作为z轴
            
            # 绘制路径
            ax.plot(xs, ys, zs, c=colors[i % len(colors)], marker=markers[i % len(markers)], 
                   markersize=4, label=f'{algorithm} (len={len(path)})')
    
    # 设置坐标轴标签
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Y (m)')
    
    # 设置坐标轴范围
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_z, max_z)
    ax.set_zlim(0, 30)  # 假设最高建筑物不超过30米
    
    # 设置标题
    if title:
        ax.set_title(title)
    
    # 添加图例
    ax.legend()
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    
    # 保存图像
    if save_fig:
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)
        plt.savefig(save_fig, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_fig}")
    
    plt.close()

def create_scenario_grid_visualization(scenario_results, metrics_to_plot, title=None, save_fig=None):
    """
    创建跨场景比较的网格可视化
    
    参数:
        scenario_results: 场景结果字典 {"场景名": {"算法名": {"指标名": 值, ...}, ...}, ...}
        metrics_to_plot: 要绘制的指标列表
        title: 图表标题
        save_fig: 保存图像的文件名
    """
    n_scenarios = len(scenario_results)
    n_metrics = len(metrics_to_plot)
    
    # 创建网格布局
    fig = plt.figure(figsize=(5*n_metrics, 4*n_scenarios))
    gs = GridSpec(n_scenarios, n_metrics, figure=fig)
    
    # 获取所有算法
    all_algorithms = set()
    for scenario, alg_results in scenario_results.items():
        all_algorithms.update(alg_results.keys())
    all_algorithms = sorted(list(all_algorithms))
    
    # 为每个场景和指标创建子图
    for i, scenario in enumerate(scenario_results.keys()):
        for j, metric in enumerate(metrics_to_plot):
            ax = fig.add_subplot(gs[i, j])
            
            # 准备数据
            algorithms = []
            values = []
            
            for algorithm in all_algorithms:
                if algorithm in scenario_results[scenario]:
                    if metric in scenario_results[scenario][algorithm]:
                        algorithms.append(algorithm)
                        values.append(scenario_results[scenario][algorithm][metric])
            
            # 绘制柱状图
            bars = ax.bar(algorithms, values, color=sns.color_palette("husl", len(algorithms)))
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05*max(values) if max(values) > 0 else 0.05,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=8)
            
            # 设置标题和标签
            if i == 0:
                ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
            if j == 0:
                ax.set_ylabel(f'{scenario.capitalize()} Scenario', fontsize=12)
            
            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            
            # 调整y轴范围，确保所有子图在同一指标上使用相同的比例
            if j > 0:
                prev_ax = fig.add_subplot(gs[i, j-1])
                ax.set_ylim(prev_ax.get_ylim())
    
    # 设置总标题
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    # 保存图像
    if save_fig:
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)
        plt.savefig(save_fig, dpi=300, bbox_inches='tight')
        print(f"Grid visualization saved to {save_fig}")
    
    plt.close()

if __name__ == "__main__":
    # 测试代码
    pass 