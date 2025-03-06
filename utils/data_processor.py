#!/usr/bin/env python3
"""
数据处理相关的实用工具函数
"""
import numpy as np
import pandas as pd
import json
import os

def load_trajectory_data(file_path):
    """加载轨迹数据"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path).values.tolist()
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def save_trajectory_data(data, file_path, format='csv'):
    """保存轨迹数据"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if format == 'csv':
        df = pd.DataFrame(data, columns=['x', 'y', 'z'])
        df.to_csv(file_path, index=False)
    elif format == 'json':
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")

def calculate_path_metrics(path):
    """计算路径的各种指标"""
    if not path or len(path) < 2:
        return {'length': 0, 'avg_altitude': 0, 'max_altitude': 0}
    
    # 计算路径长度
    length = 0
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        segment_length = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
        length += segment_length
    
    # 计算高度指标
    altitudes = [p[2] for p in path]
    avg_altitude = sum(altitudes) / len(altitudes)
    max_altitude = max(altitudes)
    
    return {
        'length': length,
        'avg_altitude': avg_altitude,
        'max_altitude': max_altitude
    }

def merge_metrics_data(metrics_files):
    """合并多个指标数据文件"""
    merged_data = []
    
    for file_path in metrics_files:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            merged_data.append(df)
    
    if merged_data:
        return pd.concat(merged_data, ignore_index=True)
    else:
        return pd.DataFrame()
