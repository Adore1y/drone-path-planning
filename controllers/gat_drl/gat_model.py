#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图注意力网络(GAT)模型
实现了用于无人机路径规划的GAT结构和LSTM时间建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np

class GraphAttentionLayer(nn.Module):
    """标准图注意力层"""
    
    def __init__(self, in_features, out_features, num_heads, dropout, alpha, concat=True):
        """
        初始化GAT层
        
        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
            num_heads: 注意力头数
            dropout: dropout率
            alpha: LeakyReLU负斜率参数
            concat: 是否连接多头注意力结果（True）或平均（False）
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        self.gat = GATConv(in_features, out_features, heads=num_heads, 
                           dropout=dropout, concat=concat)
        
    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
            x: 节点特征矩阵 [节点数, 特征维度]
            edge_index: 边索引 [2, 边数]
            
        返回:
            x: 更新后的节点特征 [节点数, out_features * num_heads] 或 [节点数, out_features]
        """
        x = self.gat(x, edge_index)
        return x

class TemporalAttentionLayer(nn.Module):
    """时间注意力层，用于关注历史状态"""
    
    def __init__(self, feature_dim, hidden_dim):
        """
        初始化时间注意力层
        
        参数:
            feature_dim: 特征维度
            hidden_dim: 隐藏层维度
        """
        super(TemporalAttentionLayer, self).__init__()
        
        self.lstm = nn.LSTM(input_size=feature_dim, 
                            hidden_size=hidden_dim, 
                            batch_first=True)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
    
    def forward(self, x, memory=None):
        """
        前向传播
        
        参数:
            x: 当前特征 [batch_size, feature_dim]
            memory: 历史特征序列 [batch_size, seq_len, feature_dim] 或 None
            
        返回:
            out: 融合时间信息的特征 [batch_size, hidden_dim]
            new_memory: 更新的历史特征序列 [batch_size, seq_len, feature_dim]
        """
        batch_size = x.size(0)
        
        # 如果没有历史记忆，初始化
        if memory is None:
            memory = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
        else:
            # 添加当前特征到历史序列末尾
            memory = torch.cat([memory, x.unsqueeze(1)], dim=1)  # [batch_size, seq_len+1, feature_dim]
            
            # 限制历史长度
            if memory.size(1) > 10:  # 保留最近10个时间步
                memory = memory[:, -10:, :]
        
        # LSTM处理序列
        lstm_out, _ = self.lstm(memory)  # [batch_size, seq_len, hidden_dim]
        
        # 注意力权重
        attn_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 加权求和
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_out)  # [batch_size, 1, hidden_dim]
        out = context.squeeze(1)  # [batch_size, hidden_dim]
        
        return out, memory

class GAT_LSTM(nn.Module):
    """结合GAT和LSTM的模型，用于环境感知和时间建模"""
    
    def __init__(self, num_node_features, hidden_dim, output_dim, num_heads=8, dropout=0.1, alpha=0.2):
        """
        初始化GAT-LSTM模型
        
        参数:
            num_node_features: 节点特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_heads: 注意力头数
            dropout: dropout率
            alpha: LeakyReLU负斜率参数
        """
        super(GAT_LSTM, self).__init__()
        
        # GAT层
        self.gat1 = GraphAttentionLayer(num_node_features, hidden_dim, num_heads, 
                                        dropout, alpha, concat=True)
        
        # 如果连接多头结果，第二层输入维度为hidden_dim*num_heads
        self.gat2 = GraphAttentionLayer(hidden_dim * num_heads, hidden_dim, 1, 
                                        dropout, alpha, concat=False)
        
        # 时间注意力层
        self.temporal_attn = TemporalAttentionLayer(hidden_dim, hidden_dim)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 可学习的融合参数
        self.lambda_param = nn.Parameter(torch.tensor(0.5))
        
        # 记忆状态
        self.memory = None
        
    def forward(self, data):
        """
        前向传播
        
        参数:
            data: PyTorch Geometric的Data对象，包含x和edge_index
            
        返回:
            out: 节点嵌入 [节点数, output_dim]
        """
        x, edge_index = data.x, data.edge_index
        
        # GAT层
        x_gat = self.gat1(x, edge_index)
        x_gat = F.relu(x_gat)
        x_gat = F.dropout(x_gat, p=0.1, training=self.training)
        x_gat = self.gat2(x_gat, edge_index)
        
        # 时间注意力处理
        batch_size = 1  # 默认批大小为1
        nodes = x_gat.size(0)
        
        # 处理所有节点
        all_temporal_features = []
        all_new_memories = []
        
        for i in range(nodes):
            node_feature = x_gat[i:i+1]  # 取一个节点的特征
            
            # 获取该节点的历史记忆（如果有）
            node_memory = None
            if self.memory is not None and i < len(self.memory):
                node_memory = self.memory[i]
                
            # 时间注意力处理
            temporal_feature, new_memory = self.temporal_attn(node_feature, node_memory)
            
            all_temporal_features.append(temporal_feature)
            all_new_memories.append(new_memory)
            
        # 更新记忆
        self.memory = all_new_memories
        
        # 将所有节点的时间特征堆叠
        x_temporal = torch.cat(all_temporal_features, dim=0)  # [nodes, hidden_dim]
        
        # 特征融合
        lambda_val = torch.sigmoid(self.lambda_param)  # 将参数映射到(0,1)范围
        
        combined_features = torch.cat([
            lambda_val * x_gat,
            (1 - lambda_val) * x_temporal
        ], dim=1)
        
        # 最终输出
        out = self.fc(combined_features)
        
        return out
    
    def reset_memory(self):
        """重置时间记忆"""
        self.memory = None

class GATEnvironmentEncoder:
    """使用GAT对环境进行编码的封装类"""
    
    def __init__(self, num_node_features=2, hidden_dim=64, output_dim=32, device='cpu'):
        """
        初始化环境编码器
        
        参数:
            num_node_features: 节点特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            device: 计算设备
        """
        self.model = GAT_LSTM(num_node_features, hidden_dim, output_dim)
        self.device = device
        self.model.to(device)
        self.node_positions = None
        self.node_embeddings = None
        
    def encode_environment(self, graph, node_positions=None, node_features=None):
        """
        对环境进行编码
        
        参数:
            graph: networkx图
            node_positions: 节点位置字典 {node_id: (x, y)}
            node_features: 节点特征字典 {node_id: [特征]}
            
        返回:
            node_embeddings: 节点嵌入字典 {node_id: [嵌入向量]}
        """
        from utils.graph_utils import convert_nx_graph_to_pyg_graph
        
        # 将networkx图转换为PyG图
        pyg_data = convert_nx_graph_to_pyg_graph(graph, node_positions, node_features)
        pyg_data = pyg_data.to(self.device)
        
        # 编码
        self.model.eval()
        with torch.no_grad():
            node_embeddings_tensor = self.model(pyg_data)
        
        # 将张量转换为字典
        node_embeddings = {}
        for i, node in enumerate(graph.nodes()):
            node_embeddings[node] = node_embeddings_tensor[i].cpu().numpy()
        
        # 保存结果
        self.node_positions = node_positions
        self.node_embeddings = node_embeddings
        
        return node_embeddings
    
    def get_embedding_for_position(self, position):
        """
        获取给定位置的嵌入
        
        参数:
            position: 位置坐标 (x, y)
            
        返回:
            embedding: 嵌入向量
        """
        if self.node_positions is None or self.node_embeddings is None:
            raise ValueError("Environment not encoded yet")
        
        # 找到最近的节点
        min_dist = float('inf')
        nearest_node = None
        
        for node, pos in self.node_positions.items():
            dist = np.sqrt((pos[0] - position[0])**2 + (pos[1] - position[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        if nearest_node is None:
            raise ValueError("Could not find nearest node")
        
        return self.node_embeddings[nearest_node]
    
    def reset(self):
        """重置模型记忆"""
        self.model.reset_memory()
    
    def save(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        """加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval() 