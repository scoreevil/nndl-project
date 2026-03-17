"""
局部表示+自注意力→RNN+注意力（增强型RNN）服饰描述生成模型
Model2: Enhanced CNN Encoder with Local Self-Attention + Attention-Enhanced LSTM Decoder
核心改进：
1. CNN编码器：新增局部特征自注意力层（3头自注意力）
2. LSTM解码器：新增标准注意力机制（LSTM隐藏状态与局部特征的注意力）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import random
import math


class LocalSelfAttention(nn.Module):
    """
    局部特征自注意力层（改进版）
    将7×7局部特征展平为49个序列（每个维度2048），使用多头自注意力增强局部特征
    改进：Pre-LayerNorm、前馈网络、更好的归一化
    """
    
    def __init__(self, feature_dim: int = 2048, num_heads: int = 8, dropout: float = 0.15, ffn_dim: int = None):
        """
        初始化局部自注意力层（改进版：增强正则化）
        
        Args:
            feature_dim: 特征维度，默认2048
            num_heads: 注意力头数，默认8（2048/8=256，可以整除）
            dropout: Dropout比例，默认0.15（从0.1增加到0.15，增强正则化）
            ffn_dim: 前馈网络维度，默认feature_dim * 4
        """
        super(LocalSelfAttention, self).__init__()
        
        assert feature_dim % num_heads == 0, f"feature_dim ({feature_dim}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.ffn_dim = ffn_dim if ffn_dim is not None else feature_dim * 4
        
        # Pre-LayerNorm（在注意力计算之前归一化）
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        
        # QKV线性层（添加Dropout增强正则化）
        self.q_linear = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Dropout(dropout * 0.5)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Dropout(dropout * 0.5)
        )
        self.v_linear = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Dropout(dropout * 0.5)
        )
        
        # 输出层
        self.output_linear = nn.Linear(feature_dim, feature_dim)
        
        # 前馈网络（FFN）- 增强正则化
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, self.ffn_dim),
            nn.GELU(),  # 使用GELU激活函数
            nn.Dropout(dropout),
            nn.Linear(self.ffn_dim, feature_dim),
            nn.Dropout(dropout)
        )
        
        # Dropout（增强正则化）
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout * 1.5)  # 注意力dropout稍微增加
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        # 初始化QKV线性层（需要从Sequential中获取Linear层）
        for module in self.q_linear:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        for module in self.k_linear:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        for module in self.v_linear:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.constant_(self.output_linear.bias, 0)
    
    def forward(self, local_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播：局部特征自注意力（改进版：Pre-LayerNorm + FFN）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
        
        Returns:
            自注意力增强后的局部特征 (batch, 2048, 7, 7)
        """
        batch_size, feat_dim, h, w = local_feat.size()
        
        # 展平：将7×7局部特征展平为49个序列
        # (batch, 2048, 7, 7) → (batch, 2048, 49) → (batch, 49, 2048)
        x = local_feat.view(batch_size, feat_dim, h * w)  # (batch, 2048, 49)
        x = x.transpose(1, 2)  # (batch, 49, 2048)
        
        # Pre-LayerNorm：在注意力计算之前归一化
        residual = x
        x = self.layer_norm1(x)
        
        # QKV计算（Sequential包含Dropout）
        # (batch, 49, 2048) → (batch, 49, 2048)
        Q = self.q_linear(x)  # (batch, 49, 2048)
        K = self.k_linear(x)   # (batch, 49, 2048)
        V = self.v_linear(x)   # (batch, 49, 2048)
        
        # 分多头
        # (batch, 49, 2048) → (batch, 49, num_heads, head_dim) → (batch, num_heads, 49, head_dim)
        Q = Q.view(batch_size, h * w, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 49, head_dim)
        K = K.view(batch_size, h * w, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 49, head_dim)
        V = V.view(batch_size, h * w, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 49, head_dim)
        
        # 计算注意力得分（Scaled Dot-Product Attention）
        # (batch, num_heads, 49, head_dim) × (batch, num_heads, head_dim, 49) → (batch, num_heads, 49, 49)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, 49, 49)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 加权求和
        # (batch, num_heads, 49, 49) × (batch, num_heads, 49, head_dim) → (batch, num_heads, 49, head_dim)
        attn_output = torch.matmul(attn_weights, V)  # (batch, num_heads, 49, head_dim)
        
        # 合并多头
        # (batch, num_heads, 49, head_dim) → (batch, 49, num_heads, head_dim) → (batch, 49, 2048)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, 49, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, h * w, feat_dim)  # (batch, 49, 2048)
        
        # 输出层
        attn_output = self.output_linear(attn_output)
        attn_output = self.dropout(attn_output)
        
        # 残差连接
        x = residual + attn_output
        
        # 前馈网络（FFN）部分
        residual2 = x
        x = self.layer_norm2(x)
        ffn_output = self.ffn(x)
        x = residual2 + ffn_output
        
        # 恢复形状：(batch, 49, 2048) → (batch, 2048, 49) → (batch, 2048, 7, 7)
        output = x.transpose(1, 2)  # (batch, 2048, 49)
        output = output.view(batch_size, feat_dim, h, w)  # (batch, 2048, 7, 7)
        
        return output


class EnhancedCNNEncoder(nn.Module):
    """
    增强型CNN特征编码器（增强版：更大的容量、更深的网络）
    输入：局部特征 (batch, 2048, 7, 7)
    输出：768维融合自注意力的视觉特征向量 (batch, 768)
    """
    
    def __init__(self, dropout: float = 0.15, output_dim: int = 768):
        super(EnhancedCNNEncoder, self).__init__()
        
        # 局部特征自注意力层（改进版：增强正则化）
        # 使用8头自注意力（2048/8=256，可以整除）
        self.self_attention = LocalSelfAttention(feature_dim=2048, num_heads=8, dropout=dropout)
        
        # 第一层卷积：2048 → 1024（带残差连接）
        # Conv2d(2048, 1024, kernel_size=3, padding=1) → BatchNorm2d(1024) → ReLU → MaxPool2d(kernel_size=2, stride=2)
        # 输入: (batch, 2048, 7, 7)
        # 输出: (batch, 1024, 3, 3)
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(dropout * 0.5)
        
        # 第二层卷积：1024 → 768（带残差连接）
        # Conv2d(1024, 768, kernel_size=3, padding=1) → BatchNorm2d(768) → ReLU → AvgPool2d(kernel_size=3, stride=1)
        # 输入: (batch, 1024, 3, 3)
        # 输出: (batch, 768, 1, 1)
        self.conv2 = nn.Conv2d(1024, 768, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(768)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=1)
        self.dropout2 = nn.Dropout2d(dropout * 0.5)
        
        # 特征扁平化和全连接层（带残差连接，输出更大维度）
        # Flatten() → Linear(768, output_dim) → ReLU
        # 输入: (batch, 768, 1, 1) → Flatten → (batch, 768)
        # 输出: (batch, output_dim)
        self.flatten = nn.Flatten()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(768, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout_fc = nn.Dropout(dropout)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重：卷积层He_normal，全连接层Xavier_uniform"""
        # 卷积层：He_normal初始化
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        # BatchNorm：权重1，偏置0
        for m in [self.bn1, self.bn2]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
        # 全连接层：Xavier_uniform初始化（适配新的输出维度）
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, local_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（改进版：残差连接）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
        
        Returns:
            visual_feat: 视觉特征向量 (batch, output_dim)
            attn_enhanced_feat: 自注意力增强后的局部特征 (batch, 2048, 7, 7)，用于后续注意力机制
        """
        # 局部特征自注意力（改进版）
        # (batch, 2048, 7, 7) → (batch, 2048, 7, 7)
        attn_enhanced_feat = self.self_attention(local_feat)
        
        # 第一层卷积（带残差连接）
        # (batch, 2048, 7, 7) → (batch, 1024, 7, 7) → (batch, 1024, 3, 3)
        x = self.conv1(attn_enhanced_feat)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第二层卷积（带残差连接）
        # (batch, 1024, 3, 3) → (batch, 768, 3, 3) → (batch, 768, 1, 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 扁平化和全连接（带残差连接和LayerNorm）
        # (batch, 768, 1, 1) → (batch, 768) → (batch, output_dim)
        x = self.flatten(x)
        
        # 第一层全连接（768 -> output_dim）
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        
        # 第二层全连接（残差连接，需要维度匹配）
        residual_fc = x
        x = self.fc2(x)
        x = residual_fc + x  # 残差连接
        x = self.layer_norm(x)
        x = F.relu(x)
        
        return x, attn_enhanced_feat


class AttentionLayer(nn.Module):
    """
    标准注意力层（增强版：更大的隐藏维度、多头注意力、LayerNorm）
    计算LSTM隐藏状态与局部特征的注意力得分，生成注意力上下文向量
    """
    
    def __init__(self, hidden_dim: int = 768, feature_dim: int = 2048, num_heads: int = 12, dropout: float = 0.15):
        """
        初始化注意力层（改进版：增强正则化）
        
        Args:
            hidden_dim: LSTM隐藏状态维度，默认768
            feature_dim: 局部特征维度，默认2048
            num_heads: 注意力头数，默认12
            dropout: Dropout比例，默认0.15（从0.1增加到0.15，增强正则化）
        """
        super(AttentionLayer, self).__init__()
        
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 将局部特征映射到hidden_dim维度（添加Dropout）
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Dropout(dropout * 0.5)
        )
        
        # 多头注意力：Q、K、V投影（添加Dropout）
        self.q_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout * 0.5)
        )
        self.k_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout * 0.5)
        )
        self.v_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout * 0.5)
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # LayerNorm和Dropout（增强正则化）
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        # 初始化投影层（需要从Sequential中获取Linear层）
        for module in self.feature_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        for module in self.q_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        for module in self.k_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        for module in self.v_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)
    
    def forward(self, hidden_state: torch.Tensor, local_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：计算注意力上下文向量（改进版：多头注意力）
        
        Args:
            hidden_state: LSTM当前时刻隐藏状态 (batch, hidden_dim) 或 (batch, 1, hidden_dim)
            local_feat: 自注意力增强后的局部特征 (batch, 2048, 7, 7)
        
        Returns:
            context: 注意力上下文向量 (batch, hidden_dim)
            attn_weights: 注意力权重 (batch, 49)，用于可视化（平均所有头）
        """
        batch_size = hidden_state.size(0)
        
        # 处理hidden_state维度
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(1)  # (batch, 1, hidden_dim) → (batch, hidden_dim)
        
        # 展平局部特征：(batch, 2048, 7, 7) → (batch, 2048, 49) → (batch, 49, 2048)
        h, w = local_feat.size(2), local_feat.size(3)
        local_feat_flat = local_feat.view(batch_size, self.feature_dim, h * w)  # (batch, 2048, 49)
        local_feat_flat = local_feat_flat.transpose(1, 2)  # (batch, 49, 2048)
        
        # 将局部特征映射到hidden_dim维度（Sequential包含Dropout）
        # (batch, 49, 2048) → (batch, 49, hidden_dim)
        local_feat_proj = self.feature_proj(local_feat_flat)  # (batch, 49, hidden_dim)
        
        # 多头注意力：Q、K、V投影（Sequential包含Dropout）
        # Q来自hidden_state，K和V来自local_feat_proj
        Q = self.q_proj(hidden_state)  # (batch, hidden_dim)
        K = self.k_proj(local_feat_proj)  # (batch, 49, hidden_dim)
        V = self.v_proj(local_feat_proj)  # (batch, 49, hidden_dim)
        
        # 分多头
        # Q: (batch, 512) → (batch, 1, num_heads, head_dim) → (batch, num_heads, 1, head_dim)
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)
        # K, V: (batch, 49, 512) → (batch, 49, num_heads, head_dim) → (batch, num_heads, 49, head_dim)
        K = K.view(batch_size, h * w, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 49, head_dim)
        V = V.view(batch_size, h * w, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 49, head_dim)
        
        # 计算注意力得分（Scaled Dot-Product Attention）
        # (batch, num_heads, 1, head_dim) × (batch, num_heads, head_dim, 49) → (batch, num_heads, 1, 49)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, num_heads, 1, 49)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, 1, 49)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        # (batch, num_heads, 1, 49) × (batch, num_heads, 49, head_dim) → (batch, num_heads, 1, head_dim)
        attn_output = torch.matmul(attn_weights, V)  # (batch, num_heads, 1, head_dim)
        
        # 合并多头
        # (batch, num_heads, 1, head_dim) → (batch, 1, num_heads, head_dim) → (batch, 1, 512)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, 1, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, 1, self.hidden_dim)  # (batch, 1, 512)
        
        # 输出投影
        context = self.output_proj(attn_output)  # (batch, 1, 512)
        context = context.squeeze(1)  # (batch, 512)
        
        # LayerNorm
        context = self.layer_norm(context)
        
        # 计算平均注意力权重（用于可视化）
        attn_weights_avg = attn_weights.mean(dim=1).squeeze(1)  # (batch, 49)
        
        return context, attn_weights_avg


class AttentionLSTMDecoder(nn.Module):
    """
    注意力增强LSTM解码器（增强版：更大的容量、更深的结构、LayerNorm、更好的初始化）
    基础框架：使用3层LSTM解码器，新增"标准注意力机制"
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 512, hidden_dim: int = 768, num_layers: int = 3, dropout: float = 0.15):
        """
        初始化注意力增强LSTM解码器（改进版：增强正则化）
        
        Args:
            vocab_size: 词典大小
            embed_dim: 词嵌入维度，默认512
            hidden_dim: LSTM隐藏状态维度，默认768
            dropout: Dropout比例，默认0.15（从0.1增加到0.15，增强正则化）
        """
        super(AttentionLSTMDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 词嵌入层：更大的嵌入维度（增强正则化）
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # 注意力层（增强版：更大的隐藏维度，更多注意力头，增强正则化）
        self.attention = AttentionLayer(hidden_dim=hidden_dim, feature_dim=2048, num_heads=16, dropout=dropout)
        
        # 输入融合层（词嵌入 + 注意力上下文）
        self.input_fusion = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # 额外的融合层
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3层LSTM（增加层数以提升容量）
        # input_size=hidden_dim（融合后的输入）
        # hidden_size=hidden_dim, num_layers=3, batch_first=True, bidirectional=False
        self.lstm = nn.LSTM(
            input_size=hidden_dim,  # 融合后的输入
            hidden_size=hidden_dim,
            num_layers=num_layers,  # 3层
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0  # 多层LSTM时使用dropout
        )
        
        # 隐藏/细胞状态初始化：使用更深的网络
        self.hidden_init = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        self.cell_init = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # 输出层：更深的投影层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        # 词嵌入层：Xavier_uniform
        nn.init.xavier_uniform_(self.embedding.weight)
        # padding_idx=0的权重设为0
        self.embedding.weight.data[0].fill_(0)
        
        # 隐藏/细胞状态初始化层：初始化所有Linear层
        for module in self.hidden_init:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        for module in self.cell_init:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        # 输出层：初始化所有Linear层
        for module in self.output_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        # 输入融合层：初始化所有Linear层
        for module in self.input_fusion:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def init_hidden(self, cnn_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化LSTM隐藏状态和细胞状态（改进版：使用LayerNorm）
        
        Args:
            cnn_feat: CNN编码器输出的特征 (batch, 512)
        
        Returns:
            h0: 初始隐藏状态 (num_layers, batch, hidden_dim)
            c0: 初始细胞状态 (num_layers, batch, hidden_dim)
        """
        # (batch, hidden_dim) → (batch, hidden_dim)
        h0 = self.hidden_init(cnn_feat)
        c0 = self.cell_init(cnn_feat)
        
        # 扩展为num_layers层LSTM的初始状态
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, hidden_dim)
        c0 = c0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, hidden_dim)
        
        return h0, c0
    
    def forward(self, word_ids: torch.Tensor, cnn_feat: torch.Tensor,
                attn_enhanced_feat: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                return_attn: bool = False) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播（训练模式）
        
        Args:
            word_ids: 词ID序列 (batch, seq_len)
            cnn_feat: CNN编码器输出的特征 (batch, 512)，用于初始化
            attn_enhanced_feat: 自注意力增强后的局部特征 (batch, 2048, 7, 7)，用于注意力机制
            hidden: 初始隐藏状态和细胞状态，如果为None则自动初始化
            return_attn: 是否返回注意力权重，默认False
        
        Returns:
            outputs: LSTM输出 (batch, seq_len, vocab_size)
            hidden: 最终隐藏状态和细胞状态 ((2, batch, 512), (2, batch, 512))
            attn_weights: 注意力权重 (batch, seq_len, 49)，如果return_attn=True
        """
        batch_size, seq_len = word_ids.size()
        device = word_ids.device
        
        # 初始化隐藏状态和细胞状态
        if hidden is None:
            h0, c0 = self.init_hidden(cnn_feat)
        else:
            h0, c0 = hidden
        
        # 词嵌入
        # (batch, seq_len) → (batch, seq_len, 256)
        word_embeds = self.embedding(word_ids)
        word_embeds = self.embedding_dropout(word_embeds)
        
        # 逐时间步处理（需要计算每个时间步的注意力）
        outputs_list = []
        attn_weights_list = []
        
        # 获取当前时刻的隐藏状态（使用最后一层的隐藏状态）
        h_current = h0[-1]  # (batch, hidden_dim) - 最后一层的隐藏状态
        
        for t in range(seq_len):
            # 计算注意力上下文向量
            context, attn_w = self.attention(h_current, attn_enhanced_feat)  # context: (batch, hidden_dim), attn_w: (batch, 49)
            
            # 拼接词嵌入和注意力上下文向量，然后融合
            # word_embeds[:, t, :]: (batch, embed_dim)
            # context: (batch, hidden_dim)
            # concat: (batch, embed_dim + hidden_dim)
            concat_input = torch.cat([word_embeds[:, t, :], context], dim=1)  # (batch, embed_dim + hidden_dim)
            lstm_input = self.input_fusion(concat_input)  # (batch, hidden_dim)
            lstm_input = lstm_input.unsqueeze(1)  # (batch, 1, hidden_dim)
            
            # LSTM前向传播
            lstm_output, (h_n, c_n) = self.lstm(lstm_input, (h0, c0))
            
            # 更新隐藏状态（用于下一时间步的注意力计算）
            h0, c0 = h_n, c_n
            h_current = h0[-1]  # (batch, 512)
            
            # 输出投影
            output = self.output_proj(lstm_output)  # (batch, 1, vocab_size)
            outputs_list.append(output)
            
            if return_attn:
                attn_weights_list.append(attn_w.unsqueeze(1))  # (batch, 1, 49)
        
        # 拼接所有时间步的输出
        outputs = torch.cat(outputs_list, dim=1)  # (batch, seq_len, vocab_size)
        
        attn_weights = None
        if return_attn:
            attn_weights = torch.cat(attn_weights_list, dim=1)  # (batch, seq_len, 49)
        
        return outputs, (h0, c0), attn_weights


class FashionCaptionModelAttention(nn.Module):
    """
    服饰描述生成模型：增强型CNN编码器（局部自注意力）+ 注意力增强LSTM解码器
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 768, hidden_dim: int = 1024, 
                 num_layers: int = 4, dropout: float = 0.2, cnn_output_dim: int = 1024):
        """
        初始化模型（超大容量版 - 目标：至少一个指标50%+）
        
        Args:
            vocab_size: 词典大小
            embed_dim: 词嵌入维度，默认768（优化：512→768，更强的词表示）
            hidden_dim: LSTM隐藏状态维度，默认1024（优化：768→1024，最大化模型容量）
            num_layers: LSTM层数，默认4（优化：3→4，更深的网络）
            dropout: Dropout比例，默认0.2（优化：0.15→0.2，防止过拟合）
            cnn_output_dim: CNN编码器输出维度，默认1024（优化：768→1024，匹配hidden_dim）
        """
        super(FashionCaptionModelAttention, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 增强型CNN编码器（改进版：更大的输出维度，增强正则化）
        self.cnn_encoder = EnhancedCNNEncoder(dropout=dropout, output_dim=cnn_output_dim)
        
        # 确保CNN输出维度与隐藏维度匹配
        if cnn_output_dim != hidden_dim:
            # 添加一个投影层来匹配维度
            self.cnn_proj = nn.Linear(cnn_output_dim, hidden_dim)
        else:
            self.cnn_proj = nn.Identity()
        
        # 注意力增强LSTM解码器（增强版：更大的容量、更深的网络）
        self.lstm_decoder = AttentionLSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, local_feat: torch.Tensor, caption: torch.Tensor,
                teacher_forcing_ratio: float = 0.8, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播（训练模式：混合Teacher-Forcing）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
            caption: 文本序列 (batch, seq_len)
            teacher_forcing_ratio: Teacher-Forcing比例，默认0.8
            return_attn: 是否返回注意力权重，默认False
        
        Returns:
            outputs: 模型输出 (batch, seq_len-1, vocab_size)
            attn_weights: 注意力权重 (batch, seq_len-1, 49)，如果return_attn=True
        """
        batch_size, seq_len = caption.size()
        device = local_feat.device
        
        # CNN编码：获取视觉特征和自注意力增强后的局部特征
        cnn_feat, attn_enhanced_feat = self.cnn_encoder(local_feat)  # cnn_feat: (batch, cnn_output_dim), attn_enhanced_feat: (batch, 2048, 7, 7)
        
        # 投影CNN特征到隐藏维度
        cnn_feat = self.cnn_proj(cnn_feat)  # (batch, hidden_dim)
        
        # 初始化LSTM隐藏状态
        h0, c0 = self.lstm_decoder.init_hidden(cnn_feat)
        
        # 输入序列（去掉最后一个词）
        input_ids = caption[:, :-1]  # (batch, seq_len-1)
        
        outputs_list = []
        attn_weights_list = []
        
        # 获取当前时刻的隐藏状态
        h_current = h0[-1]  # (batch, 512)
        
        # 第一个时间步使用<START>标记
        decoder_input = caption[:, 0:1]  # (batch, 1)
        
        for t in range(seq_len - 1):
            # 计算注意力上下文向量
            context, attn_w = self.lstm_decoder.attention(h_current, attn_enhanced_feat)
            
            # 词嵌入
            word_embeds = self.lstm_decoder.embedding(decoder_input)  # (batch, 1, embed_dim)
            word_embeds = self.lstm_decoder.embedding_dropout(word_embeds)
            
            # 拼接词嵌入和注意力上下文向量，然后融合
            concat_input = torch.cat([word_embeds.squeeze(1), context], dim=1)  # (batch, embed_dim + hidden_dim)
            lstm_input = self.lstm_decoder.input_fusion(concat_input)  # (batch, hidden_dim)
            lstm_input = lstm_input.unsqueeze(1)  # (batch, 1, hidden_dim)
            
            # LSTM前向传播
            lstm_output, (h0, c0) = self.lstm_decoder.lstm(lstm_input, (h0, c0))
            
            # 输出投影
            output = self.lstm_decoder.output_proj(lstm_output)  # (batch, 1, vocab_size)
            outputs_list.append(output)
            
            if return_attn:
                attn_weights_list.append(attn_w.unsqueeze(1))  # (batch, 1, 49)
            
            # 更新隐藏状态（用于下一时间步的注意力计算）
            h_current = h0[-1]  # (batch, hidden_dim)
            
            # 混合Teacher-Forcing：80%使用真实词，20%使用模型预测词
            if self.training and random.random() < teacher_forcing_ratio:
                if t + 1 < seq_len - 1:
                    decoder_input = input_ids[:, t+1:t+2]
            else:
                decoder_input = torch.argmax(output, dim=-1)  # (batch, 1)
        
        # 拼接所有时间步的输出
        outputs = torch.cat(outputs_list, dim=1)  # (batch, seq_len-1, vocab_size)
        
        attn_weights = None
        if return_attn:
            attn_weights = torch.cat(attn_weights_list, dim=1)  # (batch, seq_len-1, 49)
        
        return outputs, attn_weights
    
    def generate(self, local_feat: torch.Tensor, max_len: int = 25,
                 start_idx: int = 2, end_idx: int = 3, pad_idx: int = 0,
                 temperature: float = 0.7, return_attn: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成描述（推理模式：贪心解码+温度系数）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
            max_len: 最大生成长度，默认25
            start_idx: <START>标记索引，默认2
            end_idx: <END>标记索引，默认3
            pad_idx: <PAD>标记索引，默认0
            temperature: 温度系数，默认0.7
            return_attn: 是否返回注意力权重，默认True
        
        Returns:
            generated_sequence: 生成的词ID序列 (batch, generated_len)
            attn_weights: 注意力权重 (batch, generated_len, 49)
        """
        batch_size = local_feat.size(0)
        device = local_feat.device
        
        # CNN编码
        cnn_feat, attn_enhanced_feat = self.cnn_encoder(local_feat)
        
        # 投影CNN特征到隐藏维度
        cnn_feat = self.cnn_proj(cnn_feat)
        
        # 初始化LSTM隐藏状态
        h0, c0 = self.lstm_decoder.init_hidden(cnn_feat)
        
        # 初始化输入：<START>标记
        decoder_input = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=device)
        generated_ids = []
        attn_weights_list = []
        
        # 获取当前时刻的隐藏状态
        h_current = h0[-1]  # (batch, hidden_dim)
        
        consecutive_pads = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 逐词生成
        for _ in range(max_len):
            # 计算注意力上下文向量
            context, attn_w = self.lstm_decoder.attention(h_current, attn_enhanced_feat)
            
            # 词嵌入
            word_embeds = self.lstm_decoder.embedding(decoder_input)  # (batch, 1, embed_dim)
            word_embeds = self.lstm_decoder.embedding_dropout(word_embeds)
            
            # 拼接词嵌入和注意力上下文向量，然后融合
            concat_input = torch.cat([word_embeds.squeeze(1), context], dim=1)  # (batch, embed_dim + hidden_dim)
            lstm_input = self.lstm_decoder.input_fusion(concat_input)  # (batch, hidden_dim)
            lstm_input = lstm_input.unsqueeze(1)  # (batch, 1, hidden_dim)
            
            # LSTM前向传播
            lstm_output, (h0, c0) = self.lstm_decoder.lstm(lstm_input, (h0, c0))
            
            # 输出投影
            output = self.lstm_decoder.output_proj(lstm_output)  # (batch, 1, vocab_size)
            
            # 温度采样
            if temperature != 1.0:
                output = output / temperature
            
            # 贪心解码：选择概率最大的词
            next_word_ids = torch.argmax(output, dim=-1)  # (batch, 1)
            generated_ids.append(next_word_ids)
            
            if return_attn:
                attn_weights_list.append(attn_w.unsqueeze(1))  # (batch, 1, 49)
            
            # 更新隐藏状态
            h_current = h0[-1]  # (batch, 512)
            
            # 检查终止条件
            end_mask = (next_word_ids.squeeze(1) == end_idx)
            pad_mask = (next_word_ids.squeeze(1) == pad_idx)
            consecutive_pads = consecutive_pads + pad_mask.long()
            consecutive_pads = consecutive_pads * pad_mask.long()
            
            if torch.all(end_mask | (consecutive_pads >= 2)):
                break
            
            # 更新输入
            decoder_input = next_word_ids
        
        # 拼接生成的序列
        generated_sequence = torch.cat(generated_ids, dim=1)  # (batch, generated_len)
        
        attn_weights = None
        if return_attn:
            attn_weights = torch.cat(attn_weights_list, dim=1)  # (batch, generated_len, 49)
        
        return generated_sequence, attn_weights
    
    def generate_beam_search(self, local_feat: torch.Tensor, max_len: int = 25,
                            start_idx: int = 2, end_idx: int = 3, pad_idx: int = 0,
                            beam_size: int = 5, length_penalty: float = 0.6,
                            return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        使用Beam Search生成描述（更好的生成质量）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
            max_len: 最大生成长度，默认25
            start_idx: <START>标记索引，默认2
            end_idx: <END>标记索引，默认3
            pad_idx: <PAD>标记索引，默认0
            beam_size: Beam大小，默认5
            length_penalty: 长度惩罚系数，默认0.6（值越小，越鼓励长句子）
            return_attn: 是否返回注意力权重，默认False（Beam Search时不支持）
        
        Returns:
            generated_sequence: 生成的词ID序列 (batch, generated_len)
            attn_weights: None（Beam Search暂不支持返回注意力权重）
        """
        batch_size = local_feat.size(0)
        device = local_feat.device
        
        # CNN编码（只需编码一次）
        cnn_feat, attn_enhanced_feat = self.cnn_encoder(local_feat)
        cnn_feat = self.cnn_proj(cnn_feat)
        
        # 为每个样本进行Beam Search
        all_sequences = []
        
        for batch_idx in range(batch_size):
            # 单个样本的特征
            single_cnn_feat = cnn_feat[batch_idx:batch_idx+1]  # (1, hidden_dim)
            single_local_feat = attn_enhanced_feat[batch_idx:batch_idx+1]  # (1, 2048, 7, 7)
            
            # 初始化Beam：每个beam包含(序列, log概率, 隐藏状态, 细胞状态, 连续pad计数)
            h0, c0 = self.lstm_decoder.init_hidden(single_cnn_feat)
            h_current = h0[-1]  # (1, hidden_dim)
            
            # 初始beam：只有<START>标记
            beams = [{
                'sequence': [start_idx],
                'log_prob': 0.0,
                'hidden': (h0, c0),
                'h_current': h_current,
                'consecutive_pads': 0,
                'finished': False
            }]
            
            # Beam Search循环
            for step in range(max_len):
                new_beams = []
                
                for beam in beams:
                    if beam['finished']:
                        new_beams.append(beam)
                        continue
                    
                    # 当前输入词
                    decoder_input = torch.tensor([[beam['sequence'][-1]]], dtype=torch.long, device=device)  # (1, 1)
                    
                    # 计算注意力上下文
                    context, _ = self.lstm_decoder.attention(beam['h_current'], single_local_feat)
                    
                    # 词嵌入和融合
                    word_embeds = self.lstm_decoder.embedding(decoder_input)  # (1, 1, embed_dim)
                    word_embeds = self.lstm_decoder.embedding_dropout(word_embeds)
                    concat_input = torch.cat([word_embeds.squeeze(1), context], dim=1)  # (1, embed_dim + hidden_dim)
                    lstm_input = self.lstm_decoder.input_fusion(concat_input)  # (1, hidden_dim)
                    lstm_input = lstm_input.unsqueeze(1)  # (1, 1, hidden_dim)
                    
                    # LSTM前向传播
                    lstm_output, (h_n, c_n) = self.lstm_decoder.lstm(lstm_input, beam['hidden'])
                    
                    # 输出投影
                    output = self.lstm_decoder.output_proj(lstm_output)  # (1, 1, vocab_size)
                    log_probs = F.log_softmax(output.squeeze(1), dim=-1)  # (1, vocab_size)
                    
                    # 获取top-k个候选词
                    top_log_probs, top_indices = torch.topk(log_probs, beam_size, dim=-1)  # (1, beam_size)
                    top_log_probs = top_log_probs[0].cpu().numpy()  # (beam_size,)
                    top_indices = top_indices[0].cpu().numpy()  # (beam_size,)
                    
                    # 为每个候选词创建新的beam
                    for i in range(beam_size):
                        word_id = int(top_indices[i])
                        word_log_prob = float(top_log_probs[i])
                        
                        new_sequence = beam['sequence'] + [word_id]
                        new_log_prob = beam['log_prob'] + word_log_prob
                        
                        # 更新连续pad计数
                        new_consecutive_pads = beam['consecutive_pads'] + 1 if word_id == pad_idx else 0
                        
                        # 检查是否结束
                        finished = (word_id == end_idx) or (new_consecutive_pads >= 2)
                        
                        new_beams.append({
                            'sequence': new_sequence,
                            'log_prob': new_log_prob,
                            'hidden': (h_n, c_n),
                            'h_current': h_n[-1],  # (1, hidden_dim)
                            'consecutive_pads': new_consecutive_pads,
                            'finished': finished
                        })
                
                # 选择top-k beams（考虑长度惩罚）
                # 得分 = log_prob / (length^length_penalty)
                beam_scores = []
                for beam in new_beams:
                    if beam['finished']:
                        # 已完成序列：使用最终长度
                        length = len(beam['sequence'])
                    else:
                        # 未完成序列：使用当前长度+1（考虑下一步）
                        length = len(beam['sequence']) + 1
                    
                    # 长度惩罚
                    score = beam['log_prob'] / (length ** length_penalty)
                    beam_scores.append(score)
                
                # 选择top beam_size个beams
                top_indices = np.argsort(beam_scores)[-beam_size:][::-1]
                beams = [new_beams[i] for i in top_indices]
                
                # 如果所有beam都完成了，提前结束
                if all(beam['finished'] for beam in beams):
                    break
            
            # 选择得分最高的beam
            if not all(beam['finished'] for beam in beams):
                # 如果有未完成的beam，应用长度惩罚选择最佳
                final_scores = []
                for beam in beams:
                    length = len(beam['sequence'])
                    score = beam['log_prob'] / (length ** length_penalty)
                    final_scores.append(score)
                best_idx = np.argmax(final_scores)
            else:
                # 所有beam都完成，选择得分最高的
                best_idx = 0
            
            best_beam = beams[best_idx]
            best_sequence = best_beam['sequence'][1:]  # 去掉<START>标记
            
            # 找到<END>标记的位置（如果存在）
            if end_idx in best_sequence:
                end_pos = best_sequence.index(end_idx)
                best_sequence = best_sequence[:end_pos]
            
            all_sequences.append(torch.tensor(best_sequence, dtype=torch.long, device=device))
        
        # 填充到相同长度
        max_seq_len = max(len(seq) for seq in all_sequences)
        padded_sequences = []
        for seq in all_sequences:
            if len(seq) < max_seq_len:
                padding = torch.full((max_seq_len - len(seq),), pad_idx, dtype=torch.long, device=device)
                padded_seq = torch.cat([seq, padding])
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq.unsqueeze(0))
        
        generated_sequence = torch.cat(padded_sequences, dim=0)  # (batch, max_seq_len)
        
        return generated_sequence, None  # Beam Search暂不支持返回注意力权重
    
    def postprocess_caption(self, sequence: torch.Tensor, idx2word: dict,
                           start_idx: int = 2, end_idx: int = 3,
                           unk_idx: int = 1, pad_idx: int = 0) -> list:
        """
        后处理生成的文本：过滤<UNK>/<PAD>/<START>，仅保留有效英文单词
        
        Args:
            sequence: 生成的词ID序列 (generated_len,)
            idx2word: 索引到单词的映射字典
            start_idx: <START>标记索引
            end_idx: <END>标记索引
            unk_idx: <UNK>标记索引
            pad_idx: <PAD>标记索引
        
        Returns:
            处理后的单词列表
        """
        words = []
        for idx in sequence.cpu().tolist():
            if idx == end_idx:
                break
            if idx not in [start_idx, end_idx, unk_idx, pad_idx]:
                word = idx2word.get(idx, f"<{idx}>")
                words.append(word)
        return words


def visualize_attention(attn_weights: torch.Tensor, save_path: Optional[str] = None):
    """
    可视化注意力权重热力图
    
    Args:
        attn_weights: 注意力权重 (seq_len, 49) 或 (49,)
        save_path: 保存路径，如果为None则不保存
    """
    if attn_weights.dim() == 1:
        attn_weights = attn_weights.unsqueeze(0)  # (1, 49)
    
    seq_len, num_pixels = attn_weights.shape
    h, w = 7, 7  # 7×7局部特征
    
    # 将注意力权重重塑为7×7
    attn_map = attn_weights.view(seq_len, h, w).cpu().numpy()
    
    # 创建子图
    fig, axes = plt.subplots(1, seq_len, figsize=(seq_len * 3, 3))
    if seq_len == 1:
        axes = [axes]
    
    for i in range(seq_len):
        ax = axes[i]
        im = ax.imshow(attn_map[i], cmap='hot', interpolation='nearest')
        ax.set_title(f'Step {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] 注意力热力图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_vocab(vocab_file: str) -> dict:
    """
    加载词典文件
    
    Args:
        vocab_file: 词典文件路径
    
    Returns:
        包含vocab_size、word2idx、idx2word的字典
    """
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    return {
        'vocab_size': vocab_data['vocab_size'],
        'word2idx': vocab_data['word2idx'],
        'idx2word': {int(k): v for k, v in vocab_data['idx2word'].items()}
    }


if __name__ == "__main__":
    """
    测试代码：模型初始化、前向计算、生成演示
    """
    print("="*60)
    print("Model2测试：增强型CNN编码器（局部自注意力）+ 注意力增强LSTM解码器")
    print("="*60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载词典
    vocab_file = "dataset/vocab.json"
    try:
        vocab_info = load_vocab(vocab_file)
        vocab_size = vocab_info['vocab_size']
        idx2word = vocab_info['idx2word']
        print(f"[OK] 词典大小: {vocab_size}")
    except FileNotFoundError:
        print(f"[WARN] 找不到词典文件 {vocab_file}，使用默认vocab_size=5000")
        vocab_size = 5000
        idx2word = {i: f"word_{i}" for i in range(5000)}
    
    # 测试1: 模型初始化
    print("\n" + "="*60)
    print("测试1: 模型初始化")
    print("="*60)
    model = FashionCaptionModelAttention(vocab_size=vocab_size)
    model = model.to(device)
    
    total_params = count_parameters(model)
    print(f"[OK] 模型参数量: {total_params:,}")
    
    # 测试2: 单batch前向计算测试（训练模式）
    print("\n" + "="*60)
    print("测试2: 前向计算测试（训练模式，混合Teacher-Forcing）")
    print("="*60)
    batch_size = 4
    seq_len = 20
    local_feat = torch.randn(batch_size, 2048, 7, 7).to(device)
    caption = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    caption[:, 0] = 2  # <START>
    caption[:, -1] = 3  # <END>
    
    print(f"输入局部特征形状: {local_feat.shape}")
    print(f"输入文本序列形状: {caption.shape}")
    
    model.train()
    outputs, attn_weights = model(local_feat, caption, teacher_forcing_ratio=0.8, return_attn=True)
    print(f"输出形状: {outputs.shape}")
    print(f"注意力权重形状: {attn_weights.shape if attn_weights is not None else None}")
    assert outputs.shape == (batch_size, seq_len - 1, vocab_size), \
        f"输出维度错误: {outputs.shape}"
    print("[OK] 前向计算测试通过")
    
    # 测试3: 5个样本生成演示（包含注意力热力图）
    print("\n" + "="*60)
    print("测试3: 生成演示（5个样本，包含注意力热力图）")
    print("="*60)
    model.eval()
    test_batch_size = 5
    test_local_feat = torch.randn(test_batch_size, 2048, 7, 7).to(device)
    
    with torch.no_grad():
        generated_sequences, attn_weights = model.generate(
            test_local_feat, max_len=25, temperature=0.7, return_attn=True
        )
    
    print(f"生成序列形状: {generated_sequences.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print("\n生成的描述示例（包含注意力可视化）:")
    
    for i in range(test_batch_size):
        seq = generated_sequences[i].cpu()
        gen_words = model.postprocess_caption(seq, idx2word)
        print(f"\n样本{i+1}:")
        print(f"  生成描述: {' '.join(gen_words) if gen_words else '(空)'}")
        
        # 可视化注意力权重（取平均）
        if attn_weights is not None:
            avg_attn = attn_weights[i].mean(dim=0)  # (49,)
            print(f"  注意力权重形状: {avg_attn.shape}")
            # 注意：实际使用时需要保存图像，这里仅打印形状
            print(f"  [注意] 注意力权重已计算，可用于可视化热力图")
    
    print("\n" + "="*60)
    print("[OK] 所有测试通过！")
    print("="*60)

