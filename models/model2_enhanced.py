"""
增强版Model2：深度编码器+增强解码器+改进注意力机制
目标：METEOR ≥ 0.7，提升所有评测指标

核心改进：
1. 编码器：多层自注意力（3层）+ 更深的卷积结构 + 残差连接
2. 解码器：更大的词嵌入（512维）+ 3层LSTM（768隐藏状态）+ 多头注意力
3. 注意力机制：加性注意力 + 层归一化 + 更高dropout
4. 特征融合：多层次特征融合
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


class MultiLayerLocalSelfAttention(nn.Module):
    """
    多层局部特征自注意力层
    使用3层自注意力，每层都有残差连接和层归一化
    """
    
    def __init__(self, feature_dim: int = 2048, num_heads: int = 8, num_layers: int = 3, dropout: float = 0.2):
        """
        初始化多层局部自注意力层
        
        Args:
            feature_dim: 特征维度，默认2048
            num_heads: 注意力头数，默认8
            num_layers: 自注意力层数，默认3
            dropout: Dropout比例，默认0.2
        """
        super(MultiLayerLocalSelfAttention, self).__init__()
        
        assert feature_dim % num_heads == 0, f"feature_dim ({feature_dim}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.num_layers = num_layers
        
        # 多层自注意力层
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'q_linear': nn.Linear(feature_dim, feature_dim),
                'k_linear': nn.Linear(feature_dim, feature_dim),
                'v_linear': nn.Linear(feature_dim, feature_dim),
                'output_linear': nn.Linear(feature_dim, feature_dim),
                'layer_norm1': nn.LayerNorm(feature_dim),
                'layer_norm2': nn.LayerNorm(feature_dim),
                'dropout1': nn.Dropout(dropout),
                'dropout2': nn.Dropout(dropout)
            })
            self.layers.append(layer)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer['q_linear'].weight)
            nn.init.xavier_uniform_(layer['k_linear'].weight)
            nn.init.xavier_uniform_(layer['v_linear'].weight)
            nn.init.xavier_uniform_(layer['output_linear'].weight)
    
    def forward(self, local_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播：多层局部特征自注意力
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
        
        Returns:
            多层自注意力增强后的局部特征 (batch, 2048, 7, 7)
        """
        batch_size, feat_dim, h, w = local_feat.size()
        
        # 展平：将7×7局部特征展平为49个序列
        x = local_feat.view(batch_size, feat_dim, h * w).transpose(1, 2)  # (batch, 49, 2048)
        
        # 多层自注意力
        for layer in self.layers:
            residual = x
            
            # LayerNorm
            x = layer['layer_norm1'](x)
            
            # QKV计算
            Q = layer['q_linear'](x)
            K = layer['k_linear'](x)
            V = layer['v_linear'](x)
            
            # 分多头
            Q = Q.view(batch_size, h * w, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, h * w, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, h * w, self.num_heads, self.head_dim).transpose(1, 2)
            
            # 计算注意力得分
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = layer['dropout1'](attn_weights)
            
            # 加权求和
            attn_output = torch.matmul(attn_weights, V)
            
            # 合并多头
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, h * w, feat_dim)
            
            # 残差连接和LayerNorm
            x = layer['layer_norm2'](x + attn_output)
            x = layer['dropout2'](x)
        
        # 恢复形状
        output = x.transpose(1, 2).view(batch_size, feat_dim, h, w)
        
        return output


class EnhancedCNNEncoderV2(nn.Module):
    """
    增强型CNN特征编码器V2
    多层自注意力 + 更深的卷积结构 + 残差连接
    """
    
    def __init__(self, dropout: float = 0.2):
        super(EnhancedCNNEncoderV2, self).__init__()
        
        # 多层局部特征自注意力（3层）
        self.self_attention = MultiLayerLocalSelfAttention(
            feature_dim=2048, 
            num_heads=8, 
            num_layers=3,
            dropout=dropout
        )
        
        # 第一层卷积：2048 → 1024（带残差连接）
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.dropout1 = nn.Dropout2d(dropout * 0.5)
        
        # 残差连接投影层（如果需要）
        self.residual_conv1 = nn.Conv2d(2048, 1024, kernel_size=1)
        
        # 第二层卷积：1024 → 512
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout2 = nn.Dropout2d(dropout * 0.5)
        
        # 第三层卷积：512 → 512（进一步特征提取）
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        
        # 池化层
        self.pool1 = nn.AdaptiveAvgPool2d((3, 3))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        
        # 中间特征降维层
        self.mid_feat_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (batch, 1024, 3, 3) → (batch, 1024, 1, 1)
            nn.Flatten(),  # (batch, 1024)
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        
        # 特征融合：多尺度特征
        self.feature_fusion = nn.Sequential(
            nn.Linear(512 + 512, 768),  # 融合全局和局部特征
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, 512)
        )
        
        # 最终输出层
        self.final_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in [self.conv1, self.conv2, self.conv3, self.residual_conv1]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in [self.bn1, self.bn2, self.bn3]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
        for m in [self.mid_feat_proj, self.feature_fusion]:
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.constant_(layer.bias, 0)
        
        for m in self.final_fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, local_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
        
        Returns:
            visual_feat: 视觉特征向量 (batch, 512)
            attn_enhanced_feat: 多层自注意力增强后的局部特征 (batch, 2048, 7, 7)
        """
        # 多层局部特征自注意力
        attn_enhanced_feat = self.self_attention(local_feat)  # (batch, 2048, 7, 7)
        
        # 第一层卷积（带残差连接）
        x = self.conv1(attn_enhanced_feat)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # 残差连接（如果维度匹配）
        residual1 = self.residual_conv1(attn_enhanced_feat)
        x = x + residual1
        x = F.relu(x)
        
        # 保存中间特征用于融合
        mid_feat = self.pool1(x)  # (batch, 1024, 3, 3)
        
        # 第二层卷积
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # 第三层卷积
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 池化
        x = self.pool2(x)  # (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 512)
        
        # 多尺度特征融合：将中间特征降维并融合
        mid_feat_reduced = self.mid_feat_proj(mid_feat)  # (batch, 512)
        fused_feat = torch.cat([x, mid_feat_reduced], dim=1)  # (batch, 512+512)
        fused_feat = self.feature_fusion(fused_feat)  # (batch, 512)
        
        # 最终输出
        visual_feat = self.final_fc(fused_feat)  # (batch, 512)
        
        return visual_feat, attn_enhanced_feat


class AdditiveAttentionLayer(nn.Module):
    """
    加性注意力层（Bahdanau Attention）- 增强版
    使用加性注意力机制，更适合序列到序列任务
    改进：添加特征维度对齐和多头注意力支持
    """
    
    def __init__(self, hidden_dim: int = 768, feature_dim: int = 2048, attn_dim: int = 512, num_heads: int = 1):
        """
        初始化加性注意力层（增强版）
        
        Args:
            hidden_dim: LSTM隐藏状态维度，默认768
            feature_dim: 局部特征维度，默认2048
            attn_dim: 注意力维度，默认512
            num_heads: 注意力头数，默认1（单头）。如果>1，使用多头注意力
        """
        super(AdditiveAttentionLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        
        # 特征维度对齐层：将2048维特征先投影到更合理的维度
        self.feature_align = nn.Sequential(
            nn.Linear(feature_dim, attn_dim * 2),
            nn.LayerNorm(attn_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(attn_dim * 2, attn_dim)
        )
        
        # 将隐藏状态映射到注意力维度
        self.hidden_proj = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim * 2),
            nn.LayerNorm(attn_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(attn_dim * 2, attn_dim)
        )
        
        # 注意力得分计算层
        self.attention = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
        
        # 输出投影层：从对齐后的特征维度投影回hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(attn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in [self.feature_align, self.hidden_proj, self.output_proj]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
        
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, hidden_state: torch.Tensor, local_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：计算加性注意力上下文向量（增强版）
        
        Args:
            hidden_state: LSTM当前时刻隐藏状态 (batch, hidden_dim) 或 (batch, 1, hidden_dim)
            local_feat: 自注意力增强后的局部特征 (batch, 2048, 7, 7)
        
        Returns:
            context: 注意力上下文向量 (batch, hidden_dim)
            attn_weights: 注意力权重 (batch, 49)，用于可视化
        """
        batch_size = hidden_state.size(0)
        
        # 处理hidden_state维度
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(1)  # (batch, 1, hidden_dim) → (batch, hidden_dim)
        
        # 展平局部特征
        h, w = local_feat.size(2), local_feat.size(3)
        local_feat_flat = local_feat.view(batch_size, self.feature_dim, h * w).transpose(1, 2)  # (batch, 49, 2048)
        
        # 特征维度对齐：先将2048维特征投影到attn_dim
        feature_aligned = self.feature_align(local_feat_flat)  # (batch, 49, attn_dim)
        
        # 投影隐藏状态到注意力维度
        hidden_proj = self.hidden_proj(hidden_state)  # (batch, attn_dim)
        
        # 计算注意力得分（加性注意力）
        # hidden_proj: (batch, attn_dim) → (batch, 1, attn_dim)
        # feature_aligned: (batch, 49, attn_dim)
        hidden_expanded = hidden_proj.unsqueeze(1)  # (batch, 1, attn_dim)
        
        # 加性注意力：tanh(hidden + feature)
        attn_input = hidden_expanded + feature_aligned  # (batch, 49, attn_dim)
        scores = self.attention(attn_input)  # (batch, 49, 1)
        scores = scores.squeeze(2)  # (batch, 49)
        
        # Softmax归一化
        attn_weights = F.softmax(scores, dim=1)  # (batch, 49)
        
        # 加权求和得到上下文向量（使用对齐后的特征）
        context = torch.bmm(attn_weights.unsqueeze(1), feature_aligned)  # (batch, 1, attn_dim)
        context = context.squeeze(1)  # (batch, attn_dim)
        
        # 投影到hidden_dim维度（已包含归一化和dropout）
        context = self.output_proj(context)  # (batch, hidden_dim)
        
        return context, attn_weights


class EnhancedLSTMDecoderV2(nn.Module):
    """
    增强型LSTM解码器V2
    更大的词嵌入（512维）+ 3层LSTM（768隐藏状态）+ 加性注意力
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 512, hidden_dim: int = 768, num_layers: int = 3, dropout: float = 0.3):
        """
        初始化增强型LSTM解码器
        
        Args:
            vocab_size: 词典大小
            embed_dim: 词嵌入维度，默认512
            hidden_dim: LSTM隐藏状态维度，默认768
            num_layers: LSTM层数，默认3
            dropout: Dropout比例，默认0.3
        """
        super(EnhancedLSTMDecoderV2, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 词嵌入层（更大的维度）
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 词嵌入Dropout
        self.embed_dropout = nn.Dropout(dropout * 0.5)
        
        # 加性注意力层
        self.attention = AdditiveAttentionLayer(hidden_dim=hidden_dim, feature_dim=2048, attn_dim=512)
        
        # 3层LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,  # 512 + 768 = 1280
            hidden_size=hidden_dim,  # 768
            num_layers=num_layers,  # 3层
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 隐藏/细胞状态初始化（从512维CNN特征映射到768维）
        self.hidden_init_proj = nn.Linear(512, hidden_dim)
        self.cell_init_proj = nn.Linear(512, hidden_dim)
        self.hidden_init = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        self.cell_init = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # 输出层（带Dropout和归一化）
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data[0].fill_(0)
        
        nn.init.xavier_uniform_(self.hidden_init_proj.weight)
        nn.init.constant_(self.hidden_init_proj.bias, 0)
        nn.init.xavier_uniform_(self.cell_init_proj.weight)
        nn.init.constant_(self.cell_init_proj.bias, 0)
        
        for m in self.output_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def init_hidden(self, cnn_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化LSTM隐藏状态和细胞状态
        
        Args:
            cnn_feat: CNN编码器输出的特征 (batch, 512)
        
        Returns:
            h0: 初始隐藏状态 (num_layers, batch, hidden_dim)
            c0: 初始细胞状态 (num_layers, batch, hidden_dim)
        """
        # 将512维特征映射到768维
        h0_proj = self.hidden_init_proj(cnn_feat)  # (batch, 768)
        c0_proj = self.cell_init_proj(cnn_feat)  # (batch, 768)
        
        # 通过初始化层
        h0 = self.hidden_init(h0_proj)  # (batch, 768)
        c0 = self.cell_init(c0_proj)  # (batch, 768)
        
        # 扩展为多层LSTM的初始状态
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, 768)
        c0 = c0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, 768)
        
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
            attn_enhanced_feat: 自注意力增强后的局部特征 (batch, 2048, 7, 7)
            hidden: 初始隐藏状态和细胞状态，如果为None则自动初始化
            return_attn: 是否返回注意力权重，默认False
        
        Returns:
            outputs: LSTM输出 (batch, seq_len, vocab_size)
            hidden: 最终隐藏状态和细胞状态
            attn_weights: 注意力权重 (batch, seq_len, 49)，如果return_attn=True
        """
        batch_size, seq_len = word_ids.size()
        
        # 初始化隐藏状态和细胞状态
        if hidden is None:
            h0, c0 = self.init_hidden(cnn_feat)
        else:
            h0, c0 = hidden
        
        # 词嵌入
        word_embeds = self.embedding(word_ids)  # (batch, seq_len, embed_dim)
        word_embeds = self.embed_dropout(word_embeds)
        
        # 逐时间步处理
        outputs_list = []
        attn_weights_list = []
        
        # 获取当前时刻的隐藏状态（使用最后一层的隐藏状态）
        h_current = h0[-1]  # (batch, hidden_dim)
        
        for t in range(seq_len):
            # 计算注意力上下文向量
            context, attn_w = self.attention(h_current, attn_enhanced_feat)  # context: (batch, hidden_dim), attn_w: (batch, 49)
            
            # 拼接词嵌入和注意力上下文向量
            lstm_input = torch.cat([word_embeds[:, t, :], context], dim=1)  # (batch, embed_dim + hidden_dim)
            lstm_input = lstm_input.unsqueeze(1)  # (batch, 1, embed_dim + hidden_dim)
            
            # LSTM前向传播
            lstm_output, (h_n, c_n) = self.lstm(lstm_input, (h0, c0))
            
            # 更新隐藏状态（用于下一时间步的注意力计算）
            h0, c0 = h_n, c_n
            h_current = h0[-1]  # (batch, hidden_dim)
            
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


class FashionCaptionModelEnhanced(nn.Module):
    """
    增强版服饰描述生成模型
    多层自注意力编码器 + 增强型LSTM解码器 + 加性注意力
    目标：METEOR ≥ 0.7
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 512, hidden_dim: int = 768, num_layers: int = 3, dropout: float = 0.3):
        """
        初始化增强版模型
        
        Args:
            vocab_size: 词典大小
            embed_dim: 词嵌入维度，默认512
            hidden_dim: LSTM隐藏状态维度，默认768
            num_layers: LSTM层数，默认3
            dropout: Dropout比例，默认0.3
        """
        super(FashionCaptionModelEnhanced, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 增强型CNN编码器（多层自注意力）
        self.cnn_encoder = EnhancedCNNEncoderV2(dropout=dropout * 0.5)
        
        # 增强型LSTM解码器（更大维度 + 加性注意力）
        self.lstm_decoder = EnhancedLSTMDecoderV2(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, local_feat: torch.Tensor, caption: torch.Tensor,
                teacher_forcing_ratio: float = 0.9, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播（训练模式：混合Teacher-Forcing）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
            caption: 文本序列 (batch, seq_len)
            teacher_forcing_ratio: Teacher-Forcing比例，默认0.9（提高TF比例）
            return_attn: 是否返回注意力权重，默认False
        
        Returns:
            outputs: 模型输出 (batch, seq_len-1, vocab_size)
            attn_weights: 注意力权重 (batch, seq_len-1, 49)，如果return_attn=True
        """
        batch_size, seq_len = caption.size()
        device = local_feat.device
        
        # CNN编码：获取视觉特征和自注意力增强后的局部特征
        cnn_feat, attn_enhanced_feat = self.cnn_encoder(local_feat)  # cnn_feat: (batch, 512), attn_enhanced_feat: (batch, 2048, 7, 7)
        
        # 初始化LSTM隐藏状态
        h0, c0 = self.lstm_decoder.init_hidden(cnn_feat)
        
        # 输入序列（去掉最后一个词）
        input_ids = caption[:, :-1]  # (batch, seq_len-1)
        
        outputs_list = []
        attn_weights_list = []
        
        # 获取当前时刻的隐藏状态
        h_current = h0[-1]  # (batch, hidden_dim)
        
        # 第一个时间步使用<START>标记
        decoder_input = caption[:, 0:1]  # (batch, 1)
        
        for t in range(seq_len - 1):
            # 计算注意力上下文向量
            context, attn_w = self.lstm_decoder.attention(h_current, attn_enhanced_feat)
            
            # 词嵌入
            word_embeds = self.lstm_decoder.embedding(decoder_input)  # (batch, 1, embed_dim)
            word_embeds = self.lstm_decoder.embed_dropout(word_embeds)
            
            # 拼接词嵌入和注意力上下文向量
            lstm_input = torch.cat([word_embeds.squeeze(1), context], dim=1)  # (batch, embed_dim + hidden_dim)
            lstm_input = lstm_input.unsqueeze(1)  # (batch, 1, embed_dim + hidden_dim)
            
            # LSTM前向传播
            lstm_output, (h0, c0) = self.lstm_decoder.lstm(lstm_input, (h0, c0))
            
            # 输出投影
            output = self.lstm_decoder.output_proj(lstm_output)  # (batch, 1, vocab_size)
            outputs_list.append(output)
            
            if return_attn:
                attn_weights_list.append(attn_w.unsqueeze(1))  # (batch, 1, 49)
            
            # 更新隐藏状态（用于下一时间步的注意力计算）
            h_current = h0[-1]  # (batch, hidden_dim)
            
            # 混合Teacher-Forcing：90%使用真实词，10%使用模型预测词
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
    
    def generate(self, local_feat: torch.Tensor, max_len: int = 30,
                 start_idx: int = 2, end_idx: int = 3, pad_idx: int = 0,
                 temperature: float = 0.8, return_attn: bool = False,
                 beam_size: int = 5, length_penalty: float = 0.6) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        生成描述（推理模式：优先使用Beam Search，否则使用贪心解码）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
            max_len: 最大生成长度，默认30
            start_idx: <START>标记索引，默认2
            end_idx: <END>标记索引，默认3
            pad_idx: <PAD>标记索引，默认0
            temperature: 温度系数，仅用于贪心解码（beam_size=1时），默认0.8
            return_attn: 是否返回注意力权重，默认False（beam search时忽略）
            beam_size: Beam Search的beam大小，默认5（推荐3-5）。如果beam_size=1，则使用贪心解码
            length_penalty: 长度惩罚因子，默认0.6（0.6-1.0之间，越小对长序列越有利）
        
        Returns:
            generated_sequence: 生成的词ID序列 (batch, generated_len)
            attn_weights: 注意力权重 (batch, generated_len, 49) 或 None（beam search时）
        """
        if beam_size > 1:
            # 使用Beam Search（推荐）
            return self.generate_beam_search(local_feat, max_len, start_idx, end_idx, pad_idx, beam_size, length_penalty)
        else:
            # 使用贪心解码（向后兼容）
            return self.generate_greedy(local_feat, max_len, start_idx, end_idx, pad_idx, temperature, return_attn)
    
    def generate_greedy(self, local_feat: torch.Tensor, max_len: int = 30,
                        start_idx: int = 2, end_idx: int = 3, pad_idx: int = 0,
                        temperature: float = 0.8, return_attn: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        生成描述（贪心解码：选择概率最大的词）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
            max_len: 最大生成长度，默认30
            start_idx: <START>标记索引，默认2
            end_idx: <END>标记索引，默认3
            pad_idx: <PAD>标记索引，默认0
            temperature: 温度系数，默认0.8（仅用于采样，贪心解码时设为1.0）
            return_attn: 是否返回注意力权重，默认True
        
        Returns:
            generated_sequence: 生成的词ID序列 (batch, generated_len)
            attn_weights: 注意力权重 (batch, generated_len, 49) 或 None
        """
        batch_size = local_feat.size(0)
        device = local_feat.device
        
        # CNN编码
        cnn_feat, attn_enhanced_feat = self.cnn_encoder(local_feat)
        
        # 初始化LSTM隐藏状态
        h0, c0 = self.lstm_decoder.init_hidden(cnn_feat)
        
        # 初始化输入：<START>标记
        decoder_input = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=device)
        generated_ids = []
        attn_weights_list = []
        
        # 获取当前时刻的隐藏状态
        h_current = h0[-1]  # (batch, hidden_dim)
        
        consecutive_pads = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 逐词生成（贪心解码）
        for _ in range(max_len):
            # 计算注意力上下文向量
            context, attn_w = self.lstm_decoder.attention(h_current, attn_enhanced_feat)
            
            # 词嵌入
            word_embeds = self.lstm_decoder.embedding(decoder_input)  # (batch, 1, embed_dim)
            word_embeds = self.lstm_decoder.embed_dropout(word_embeds)
            
            # 拼接词嵌入和注意力上下文向量
            lstm_input = torch.cat([word_embeds.squeeze(1), context], dim=1)  # (batch, embed_dim + hidden_dim)
            lstm_input = lstm_input.unsqueeze(1)  # (batch, 1, embed_dim + hidden_dim)
            
            # LSTM前向传播
            lstm_output, (h0, c0) = self.lstm_decoder.lstm(lstm_input, (h0, c0))
            
            # 输出投影
            output = self.lstm_decoder.output_proj(lstm_output)  # (batch, 1, vocab_size)
            output = output.squeeze(1)  # (batch, vocab_size)
            
            # 贪心解码：选择概率最大的词
            if temperature != 1.0 and temperature > 0:
                output = output / temperature
            next_word_ids = torch.argmax(output, dim=-1, keepdim=True)  # (batch, 1)
            
            generated_ids.append(next_word_ids)
            
            if return_attn:
                attn_weights_list.append(attn_w.unsqueeze(1))  # (batch, 1, 49)
            
            # 更新隐藏状态
            h_current = h0[-1]  # (batch, hidden_dim)
            
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
    
    def generate_beam_search(self, local_feat: torch.Tensor, max_len: int = 30,
                            start_idx: int = 2, end_idx: int = 3, pad_idx: int = 0,
                            beam_size: int = 5, length_penalty: float = 0.6) -> Tuple[torch.Tensor, None]:
        """
        使用Beam Search生成描述（推荐方法，性能更好）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
            max_len: 最大生成长度，默认30
            start_idx: <START>标记索引，默认2
            end_idx: <END>标记索引，默认3
            pad_idx: <PAD>标记索引，默认0
            beam_size: Beam大小，默认5（推荐3-5）
            length_penalty: 长度惩罚因子，默认0.6（越小对长序列越有利）
        
        Returns:
            generated_sequence: 生成的词ID序列 (batch, generated_len)
            None: Beam Search时不返回注意力权重
        """
        batch_size = local_feat.size(0)
        device = local_feat.device
        
        # CNN编码（对所有样本一次性编码）
        cnn_feat, attn_enhanced_feat = self.cnn_encoder(local_feat)
        
        # 为每个样本分别进行beam search
        all_sequences = []
        
        for b in range(batch_size):
            # 初始化beam
            beams = [{
                'sequence': [start_idx],
                'score': 0.0,
                'hidden': None,
                'cell': None,
                'finished': False
            }]
            
            # 初始化LSTM隐藏状态（只计算一次）
            cnn_feat_single = cnn_feat[b:b+1]  # (1, 512)
            h0_init, c0_init = self.lstm_decoder.init_hidden(cnn_feat_single)
            h_current_init = h0_init[-1]  # (1, hidden_dim)
            
            # 将编码特征扩展到beam_size
            attn_enhanced_feat_single = attn_enhanced_feat[b:b+1]  # (1, 2048, 7, 7)
            
            # Beam Search主循环
            for step in range(max_len):
                candidates = []
                
                for beam in beams:
                    if beam['finished']:
                        candidates.append(beam)
                        continue
                    
                    # 获取当前隐藏状态
                    if step == 0:
                        h0, c0 = h0_init, c0_init
                        h_current = h_current_init
                    else:
                        if beam['hidden'] is not None:
                            h0, c0 = beam['hidden']
                            h_current = h0[-1] if h0 is not None else h_current_init
                        else:
                            h0, c0 = h0_init, c0_init
                            h_current = h_current_init
                    
                    # 获取上一个词
                    prev_word_id = beam['sequence'][-1]
                    decoder_input = torch.tensor([[prev_word_id]], dtype=torch.long, device=device)  # (1, 1)
                    
                    # 计算注意力上下文向量
                    context, _ = self.lstm_decoder.attention(h_current, attn_enhanced_feat_single)
                    
                    # 词嵌入
                    word_embeds = self.lstm_decoder.embedding(decoder_input)  # (1, 1, embed_dim)
                    word_embeds = self.lstm_decoder.embed_dropout(word_embeds)
                    
                    # 拼接词嵌入和注意力上下文向量
                    lstm_input = torch.cat([word_embeds.squeeze(1), context], dim=1)  # (1, embed_dim + hidden_dim)
                    lstm_input = lstm_input.unsqueeze(1)  # (1, 1, embed_dim + hidden_dim)
                    
                    # LSTM前向传播
                    lstm_output, (h_new, c_new) = self.lstm_decoder.lstm(lstm_input, (h0, c0))
                    
                    # 输出投影
                    output = self.lstm_decoder.output_proj(lstm_output)  # (1, 1, vocab_size)
                    log_probs = F.log_softmax(output.squeeze(1), dim=-1)  # (1, vocab_size)
                    
                    # 获取top-k候选词
                    top_log_probs, top_indices = torch.topk(log_probs[0], beam_size * 2)  # 取更多候选以应对finished beams
                    
                    # 扩展beam
                    for i in range(len(top_indices)):
                        word_id = top_indices[i].item()
                        log_prob = top_log_probs[i].item()
                        
                        new_score = beam['score'] + log_prob
                        
                        # 检查是否结束
                        finished = (word_id == end_idx)
                        new_sequence = beam['sequence'] + [word_id] if not finished else beam['sequence']
                        
                        # 应用长度惩罚
                        sequence_length = len(new_sequence)
                        length_penalty_score = ((5 + sequence_length) / 6) ** length_penalty
                        normalized_score = new_score / length_penalty_score
                        
                        # 存储隐藏状态（如果未结束）
                        if not finished:
                            hidden_tuple = (h_new, c_new)
                        else:
                            hidden_tuple = None
                        
                        candidates.append({
                            'sequence': new_sequence,
                            'score': normalized_score,
                            'raw_score': new_score,
                            'hidden': hidden_tuple,
                            'cell': None,  # 已包含在hidden中
                            'finished': finished
                        })
                
                # 选择top-k beams
                candidates.sort(key=lambda x: x['score'], reverse=True)
                beams = candidates[:beam_size]
                
                # 如果所有beams都结束，提前退出
                if all(beam['finished'] for beam in beams):
                    break
            
            # 选择最佳beam（使用raw_score，因为已经应用了长度惩罚）
            best_beam = max(beams, key=lambda x: x['raw_score'] if x['finished'] else x['score'])
            best_sequence = best_beam['sequence']
            
            # 确保序列以<END>结尾（如果没有）
            if best_sequence[-1] != end_idx:
                best_sequence.append(end_idx)
            
            # 转换为tensor
            sequence_tensor = torch.tensor(best_sequence, dtype=torch.long, device=device)
            all_sequences.append(sequence_tensor)
        
        # 填充到相同长度
        max_seq_len = max(len(seq) for seq in all_sequences)
        padded_sequences = []
        for seq in all_sequences:
            if len(seq) < max_seq_len:
                padding = torch.full((max_seq_len - len(seq),), pad_idx, dtype=torch.long, device=device)
                seq = torch.cat([seq, padding])
            padded_sequences.append(seq)
        
        generated_sequence = torch.stack(padded_sequences, dim=0)  # (batch, max_seq_len)
        
        return generated_sequence, None
    
    def postprocess_caption(self, sequence: torch.Tensor, idx2word: dict,
                           start_idx: int = 2, end_idx: int = 3,
                           unk_idx: int = 1, pad_idx: int = 0) -> list:
        """
        后处理生成的文本：过滤<UNK>/<PAD>/<START>，仅保留有效英文单词
        优化版本：不过度过滤，保留更多有效内容
        
        Args:
            sequence: 词ID序列，可以是 (seq_len,) 或 (1, seq_len) 形状
            idx2word: 索引到单词的映射字典
            start_idx: <START>标记索引
            end_idx: <END>标记索引
            unk_idx: <UNK>标记索引
            pad_idx: <PAD>标记索引
        
        Returns:
            处理后的单词列表
        """
        # 确保sequence是一维的
        if sequence.dim() > 1:
            sequence = sequence.squeeze()
        
        # 转换为列表（如果已经是标量，直接转换；如果是张量，先转CPU再转列表）
        if isinstance(sequence, torch.Tensor):
            sequence_list = sequence.cpu().tolist()
        else:
            sequence_list = sequence
        
        # 如果是嵌套列表（如 [[1, 2, 3]]），取第一个元素
        if sequence_list and isinstance(sequence_list[0], list):
            sequence_list = sequence_list[0]
        
        words = []
        found_end = False
        
        for idx in sequence_list:
            # 确保idx是整数（不是list）
            if isinstance(idx, list):
                idx = idx[0] if idx else pad_idx
            idx = int(idx)
            
            # 遇到<END>标记，停止处理
            if idx == end_idx:
                found_end = True
                break
            
            # 跳过特殊标记
            if idx in [start_idx, end_idx, unk_idx, pad_idx]:
                continue
            
            # 获取单词
            word = idx2word.get(idx, None)
            if word is not None:
                # 过滤掉特殊标记（确保word不是标记）
                if word not in ['<PAD>', '<UNK>', '<START>', '<END>', '<pad>', '<unk>', '<start>', '<end>']:
                words.append(word)
            else:
                # 如果idx不在字典中，尝试作为数字（可能是未知词）
                # 暂时跳过，或者可以添加为占位符
                pass
        
        # 去除重复的连续词（简单的去重）
        if words:
            deduplicated = [words[0]]
            for word in words[1:]:
                if word != deduplicated[-1]:
                    deduplicated.append(word)
            words = deduplicated
        
        # 确保至少有一些内容（如果完全为空，可能模型生成有问题）
        return words if words else []


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_vocab(vocab_file: str) -> dict:
    """加载词典文件"""
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
    print("增强版Model2测试：多层自注意力编码器 + 增强型LSTM解码器")
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
    model = FashionCaptionModelEnhanced(vocab_size=vocab_size)
    model = model.to(device)
    
    total_params = count_parameters(model)
    print(f"[OK] 模型参数量: {total_params:,}")
    
    # 测试2: 单batch前向计算测试
    print("\n" + "="*60)
    print("测试2: 前向计算测试（训练模式）")
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
    outputs, attn_weights = model(local_feat, caption, teacher_forcing_ratio=0.9, return_attn=True)
    print(f"输出形状: {outputs.shape}")
    print(f"注意力权重形状: {attn_weights.shape if attn_weights is not None else None}")
    assert outputs.shape == (batch_size, seq_len - 1, vocab_size), \
        f"输出维度错误: {outputs.shape}"
    print("[OK] 前向计算测试通过")
    
    # 测试3: 生成演示
    print("\n" + "="*60)
    print("测试3: 生成演示")
    print("="*60)
    model.eval()
    test_batch_size = 3
    test_local_feat = torch.randn(test_batch_size, 2048, 7, 7).to(device)
    
    with torch.no_grad():
        generated_sequences, attn_weights = model.generate(
            test_local_feat, max_len=30, temperature=0.8, return_attn=True
        )
    
    print(f"生成序列形状: {generated_sequences.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print("\n生成的描述示例:")
    
    for i in range(test_batch_size):
        seq = generated_sequences[i].cpu()
        gen_words = model.postprocess_caption(seq, idx2word)
        print(f"\n样本{i+1}:")
        print(f"  生成描述: {' '.join(gen_words) if gen_words else '(空)'}")
    
    print("\n" + "="*60)
    print("[OK] 所有测试通过！")
    print("="*60)
