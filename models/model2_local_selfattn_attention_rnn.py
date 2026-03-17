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
    局部特征自注意力层
    将7×7局部特征展平为49个序列（每个维度2048），使用多头自注意力增强局部特征
    """
    
    def __init__(self, feature_dim: int = 2048, num_heads: int = 8):
        """
        初始化局部自注意力层
        
        Args:
            feature_dim: 特征维度，默认2048
            num_heads: 注意力头数，默认8（2048/8=256，可以整除）
        """
        super(LocalSelfAttention, self).__init__()
        
        assert feature_dim % num_heads == 0, f"feature_dim ({feature_dim}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # QKV线性层
        self.q_linear = nn.Linear(feature_dim, feature_dim)
        self.k_linear = nn.Linear(feature_dim, feature_dim)
        self.v_linear = nn.Linear(feature_dim, feature_dim)
        
        # 输出层
        self.output_linear = nn.Linear(feature_dim, feature_dim)
        
        # LayerNorm和Dropout
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.output_linear.weight)
    
    def forward(self, local_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播：局部特征自注意力
        
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
        
        # 残差连接
        residual = x
        
        # LayerNorm
        x = self.layer_norm1(x)
        
        # QKV计算
        # (batch, 49, 2048) → (batch, 49, 2048)
        Q = self.q_linear(x)  # (batch, 49, 2048)
        K = self.k_linear(x)   # (batch, 49, 2048)
        V = self.v_linear(x)   # (batch, 49, 2048)
        
        # 分多头
        # (batch, 49, 2048) → (batch, 49, num_heads, head_dim) → (batch, num_heads, 49, head_dim)
        Q = Q.view(batch_size, h * w, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 49, head_dim)
        K = K.view(batch_size, h * w, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 49, head_dim)
        V = V.view(batch_size, h * w, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 49, head_dim)
        
        # 计算注意力得分
        # (batch, num_heads, 49, head_dim) × (batch, num_heads, head_dim, 49) → (batch, num_heads, 49, 49)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, 49, 49)
        attn_weights = self.dropout(attn_weights)
        
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
        
        # 残差连接和LayerNorm
        output = self.layer_norm2(attn_output + residual)
        
        # 恢复形状：(batch, 49, 2048) → (batch, 2048, 49) → (batch, 2048, 7, 7)
        output = output.transpose(1, 2)  # (batch, 2048, 49)
        output = output.view(batch_size, feat_dim, h, w)  # (batch, 2048, 7, 7)
        
        return output


class EnhancedCNNEncoder(nn.Module):
    """
    增强型CNN特征编码器（新增局部特征自注意力层）
    输入：局部特征 (batch, 2048, 7, 7)
    输出：512维融合自注意力的视觉特征向量 (batch, 512)
    """
    
    def __init__(self):
        super(EnhancedCNNEncoder, self).__init__()
        
        # 局部特征自注意力层（新增）
        # 使用8头自注意力（2048/8=256，可以整除）
        self.self_attention = LocalSelfAttention(feature_dim=2048, num_heads=8)
        
        # 第一层卷积：2048 → 1024
        # Conv2d(2048, 1024, kernel_size=3, padding=1) → BatchNorm2d(1024) → ReLU → MaxPool2d(kernel_size=2, stride=2)
        # 输入: (batch, 2048, 7, 7)
        # 输出: (batch, 1024, 3, 3)
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二层卷积：1024 → 512
        # Conv2d(1024, 512, kernel_size=3, padding=1) → BatchNorm2d(512) → ReLU → AvgPool2d(kernel_size=3, stride=1)
        # 输入: (batch, 1024, 3, 3)
        # 输出: (batch, 512, 1, 1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=1)
        
        # 特征扁平化和全连接层
        # Flatten() → Linear(512, 512) → ReLU
        # 输入: (batch, 512, 1, 1) → Flatten → (batch, 512)
        # 输出: (batch, 512)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 512)
        
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
        
        # 全连接层：Xavier_uniform初始化
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, local_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
        
        Returns:
            visual_feat: 视觉特征向量 (batch, 512)
            attn_enhanced_feat: 自注意力增强后的局部特征 (batch, 2048, 7, 7)，用于后续注意力机制
        """
        # 局部特征自注意力（新增）
        # (batch, 2048, 7, 7) → (batch, 2048, 7, 7)
        attn_enhanced_feat = self.self_attention(local_feat)
        
        # 第一层卷积
        # (batch, 2048, 7, 7) → (batch, 1024, 7, 7) → (batch, 1024, 3, 3)
        x = self.conv1(attn_enhanced_feat)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 第二层卷积
        # (batch, 1024, 3, 3) → (batch, 512, 3, 3) → (batch, 512, 1, 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 扁平化和全连接
        # (batch, 512, 1, 1) → (batch, 512) → (batch, 512)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.relu(x)
        
        return x, attn_enhanced_feat


class AttentionLayer(nn.Module):
    """
    标准注意力层
    计算LSTM隐藏状态与局部特征的注意力得分，生成注意力上下文向量
    """
    
    def __init__(self, hidden_dim: int = 512, feature_dim: int = 2048):
        """
        初始化注意力层
        
        Args:
            hidden_dim: LSTM隐藏状态维度，默认512
            feature_dim: 局部特征维度，默认2048
        """
        super(AttentionLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        # 将局部特征映射到hidden_dim维度
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        
        # 注意力得分计算
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.feature_proj.weight)
        nn.init.constant_(self.feature_proj.bias, 0)
        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.constant_(self.attention.bias, 0)
    
    def forward(self, hidden_state: torch.Tensor, local_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：计算注意力上下文向量
        
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
        
        # 展平局部特征：(batch, 2048, 7, 7) → (batch, 2048, 49) → (batch, 49, 2048)
        h, w = local_feat.size(2), local_feat.size(3)
        local_feat_flat = local_feat.view(batch_size, self.feature_dim, h * w)  # (batch, 2048, 49)
        local_feat_flat = local_feat_flat.transpose(1, 2)  # (batch, 49, 2048)
        
        # 将局部特征映射到hidden_dim维度
        # (batch, 49, 2048) → (batch, 49, 512)
        local_feat_proj = self.feature_proj(local_feat_flat)  # (batch, 49, 512)
        
        # 计算注意力得分
        # hidden_state: (batch, 512) → (batch, 1, 512)
        # local_feat_proj: (batch, 49, 512)
        # scores: (batch, 49)
        hidden_expanded = hidden_state.unsqueeze(1)  # (batch, 1, 512)
        hidden_attn = self.attention(hidden_expanded)  # (batch, 1, 512)
        
        # 计算注意力得分：点积
        scores = torch.sum(hidden_attn * local_feat_proj, dim=2)  # (batch, 49)
        
        # Softmax归一化
        attn_weights = F.softmax(scores, dim=1)  # (batch, 49)
        
        # 加权求和得到上下文向量
        # (batch, 1, 49) × (batch, 49, 512) → (batch, 1, 512) → (batch, 512)
        context = torch.bmm(attn_weights.unsqueeze(1), local_feat_proj)  # (batch, 1, 512)
        context = context.squeeze(1)  # (batch, 512)
        
        return context, attn_weights


class AttentionLSTMDecoder(nn.Module):
    """
    注意力增强LSTM解码器（新增标准注意力机制）
    基础框架：复用2层LSTM解码器，但新增"标准注意力机制"
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512):
        """
        初始化注意力增强LSTM解码器
        
        Args:
            vocab_size: 词典大小
            embed_dim: 词嵌入维度，默认256
            hidden_dim: LSTM隐藏状态维度，默认512
        """
        super(AttentionLSTMDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # 词嵌入层：Embedding(vocab_size, 256, padding_idx=0)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 注意力层（新增）
        self.attention = AttentionLayer(hidden_dim=hidden_dim, feature_dim=2048)
        
        # 2层LSTM
        # input_size=256+512（词嵌入+注意力上下文向量）
        # hidden_size=512, num_layers=2, batch_first=True, bidirectional=False
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,  # 256 + 512 = 768
            hidden_size=hidden_dim,  # 512
            num_layers=2,  # 2层
            batch_first=True,
            bidirectional=False
        )
        
        # 隐藏/细胞状态初始化：CNN输出的512维特征 → Linear(512, 512) → tanh
        self.hidden_init = nn.Linear(hidden_dim, hidden_dim)
        self.cell_init = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出层：Linear(512, vocab_size)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        # 词嵌入层：Xavier_uniform
        nn.init.xavier_uniform_(self.embedding.weight)
        # padding_idx=0的权重设为0
        self.embedding.weight.data[0].fill_(0)
        
        # 隐藏/细胞状态初始化层：Xavier_uniform
        nn.init.xavier_uniform_(self.hidden_init.weight)
        nn.init.constant_(self.hidden_init.bias, 0)
        nn.init.xavier_uniform_(self.cell_init.weight)
        nn.init.constant_(self.cell_init.bias, 0)
        
        # 输出层：Xavier_uniform
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)
    
    def init_hidden(self, cnn_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化LSTM隐藏状态和细胞状态
        
        Args:
            cnn_feat: CNN编码器输出的特征 (batch, 512)
        
        Returns:
            h0: 初始隐藏状态 (2, batch, 512)
            c0: 初始细胞状态 (2, batch, 512)
        """
        # (batch, 512) → (batch, 512)
        h0 = self.hidden_init(cnn_feat)
        h0 = torch.tanh(h0)
        
        c0 = self.cell_init(cnn_feat)
        c0 = torch.tanh(c0)
        
        # 扩展为2层LSTM的初始状态
        h0 = h0.unsqueeze(0).repeat(2, 1, 1)  # (2, batch, 512)
        c0 = c0.unsqueeze(0).repeat(2, 1, 1)  # (2, batch, 512)
        
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
        
        # 逐时间步处理（需要计算每个时间步的注意力）
        outputs_list = []
        attn_weights_list = []
        
        # 获取当前时刻的隐藏状态（使用最后一层的隐藏状态）
        h_current = h0[-1]  # (batch, 512) - 最后一层的隐藏状态
        
        for t in range(seq_len):
            # 计算注意力上下文向量
            context, attn_w = self.attention(h_current, attn_enhanced_feat)  # context: (batch, 512), attn_w: (batch, 49)
            
            # 拼接词嵌入和注意力上下文向量
            # word_embeds[:, t, :]: (batch, 256)
            # context: (batch, 512)
            # lstm_input: (batch, 768)
            lstm_input = torch.cat([word_embeds[:, t, :], context], dim=1)  # (batch, 768)
            lstm_input = lstm_input.unsqueeze(1)  # (batch, 1, 768)
            
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
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512):
        """
        初始化模型
        
        Args:
            vocab_size: 词典大小
            embed_dim: 词嵌入维度，默认256
            hidden_dim: LSTM隐藏状态维度，默认512
        """
        super(FashionCaptionModelAttention, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # 增强型CNN编码器（新增局部特征自注意力层）
        self.cnn_encoder = EnhancedCNNEncoder()
        
        # 注意力增强LSTM解码器（新增标准注意力机制）
        self.lstm_decoder = AttentionLSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim
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
        cnn_feat, attn_enhanced_feat = self.cnn_encoder(local_feat)  # cnn_feat: (batch, 512), attn_enhanced_feat: (batch, 2048, 7, 7)
        
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
            word_embeds = self.lstm_decoder.embedding(decoder_input)  # (batch, 1, 256)
            
            # 拼接词嵌入和注意力上下文向量
            lstm_input = torch.cat([word_embeds.squeeze(1), context], dim=1)  # (batch, 768)
            lstm_input = lstm_input.unsqueeze(1)  # (batch, 1, 768)
            
            # LSTM前向传播
            lstm_output, (h0, c0) = self.lstm_decoder.lstm(lstm_input, (h0, c0))
            
            # 输出投影
            output = self.lstm_decoder.output_proj(lstm_output)  # (batch, 1, vocab_size)
            outputs_list.append(output)
            
            if return_attn:
                attn_weights_list.append(attn_w.unsqueeze(1))  # (batch, 1, 49)
            
            # 更新隐藏状态（用于下一时间步的注意力计算）
            h_current = h0[-1]  # (batch, 512)
            
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
        
        # 初始化LSTM隐藏状态
        h0, c0 = self.lstm_decoder.init_hidden(cnn_feat)
        
        # 初始化输入：<START>标记
        decoder_input = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=device)
        generated_ids = []
        attn_weights_list = []
        
        # 获取当前时刻的隐藏状态
        h_current = h0[-1]  # (batch, 512)
        
        consecutive_pads = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 逐词生成
        for _ in range(max_len):
            # 计算注意力上下文向量
            context, attn_w = self.lstm_decoder.attention(h_current, attn_enhanced_feat)
            
            # 词嵌入
            word_embeds = self.lstm_decoder.embedding(decoder_input)  # (batch, 1, 256)
            
            # 拼接词嵌入和注意力上下文向量
            lstm_input = torch.cat([word_embeds.squeeze(1), context], dim=1)  # (batch, 768)
            lstm_input = lstm_input.unsqueeze(1)  # (batch, 1, 768)
            
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

