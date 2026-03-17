"""
ResNet编码器 + 6层基础RNN解码器服饰描述生成模型
Model1_ResNet: ResNet Encoder + 6-Layer Basic RNN Decoder
改进点：使用ResNet作为编码器替代常规CNN，提升特征提取能力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import json
from typing import Tuple, Optional
import random


class ResNetEncoder(nn.Module):
    """
    ResNet特征编码器
    使用预训练的ResNet作为backbone，适配2048维输入特征
    输入：局部特征 (batch, 2048, 7, 7)
    输出：512维视觉特征向量 (batch, 512)
    """
    
    def __init__(self, resnet_type: str = 'resnet50', pretrained: bool = True):
        """
        初始化ResNet编码器
        
        Args:
            resnet_type: ResNet类型，'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
            pretrained: 是否使用预训练权重，默认True
        """
        super(ResNetEncoder, self).__init__()
        
        self.resnet_type = resnet_type
        self.pretrained = pretrained
        
        # 加载预训练ResNet（去掉最后的分类层）
        # 使用新的weights参数替代deprecated的pretrained参数
        try:
            # 新版本torchvision使用weights参数
            if resnet_type == 'resnet18':
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                resnet = models.resnet18(weights=weights)
                self.feature_dim = 512
            elif resnet_type == 'resnet34':
                weights = models.ResNet34_Weights.DEFAULT if pretrained else None
                resnet = models.resnet34(weights=weights)
                self.feature_dim = 512
            elif resnet_type == 'resnet50':
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                resnet = models.resnet50(weights=weights)
                self.feature_dim = 2048
            elif resnet_type == 'resnet101':
                weights = models.ResNet101_Weights.DEFAULT if pretrained else None
                resnet = models.resnet101(weights=weights)
                self.feature_dim = 2048
            elif resnet_type == 'resnet152':
                weights = models.ResNet152_Weights.DEFAULT if pretrained else None
                resnet = models.resnet152(weights=weights)
                self.feature_dim = 2048
            else:
                raise ValueError(f"不支持的ResNet类型: {resnet_type}")
        except AttributeError:
            # 旧版本torchvision使用pretrained参数
            if resnet_type == 'resnet18':
                resnet = models.resnet18(pretrained=pretrained)
                self.feature_dim = 512
            elif resnet_type == 'resnet34':
                resnet = models.resnet34(pretrained=pretrained)
                self.feature_dim = 512
            elif resnet_type == 'resnet50':
                resnet = models.resnet50(pretrained=pretrained)
                self.feature_dim = 2048
            elif resnet_type == 'resnet101':
                resnet = models.resnet101(pretrained=pretrained)
                self.feature_dim = 2048
            elif resnet_type == 'resnet152':
                resnet = models.resnet152(pretrained=pretrained)
                self.feature_dim = 2048
            else:
                raise ValueError(f"不支持的ResNet类型: {resnet_type}")
        
        # 输入适配层：将2048维特征映射到ResNet layer1的输入维度（通常是64通道）
        # 直接适配到layer1的输入通道数（ResNet18/34是64，ResNet50/101/152也是64）
        # 使用1x1卷积进行降维：2048 -> 64
        self.input_adapter = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 提取ResNet的backbone（跳过conv1和maxpool，直接从layer1开始）
        # 因为输入已经是7x7的小特征图，不需要ResNet的初始下采样
        # ResNet结构: layer1 -> layer2 -> layer3 -> layer4
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 输出适配层：将ResNet的特征维度映射到512维
        # ResNet50/101/152输出2048维，ResNet18/34输出512维
        if self.feature_dim != 512:
            self.output_adapter = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(self.feature_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )
        else:
            # ResNet18/34已经是512维，只需要池化和归一化
            self.output_adapter = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )
        
        # 初始化适配层权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化适配层权重"""
        for m in self.input_adapter.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for m in self.output_adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, local_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
        
        Returns:
            视觉特征向量 (batch, 512)
        """
        # 输入适配：2048维 -> 64维（ResNet layer1的输入通道数）
        # (batch, 2048, 7, 7) -> (batch, 64, 7, 7)
        x = self.input_adapter(local_feat)
        
        # ResNet特征提取（跳过conv1和maxpool，直接从layer1开始）
        # (batch, 64, 7, 7) -> ResNet layers -> (batch, feature_dim, H, W)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 输出适配：ResNet特征 -> 512维
        # (batch, feature_dim, H, W) -> (batch, 512)
        x = self.output_adapter(x)
        
        return x


class AttentionLayer(nn.Module):
    """
    加性注意力层（Bahdanau Attention）
    用于动态关注ResNet编码器的不同特征
    """
    
    def __init__(self, hidden_dim: int = 1024, feature_dim: int = 512, attn_dim: int = 768):
        """
        初始化加性注意力层（优化版 - 更大容量）
        
        Args:
            hidden_dim: LSTM隐藏状态维度，默认1024（优化：768→1024）
            feature_dim: 视觉特征维度，默认512
            attn_dim: 注意力维度，默认768（优化：512→768，更强的注意力）
        """
        super(AttentionLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.attn_dim = attn_dim
        
        # 将隐藏状态映射到注意力维度（更深的结构）
        self.hidden_proj = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim * 2),
            nn.LayerNorm(attn_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(attn_dim * 2, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 将视觉特征映射到注意力维度（更深的结构）
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, attn_dim * 2),
            nn.LayerNorm(attn_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(attn_dim * 2, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 注意力得分计算层（更深的结构）
        self.attention = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(attn_dim, attn_dim // 2),
            nn.Tanh(),
            nn.Linear(attn_dim // 2, 1)
        )
        
        # 输出投影层（带残差连接的思想）
        self.output_proj = nn.Sequential(
            nn.Linear(attn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in [self.hidden_proj, self.feature_proj, self.output_proj]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
        
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, hidden_state: torch.Tensor, visual_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：计算加性注意力上下文向量
        
        Args:
            hidden_state: LSTM当前时刻隐藏状态 (batch, hidden_dim)
            visual_feat: ResNet编码器输出的特征 (batch, feature_dim)
        
        Returns:
            context: 注意力上下文向量 (batch, hidden_dim)
            attn_weight: 注意力权重 (batch, 1)，用于可视化
        """
        batch_size = hidden_state.size(0)
        
        # 投影到注意力维度
        hidden_proj = self.hidden_proj(hidden_state)  # (batch, attn_dim)
        feature_proj = self.feature_proj(visual_feat)  # (batch, attn_dim)
        
        # 计算注意力得分（加性注意力）
        # hidden_proj: (batch, attn_dim) → (batch, 1, attn_dim)
        # feature_proj: (batch, attn_dim) → (batch, 1, attn_dim)
        hidden_expanded = hidden_proj.unsqueeze(1)  # (batch, 1, attn_dim)
        feature_expanded = feature_proj.unsqueeze(1)  # (batch, 1, attn_dim)
        
        # 加性注意力：tanh(hidden + feature)
        attn_input = hidden_expanded + feature_expanded  # (batch, 1, attn_dim)
        scores = self.attention(attn_input)  # (batch, 1, 1)
        scores = scores.squeeze(2)  # (batch, 1)
        
        # Softmax归一化（虽然只有一个特征，但保持一致性）
        attn_weight = F.softmax(scores, dim=1)  # (batch, 1)
        
        # 加权求和得到上下文向量
        context = attn_weight * feature_proj  # (batch, attn_dim)
        
        # 投影到hidden_dim维度
        context = self.output_proj(context)  # (batch, hidden_dim)
        
        return context, attn_weight.squeeze(1)


class LSTMAttentionDecoder(nn.Module):
    """
    LSTM解码器 + 注意力机制（改进版）
    使用LSTM替代RNN，添加注意力机制，提升生成质量和长度
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 768, hidden_dim: int = 1024, num_layers: int = 5, dropout: float = 0.4):
        """
        初始化LSTM解码器（带注意力）- 超大容量版（目标：至少一个指标80%+）
        
        Args:
            vocab_size: 词典大小
            embed_dim: 词嵌入维度，默认768（优化：512→768，更强的词表示）
            hidden_dim: LSTM隐藏状态维度，默认1024（优化：768→1024，最大化模型容量）
            num_layers: LSTM层数，默认5（5层深度LSTM）
            dropout: Dropout比例，默认0.4（防止过拟合）
        """
        super(LSTMAttentionDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 词嵌入层（更大的维度）
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout * 0.5)
        
        # 注意力层（优化：更大的attn_dim）
        self.attention = AttentionLayer(hidden_dim=hidden_dim, feature_dim=512, attn_dim=768)
        
        # LSTM层（替代RNN）- 超大容量版
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,  # 768 + 1024 = 1792（更大的输入）
            hidden_size=hidden_dim,  # 1024（优化：768→1024）
            num_layers=num_layers,  # 5层（深度优化）
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 隐藏/细胞状态初始化（从512维视觉特征映射到1024维）- 优化版
        self.hidden_init_proj = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        self.cell_init_proj = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # 输出层（带归一化和Dropout）- 更深的结构
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        # 词嵌入层：Xavier_uniform
        nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data[0].fill_(0)
        
        # 隐藏/细胞状态初始化层（Sequential结构）
        for module in [self.hidden_init_proj, self.cell_init_proj]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
        
        # 输出层
        for m in self.output_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def init_hidden(self, visual_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化LSTM隐藏状态和细胞状态
        
        Args:
            visual_feat: ResNet编码器输出的特征 (batch, 512)
        
        Returns:
            h0: 初始隐藏状态 (num_layers, batch, hidden_dim)
            c0: 初始细胞状态 (num_layers, batch, hidden_dim)
        """
        # 将512维特征映射到1024维（优化：768→1024）
        h0 = self.hidden_init_proj(visual_feat)  # (batch, 1024)
        c0 = self.cell_init_proj(visual_feat)  # (batch, 1024)
        
        # 扩展为多层LSTM的初始状态
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, 1024)
        c0 = c0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, 1024)
        
        return h0, c0
    
    def forward(self, word_ids: torch.Tensor, visual_feat: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                return_attn: bool = False) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播（训练模式：Teacher-Forcing + 注意力）
        
        Args:
            word_ids: 词ID序列 (batch, seq_len)
            visual_feat: ResNet编码器输出的特征 (batch, 512)
            hidden: 初始隐藏状态和细胞状态，如果为None则自动初始化
            return_attn: 是否返回注意力权重，默认False
        
        Returns:
            outputs: LSTM输出 (batch, seq_len, vocab_size)
            hidden: 最终隐藏状态和细胞状态
            attn_weights: 注意力权重 (batch, seq_len, 1)，如果return_attn=True
        """
        batch_size, seq_len = word_ids.size()
        
        # 初始化隐藏状态和细胞状态
        if hidden is None:
            h0, c0 = self.init_hidden(visual_feat)
        else:
            h0, c0 = hidden
        
        # 词嵌入
        word_embeds = self.embedding(word_ids)  # (batch, seq_len, embed_dim)
        word_embeds = self.embed_dropout(word_embeds)
        
        # 逐时间步处理（使用注意力）
        outputs_list = []
        attn_weights_list = []
        
        # 获取当前时刻的隐藏状态（使用最后一层的隐藏状态）
        h_current = h0[-1]  # (batch, hidden_dim)
        
        for t in range(seq_len):
            # 计算注意力上下文向量
            context, attn_w = self.attention(h_current, visual_feat)  # context: (batch, hidden_dim), attn_w: (batch,)
            
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
                attn_weights_list.append(attn_w.unsqueeze(1))  # (batch, 1, 1)
        
        # 拼接所有时间步的输出
        outputs = torch.cat(outputs_list, dim=1)  # (batch, seq_len, vocab_size)
        
        attn_weights = None
        if return_attn:
            attn_weights = torch.cat(attn_weights_list, dim=1)  # (batch, seq_len, 1)
        
        return outputs, (h0, c0), attn_weights


class FashionCaptionModelResNet(nn.Module):
    """
    服饰描述生成模型：ResNet编码器 + LSTM解码器 + 注意力机制（改进版）
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 768, hidden_dim: int = 1024, 
                 resnet_type: str = 'resnet50', pretrained: bool = True,
                 num_layers: int = 5, dropout: float = 0.4):
        """
        初始化模型（超大容量版 - 目标：至少一个指标80%+）
        
        Args:
            vocab_size: 词典大小
            embed_dim: 词嵌入维度，默认768（优化：512→768，更强的词表示）
            hidden_dim: LSTM隐藏状态维度，默认1024（优化：768→1024，最大化模型容量）
            resnet_type: ResNet类型，默认'resnet50'
            pretrained: 是否使用预训练权重，默认True
            num_layers: LSTM层数，默认5（5层深度LSTM）
            dropout: Dropout比例，默认0.4（防止过拟合）
        """
        super(FashionCaptionModelResNet, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.resnet_type = resnet_type
        self.num_layers = num_layers
        
        # ResNet编码器
        self.cnn_encoder = ResNetEncoder(resnet_type=resnet_type, pretrained=pretrained)
        
        # LSTM解码器（带注意力）
        self.lstm_decoder = LSTMAttentionDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 打印模型参数量
        self._print_model_info()
    
    def _print_model_info(self):
        """打印模型各部分参数量"""
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        cnn_params = count_parameters(self.cnn_encoder)
        embed_params = count_parameters(self.lstm_decoder.embedding)
        lstm_params = count_parameters(self.lstm_decoder.lstm)
        attention_params = count_parameters(self.lstm_decoder.attention)
        output_params = count_parameters(self.lstm_decoder.output_proj)
        total_params = count_parameters(self)
        
        print("="*60)
        print(f"模型参数量统计 (ResNet编码器 + {self.num_layers}层深度LSTM解码器 + 注意力)")
        print("="*60)
        print(f"ResNet编码器: {cnn_params:,} 参数")
        print(f"词嵌入层: {embed_params:,} 参数")
        print(f"注意力层: {attention_params:,} 参数")
        print(f"{self.num_layers}层深度LSTM: {lstm_params:,} 参数 (深度优化，最大化序列建模能力)")
        print(f"输出层: {output_params:,} 参数")
        print(f"总参数量: {total_params:,} 参数")
        print(f"⚠️  深度模型：需要更多训练时间，但性能将显著提升！")
        print("="*60)
    
    def forward(self, local_feat: torch.Tensor, caption: torch.Tensor,
                teacher_forcing_ratio: float = 0.9, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播（训练模式：混合Teacher-Forcing + 注意力）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
            caption: 文本序列 (batch, seq_len)，包含<START>和<END>
            teacher_forcing_ratio: Teacher-Forcing比例，默认0.9
            return_attn: 是否返回注意力权重，默认False
        
        Returns:
            outputs: 模型输出 (batch, seq_len-1, vocab_size)
            attn_weights: 注意力权重 (batch, seq_len-1, 1)，如果return_attn=True
        """
        batch_size, seq_len = caption.size()
        device = local_feat.device
        
        # ResNet编码
        visual_feat = self.cnn_encoder(local_feat)  # (batch, 512)
        
        # 初始化LSTM隐藏状态
        h0, c0 = self.lstm_decoder.init_hidden(visual_feat)
        
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
            context, attn_w = self.lstm_decoder.attention(h_current, visual_feat)
            
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
                attn_weights_list.append(attn_w.unsqueeze(1))  # (batch, 1, 1)
            
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
            attn_weights = torch.cat(attn_weights_list, dim=1)  # (batch, seq_len-1, 1)
        
        return outputs, attn_weights
    
    def generate(self, local_feat: torch.Tensor, max_len: int = 35,
                 start_idx: int = 2, end_idx: int = 3, pad_idx: int = 0,
                 temperature: float = 0.8, return_attn: bool = False,
                 beam_size: int = 5, length_penalty: float = 0.6,
                 min_length: int = 25, end_penalty: float = 2.0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        生成描述（推理模式：优先使用Beam Search，否则使用贪心解码）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
            max_len: 最大生成长度，默认35（提升，允许生成长序列）
            start_idx: <START>标记索引，默认2
            end_idx: <END>标记索引，默认3
            pad_idx: <PAD>标记索引，默认0
            temperature: 温度系数，仅用于贪心解码（beam_size=1时），默认0.8
            return_attn: 是否返回注意力权重，默认False
            beam_size: Beam Search的beam大小，默认5。如果beam_size=1，则使用贪心解码
            length_penalty: 长度惩罚因子，默认0.6（越小对长序列越有利）
        
        Returns:
            generated_sequence: 生成的词ID序列 (batch, generated_len)
            attn_weights: 注意力权重 (batch, generated_len, 1) 或 None（beam search时）
        """
        if beam_size > 1:
            # 使用Beam Search（推荐）
            return self.generate_beam_search(local_feat, max_len, start_idx, end_idx, pad_idx, beam_size, length_penalty, min_length, end_penalty)
        else:
            # 使用贪心解码（向后兼容）
            return self.generate_greedy(local_feat, max_len, start_idx, end_idx, pad_idx, temperature, return_attn, min_length, end_penalty)
    
    def generate_greedy(self, local_feat: torch.Tensor, max_len: int = 35,
                        start_idx: int = 2, end_idx: int = 3, pad_idx: int = 0,
                        temperature: float = 0.8, return_attn: bool = True,
                        min_length: int = 25, end_penalty: float = 2.0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        生成描述（贪心解码：选择概率最大的词）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
            max_len: 最大生成长度，默认35
            start_idx: <START>标记索引，默认2
            end_idx: <END>标记索引，默认3
            pad_idx: <PAD>标记索引，默认0
            temperature: 温度系数，默认0.8（仅用于采样，贪心解码时设为1.0）
            return_attn: 是否返回注意力权重，默认True
        
        Returns:
            generated_sequence: 生成的词ID序列 (batch, generated_len)
            attn_weights: 注意力权重 (batch, generated_len, 1) 或 None
        """
        batch_size = local_feat.size(0)
        device = local_feat.device
        
        # ResNet编码
        visual_feat = self.cnn_encoder(local_feat)  # (batch, 512)
        
        # 初始化LSTM隐藏状态
        h0, c0 = self.lstm_decoder.init_hidden(visual_feat)
        
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
            context, attn_w = self.lstm_decoder.attention(h_current, visual_feat)
            
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
            
            # 对<END>标记进行惩罚（在序列长度小于min_length时）
            # 这样可以让模型生成更长的序列
            current_length = len(generated_ids)  # 当前已生成的词数（不包括START）
            if current_length < min_length:
                # 在序列长度小于min_length时，对<END>标记的概率进行惩罚
                # 惩罚因子：长度越短，惩罚越大
                penalty_factor = end_penalty * (1.0 - current_length / min_length)  # 从end_penalty逐步降到0
                output[:, end_idx] = output[:, end_idx] - penalty_factor  # 降低<END>的logit
            
            # 贪心解码：选择概率最大的词
            if temperature != 1.0 and temperature > 0:
                output = output / temperature
            next_word_ids = torch.argmax(output, dim=-1, keepdim=True)  # (batch, 1)
            
            generated_ids.append(next_word_ids)
            
            if return_attn:
                attn_weights_list.append(attn_w.unsqueeze(1))  # (batch, 1, 1)
            
            # 更新隐藏状态
            h_current = h0[-1]  # (batch, hidden_dim)
            
            # 检查终止条件（添加最小长度限制）
            end_mask = (next_word_ids.squeeze(1) == end_idx)
            pad_mask = (next_word_ids.squeeze(1) == pad_idx)
            consecutive_pads = consecutive_pads + pad_mask.long()
            consecutive_pads = consecutive_pads * pad_mask.long()
            
            # 如果序列长度小于min_length，强制忽略<END>标记
            if current_length < min_length:
                end_mask = torch.zeros_like(end_mask, dtype=torch.bool)  # 强制忽略<END>
            
            if torch.all(end_mask | (consecutive_pads >= 2)):
                break
            
            # 更新输入
            decoder_input = next_word_ids
        
        # 拼接生成的序列
        generated_sequence = torch.cat(generated_ids, dim=1)  # (batch, generated_len)
        
        attn_weights = None
        if return_attn:
            attn_weights = torch.cat(attn_weights_list, dim=1)  # (batch, generated_len, 1)
        
        return generated_sequence, attn_weights
    
    def generate_beam_search(self, local_feat: torch.Tensor, max_len: int = 35,
                            start_idx: int = 2, end_idx: int = 3, pad_idx: int = 0,
                            beam_size: int = 5, length_penalty: float = 0.6,
                            min_length: int = 25, end_penalty: float = 2.0) -> Tuple[torch.Tensor, None]:
        """
        使用Beam Search生成描述（推荐方法，性能更好）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
            max_len: 最大生成长度，默认35
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
        
        # ResNet编码
        visual_feat = self.cnn_encoder(local_feat)
        
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
            
            # 初始化LSTM隐藏状态
            visual_feat_single = visual_feat[b:b+1]  # (1, 512)
            h0_init, c0_init = self.lstm_decoder.init_hidden(visual_feat_single)
            h_current_init = h0_init[-1]  # (1, hidden_dim)
            
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
                    decoder_input = torch.tensor([[prev_word_id]], dtype=torch.long, device=device)
                    
                    # 计算注意力上下文向量
                    context, _ = self.lstm_decoder.attention(h_current, visual_feat_single)
                    
                    # 词嵌入
                    word_embeds = self.lstm_decoder.embedding(decoder_input)
                    word_embeds = self.lstm_decoder.embed_dropout(word_embeds)
                    
                    # 拼接词嵌入和注意力上下文向量
                    lstm_input = torch.cat([word_embeds.squeeze(1), context], dim=1)
                    lstm_input = lstm_input.unsqueeze(1)
                    
                    # LSTM前向传播
                    lstm_output, (h_new, c_new) = self.lstm_decoder.lstm(lstm_input, (h0, c0))
                    
                    # 输出投影
                    output = self.lstm_decoder.output_proj(lstm_output)
                    output_logits = output.squeeze(1)  # (1, vocab_size)
                    
                    # 对<END>标记进行惩罚（在序列长度小于min_length时）
                    current_seq_length = len(beam['sequence']) - 1  # 当前已生成的词数（不包括START）
                    if current_seq_length < min_length:
                        # 在序列长度小于min_length时，对<END>标记的概率进行惩罚
                        penalty_factor = end_penalty * (1.0 - current_seq_length / min_length)  # 从end_penalty逐步降到0
                        output_logits[0, end_idx] = output_logits[0, end_idx] - penalty_factor  # 降低<END>的logit
                    
                    log_probs = F.log_softmax(output_logits, dim=-1)
                    
                    # 获取top-k候选词
                    top_log_probs, top_indices = torch.topk(log_probs[0], beam_size * 2)
                    
                    # 扩展beam
                    for i in range(len(top_indices)):
                        word_id = top_indices[i].item()
                        log_prob = top_log_probs[i].item()
                        
                        new_score = beam['score'] + log_prob
                        
                        # 检查是否结束（添加最小长度限制）
                        finished = (word_id == end_idx) and (current_seq_length >= min_length)  # 只有长度>=min_length时才允许结束
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
                            'cell': None,
                            'finished': finished
                        })
                
                # 选择top-k beams
                candidates.sort(key=lambda x: x['score'], reverse=True)
                beams = candidates[:beam_size]
                
                # 如果所有beams都结束，提前退出
                if all(beam['finished'] for beam in beams):
                    break
            
            # 选择最佳beam
            best_beam = max(beams, key=lambda x: x['raw_score'] if x['finished'] else x['score'])
            best_sequence = best_beam['sequence']
            
            # 确保序列以<END>结尾
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
        
        generated_sequence = torch.stack(padded_sequences, dim=0)
        
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
        
        # 转换为列表
        if isinstance(sequence, torch.Tensor):
            sequence_list = sequence.cpu().tolist()
        else:
            sequence_list = sequence
        
        # 如果是嵌套列表，取第一个元素
        if sequence_list and isinstance(sequence_list[0], list):
            sequence_list = sequence_list[0]
        
        words = []
        found_end = False
        
        for idx in sequence_list:
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
        
        # 去除重复的连续词
        if words:
            deduplicated = [words[0]]
            for word in words[1:]:
                if word != deduplicated[-1]:
                    deduplicated.append(word)
            words = deduplicated
        
        # 确保至少有一些内容
        return words if words else []


def load_vocab(vocab_file: str) -> dict:
    """
    加载词典文件
    
    Args:
        vocab_file: 词典文件路径
    
    Returns:
        包含vocab_size的字典
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
    print("模型测试 (ResNet编码器)")
    print("="*60)
    
    # 加载词典
    vocab_file = "dataset/vocab.json"
    try:
        vocab_info = load_vocab(vocab_file)
        vocab_size = vocab_info['vocab_size']
        idx2word = vocab_info['idx2word']
        print(f"\n[OK] 词典加载成功，词典大小: {vocab_size}")
    except FileNotFoundError:
        print(f"\n[WARNING] 警告: 找不到词典文件 {vocab_file}，使用默认大小5000")
        vocab_size = 5000
        idx2word = {i: f"word_{i}" for i in range(vocab_size)}
    
    # 创建模型
    print("\n" + "="*60)
    print("创建模型 (ResNet50编码器)")
    print("="*60)
    model = FashionCaptionModelResNet(vocab_size=vocab_size, resnet_type='resnet50', pretrained=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\n[OK] 模型已移动到设备: {device}")
    
    # 测试1: ResNet编码器维度验证
    print("\n" + "="*60)
    print("测试1: ResNet编码器维度验证")
    print("="*60)
    batch_size = 4
    local_feat = torch.randn(batch_size, 2048, 7, 7).to(device)
    print(f"输入局部特征形状: {local_feat.shape}")
    
    with torch.no_grad():
        cnn_feat = model.cnn_encoder(local_feat)
    print(f"ResNet输出特征形状: {cnn_feat.shape}")
    assert cnn_feat.shape == (batch_size, 512), f"ResNet输出维度错误: {cnn_feat.shape}"
    print("[OK] ResNet编码器维度验证通过")
    
    # 测试2: 单batch前向计算
    print("\n" + "="*60)
    print("测试2: 单batch前向计算")
    print("="*60)
    seq_len = 20
    caption = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    # 确保第一个词是<START>，最后一个词是<END>
    caption[:, 0] = 2  # <START>
    caption[:, -1] = 3  # <END>
    
    print(f"输入局部特征形状: {local_feat.shape}")
    print(f"输入文本序列形状: {caption.shape}")
    
    model.train()
    outputs = model(local_feat, caption)
    print(f"输出形状: {outputs.shape}")
    assert outputs.shape == (batch_size, seq_len - 1, vocab_size), \
        f"输出维度错误: {outputs.shape}"
    print("[OK] 前向计算测试通过")
    
    # 测试3: 3个样本生成演示
    print("\n" + "="*60)
    print("测试3: 生成演示（3个样本）")
    print("="*60)
    model.eval()
    test_batch_size = 3
    test_local_feat = torch.randn(test_batch_size, 2048, 7, 7).to(device)
    
    with torch.no_grad():
        generated_sequences = model.generate(test_local_feat, max_len=20)
    
    print(f"生成序列形状: {generated_sequences.shape}")
    print("\n生成的描述示例:")
    for i in range(test_batch_size):
        seq = generated_sequences[i].cpu()
        gen_words = model.postprocess_caption(seq, idx2word)
        print(f"  样本{i+1}: {' '.join(gen_words) if gen_words else '(空)'}")
    
    print("\n" + "="*60)
    print("[OK] 所有测试通过！")
    print("="*60)
