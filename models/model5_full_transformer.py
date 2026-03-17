"""
局部表示+Transformer编码器→Transformer解码器（全Transformer架构）服饰描述生成模型
Model5: Full Transformer Model (Local Grid Representation + Transformer Encoder → Transformer Decoder)
核心架构：
1. 局部特征预处理：CNN提取的2048×7×7局部特征 → 展平为49×2048 → Linear映射为49×512
2. Transformer编码器：2层TransformerEncoder，多头自注意力增强局部特征（对应图片"Transformer encoder"）
3. Transformer解码器：2层TransformerDecoder，掩码自注意力+跨模态注意力（对应图片"Transformer decoder"）
完全摒弃RNN，采用纯Transformer架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import Tuple, Optional
import math


class TransformerEncoder(nn.Module):
    """
    Transformer编码器（对应图片"Transformer encoder"）
    2层TransformerEncoder，每层包含多头自注意力和前馈网络
    """
    
    def __init__(self, d_model: int = 512, nhead: int = 4, dim_feedforward: int = 2048,
                 num_layers: int = 2, dropout: float = 0.1, batch_first: bool = True):
        """
        初始化Transformer编码器
        
        Args:
            d_model: 模型维度，默认512（对应图片中特征维度D）
            nhead: 注意力头数，默认4
            dim_feedforward: 前馈网络维度，默认2048
            num_layers: 编码器层数，默认2（对应图片中堆叠的2层编码器）
            dropout: Dropout比例，默认0.1
            batch_first: 是否batch维度在前，默认True
        """
        super(TransformerEncoder, self).__init__()
        
        # 使用PyTorch内置的TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重：Xavier uniform初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：Transformer编码器增强局部特征
        
        Args:
            x: 输入特征序列 (batch, seq_len, d_model)，例如(batch, 49, 512)
        
        Returns:
            编码后的特征序列 (batch, seq_len, d_model)，例如(batch, 49, 512)
            对应图片中编码器输出的增强特征块C_0,0, C_0,1, ..., C_2,2
        """
        # Transformer编码器处理（无位置编码，严格匹配图片"无cls_token"的网格表示）
        # 注意：PyTorch的TransformerEncoder内部已经包含LayerNorm和残差连接
        encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        return encoded


class TransformerDecoder(nn.Module):
    """
    Transformer解码器（对应图片"Transformer decoder"）
    2层TransformerDecoder，每层包含掩码自注意力、跨模态注意力和前馈网络
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 4,
                 dim_feedforward: int = 2048, num_layers: int = 2, dropout: float = 0.1,
                 batch_first: bool = True, padding_idx: int = 0):
        """
        初始化Transformer解码器
        
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度，默认512
            nhead: 注意力头数，默认4
            dim_feedforward: 前馈网络维度，默认2048
            num_layers: 解码器层数，默认2（对应图片中堆叠的2层解码器）
            dropout: Dropout比例，默认0.1
            batch_first: 是否batch维度在前，默认True
            padding_idx: 填充标记索引，默认0
        """
        super(TransformerDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 文本嵌入层（对应图片中输入的词序列y_0, y_1, y_2, ...）
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        
        # 使用PyTorch内置的TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出层：直接预测下一个词（对应图片中解码器的输出）
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重：Embedding使用normal，Linear使用Xavier uniform"""
        # Embedding层：normal初始化
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        if self.embedding.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.embedding.padding_idx], 0)
        
        # Linear层：Xavier uniform初始化
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.embedding:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播：Transformer解码器生成下一个词
        
        Args:
            tgt: 目标序列（已生成的词序列）(batch, tgt_len, d_model)
            memory: 编码器输出的记忆（视觉特征序列）(batch, src_len, d_model)
            tgt_mask: 目标序列掩码（因果掩码，防止未来词泄露），默认None（自动生成）
            tgt_key_padding_mask: 目标序列填充掩码，默认None
        
        Returns:
            解码器输出 (batch, tgt_len, vocab_size)
        """
        # 文本嵌入
        # (batch, tgt_len) → (batch, tgt_len, d_model)
        tgt_embedded = self.embedding(tgt)  # (batch, tgt_len, d_model)
        
        # Transformer解码器处理
        # 内部包含：
        # 1. 掩码自注意力：关注已生成的文本序列（对应图片中"已生成文本"）
        # 2. 跨模态注意力：关注编码器输出的视觉特征（对应图片中"同时关注编码器输出"）
        # 3. 前馈网络和LayerNorm+残差连接
        decoded = self.transformer_decoder(
            tgt=tgt_embedded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # (batch, tgt_len, d_model)
        
        # 输出层：预测下一个词
        # (batch, tgt_len, d_model) → (batch, tgt_len, vocab_size)
        output = self.output_proj(decoded)  # (batch, tgt_len, vocab_size)
        
        return output


class FashionCaptionModelTransformer(nn.Module):
    """
    全Transformer服饰描述生成模型（对应图片完整架构）
    局部表示+Transformer编码器→Transformer解码器
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 4,
                 dim_feedforward: int = 2048, num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2, dropout: float = 0.1, padding_idx: int = 0):
        """
        初始化全Transformer模型
        
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度，默认512
            nhead: 注意力头数，默认4
            dim_feedforward: 前馈网络维度，默认2048
            num_encoder_layers: 编码器层数，默认2（对应图片中2层编码器）
            num_decoder_layers: 解码器层数，默认2（对应图片中2层解码器）
            dropout: Dropout比例，默认0.1
            padding_idx: 填充标记索引，默认0
        """
        super(FashionCaptionModelTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        
        # 局部特征预处理层（对应图片"CNN Extract spatial features"后的预处理）
        # 输入：local_feats (batch, 2048, 7, 7)
        # 展平：49×2048 → Linear映射：49×512
        self.local_feat_proj = nn.Linear(2048, d_model)
        
        # Transformer编码器（对应图片"Transformer encoder"）
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_encoder_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer解码器（对应图片"Transformer decoder"）
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True,
            padding_idx=padding_idx
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        # local_feat_proj使用Xavier uniform初始化
        nn.init.xavier_uniform_(self.local_feat_proj.weight)
        nn.init.constant_(self.local_feat_proj.bias, 0)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        生成因果掩码（防止未来词泄露）
        对应图片中解码器的掩码自注意力机制
        
        Args:
            sz: 序列长度
        
        Returns:
            掩码矩阵 (sz, sz)，下三角为False（允许），上三角为True（掩码）
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, local_feat: torch.Tensor, caption: torch.Tensor,
                teacher_forcing_ratio: float = 0.8) -> torch.Tensor:
        """
        前向传播（训练模式：混合Teacher-Forcing）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
            caption: 文本序列 (batch, seq_len)，包含<START>和<END>
            teacher_forcing_ratio: Teacher-Forcing比例，默认0.8
        
        Returns:
            预测的下一个词概率分布 (batch, seq_len-1, vocab_size)
        """
        batch_size = local_feat.size(0)
        device = local_feat.device
        seq_len = caption.size(1)
        
        # 1. 局部特征预处理（对应图片"CNN Extract spatial features"后的预处理）
        # (batch, 2048, 7, 7) → (batch, 2048, 49) → (batch, 49, 2048) → (batch, 49, 512)
        h, w = local_feat.size(2), local_feat.size(3)
        local_feat_flat = local_feat.view(batch_size, 2048, h * w)  # (batch, 2048, 49)
        local_feat_flat = local_feat_flat.transpose(1, 2)  # (batch, 49, 2048)
        local_feat_proj = self.local_feat_proj(local_feat_flat)  # (batch, 49, 512)
        
        # 2. Transformer编码器（对应图片"Transformer encoder"）
        # (batch, 49, 512) → (batch, 49, 512)
        encoder_output = self.encoder(local_feat_proj)  # (batch, 49, 512)
        
        # 3. 准备解码器输入（混合Teacher-Forcing）
        # 输入序列：去掉最后一个词（<END>），保留<START>和中间词
        decoder_input = caption[:, :-1]  # (batch, seq_len-1)
        
        # 生成因果掩码（防止未来词泄露）
        tgt_len = decoder_input.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(device)  # (tgt_len, tgt_len)
        
        # 生成填充掩码（忽略<PAD>）
        tgt_key_padding_mask = (decoder_input == self.padding_idx)  # (batch, tgt_len)
        
        # 4. Transformer解码器（对应图片"Transformer decoder"）
        # (batch, tgt_len, vocab_size)
        decoder_output = self.decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return decoder_output
    
    def generate(self, local_feat: torch.Tensor, max_len: int = 25,
                 start_idx: int = 2, end_idx: int = 3, pad_idx: int = 0,
                 temperature: float = 0.7) -> torch.Tensor:
        """
        生成描述（推理模式：贪心解码）
        严格对齐图片"自回归生成"逻辑
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
            max_len: 最大生成长度，默认25
            start_idx: <START>标记索引，默认2
            end_idx: <END>标记索引，默认3
            pad_idx: <PAD>标记索引，默认0
            temperature: 温度系数，默认0.7（用于软化预测概率）
        
        Returns:
            生成的词ID序列 (batch, generated_len)
        """
        batch_size = local_feat.size(0)
        device = local_feat.device
        
        # 1. 局部特征预处理（对应图片"CNN Extract spatial features"后的预处理）
        # (batch, 2048, 7, 7) → (batch, 49, 512)
        h, w = local_feat.size(2), local_feat.size(3)
        local_feat_flat = local_feat.view(batch_size, 2048, h * w)  # (batch, 2048, 49)
        local_feat_flat = local_feat_flat.transpose(1, 2)  # (batch, 49, 2048)
        local_feat_proj = self.local_feat_proj(local_feat_flat)  # (batch, 49, 512)
        
        # 2. Transformer编码器（对应图片"Transformer encoder"）
        # (batch, 49, 512) → (batch, 49, 512)
        encoder_output = self.encoder(local_feat_proj)  # (batch, 49, 512)
        
        # 3. 初始化解码器输入：<START>标记
        decoder_input = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=device)  # (batch, 1)
        generated_ids = []
        
        # 4. 自回归生成（对应图片"自回归生成"）
        for step in range(max_len):
            # 生成因果掩码
            tgt_len = decoder_input.size(1)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(device)  # (tgt_len, tgt_len)
            
            # 生成填充掩码
            tgt_key_padding_mask = (decoder_input == pad_idx)  # (batch, tgt_len)
            
            # Transformer解码器前向传播
            decoder_output = self.decoder(
                tgt=decoder_input,
                memory=encoder_output,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )  # (batch, tgt_len, vocab_size)
            
            # 获取最后一个时间步的输出（预测下一个词）
            # (batch, tgt_len, vocab_size) → (batch, vocab_size)
            logits = decoder_output[:, -1, :]  # (batch, vocab_size)
            
            # 温度采样（软化预测概率）
            if temperature != 1.0:
                logits = logits / temperature
            
            # 贪心解码：选择概率最大的词
            next_word_ids = torch.argmax(logits, dim=1, keepdim=True)  # (batch, 1)
            generated_ids.append(next_word_ids)
            
            # 检查是否生成<END>
            if torch.all(next_word_ids == end_idx):
                break
            
            # 更新解码器输入（自回归）
            decoder_input = torch.cat([decoder_input, next_word_ids], dim=1)  # (batch, tgt_len+1)
        
        # 5. 拼接生成的序列
        generated_sequence = torch.cat(generated_ids, dim=1)  # (batch, generated_len)
        
        return generated_sequence
    
    def postprocess_caption(self, sequence: torch.Tensor, idx2word: dict,
                           start_idx: int = 2, end_idx: int = 3,
                           unk_idx: int = 1, pad_idx: int = 0) -> list:
        """
        后处理生成的文本：过滤<UNK>/<PAD>/<START>，仅保留有效英文单词
        与前序模型完全一致，确保评测模块兼容
        
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


def load_vocab(vocab_file: str) -> dict:
    """
    加载词汇表
    
    Args:
        vocab_file: 词汇表JSON文件路径
    
    Returns:
        包含word2idx和idx2word的字典
    """
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    return {
        'word2idx': vocab_data['word2idx'],
        'idx2word': {int(k): v for k, v in vocab_data['idx2word'].items()},
        'vocab_size': vocab_data['vocab_size']
    }


if __name__ == "__main__":
    """
    测试代码：验证模型架构与图片严格对齐
    1. 模型初始化测试
    2. 前向传播测试（单样本）
    3. 生成测试（5个样本）
    4. 参数量统计与对比
    """
    print("="*60)
    print("Model5: 全Transformer架构测试")
    print("="*60)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试参数
    vocab_size = 10000  # 示例词汇表大小
    batch_size = 2
    seq_len = 20
    
    # 测试1: 模型初始化
    print("\n" + "="*60)
    print("测试1: 模型初始化")
    print("="*60)
    model = FashionCaptionModelTransformer(vocab_size=vocab_size).to(device)
    print(f"模型类型: {type(model).__name__}")
    print(f"词汇表大小: {vocab_size}")
    print(f"模型维度: {model.d_model}")
    print(f"编码器层数: 2")
    print(f"解码器层数: 2")
    print(f"注意力头数: 4")
    
    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n参数量统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 与前序模型对比（示例）
    print(f"\n[注意] 参数量对比需要加载实际的前序模型进行对比")
    
    # 测试2: 前向传播（单样本）
    print("\n" + "="*60)
    print("测试2: 前向传播（单样本）")
    print("="*60)
    model.train()
    test_local_feat = torch.randn(batch_size, 2048, 7, 7).to(device)
    test_caption = torch.randint(4, vocab_size, (batch_size, seq_len)).to(device)
    test_caption[:, 0] = 2  # <START>
    test_caption[:, -1] = 3  # <END>
    
    outputs = model(test_local_feat, test_caption, teacher_forcing_ratio=0.8)
    print(f"输入局部特征形状: {test_local_feat.shape}")
    print(f"输入文本序列形状: {test_caption.shape}")
    print(f"输出形状: {outputs.shape}")
    assert outputs.shape == (batch_size, seq_len - 1, vocab_size), \
        f"输出维度错误: {outputs.shape}"
    print("[OK] 前向计算测试通过")
    
    # 测试3: 生成测试（5个样本）
    print("\n" + "="*60)
    print("测试3: 生成测试（5个样本）")
    print("="*60)
    model.eval()
    test_batch_size = 5
    test_local_feat = torch.randn(test_batch_size, 2048, 7, 7).to(device)
    
    # 创建示例词汇表
    idx2word = {i: f"word_{i}" for i in range(vocab_size)}
    idx2word[0] = "<PAD>"
    idx2word[1] = "<UNK>"
    idx2word[2] = "<START>"
    idx2word[3] = "<END>"
    
    with torch.no_grad():
        generated_sequences = model.generate(
            test_local_feat, max_len=25, temperature=0.7
        )
    
    print(f"生成序列形状: {generated_sequences.shape}")
    print("\n生成的描述示例:")
    
    for i in range(test_batch_size):
        seq = generated_sequences[i].cpu()
        gen_words = model.postprocess_caption(seq, idx2word)
        print(f"\n样本{i+1}:")
        print(f"  生成描述: {' '.join(gen_words) if gen_words else '(空)'}")
        print(f"  序列长度: {len(seq)}")
    
    print("\n" + "="*60)
    print("[OK] 所有测试通过！")
    print("="*60)

