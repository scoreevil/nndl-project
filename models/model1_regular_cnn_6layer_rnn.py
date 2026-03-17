"""
常规CNN编码器 + 6层基础RNN解码器服饰描述生成模型
Model1: Regular CNN Encoder + 6-Layer Basic RNN Decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import Tuple, Optional


class RegularCNNEncoder(nn.Module):
    """
    常规CNN特征编码器
    输入：局部特征 (batch, 2048, 7, 7)
    输出：512维视觉特征向量 (batch, 512)
    """
    
    def __init__(self):
        super(RegularCNNEncoder, self).__init__()
        
        # 第一层卷积：2048 → 1024
        # Conv2d(2048, 1024, kernel_size=3, padding=1) → BatchNorm2d(1024) → ReLU → MaxPool2d(kernel_size=2, stride=2)
        # 输入: (batch, 2048, 7, 7)
        # 输出: (batch, 1024, 3, 3)  [7/2=3.5向下取整=3]
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二层卷积：1024 → 512
        # Conv2d(1024, 512, kernel_size=3, padding=1) → BatchNorm2d(512) → ReLU → AvgPool2d(kernel_size=3, stride=1)
        # 输入: (batch, 1024, 3, 3)
        # 输出: (batch, 512, 1, 1)  [3-3+1=1]
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
    
    def forward(self, local_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
        
        Returns:
            视觉特征向量 (batch, 512)
        """
        # 第一层卷积
        # (batch, 2048, 7, 7) → (batch, 1024, 7, 7) → (batch, 1024, 3, 3)
        x = self.conv1(local_feat)
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
        
        return x


class BasicRNNDecoder(nn.Module):
    """
    6层基础RNN解码器
    无残差/门控/额外增强，仅使用PyTorch原生torch.nn.RNN
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512):
        """
        初始化RNN解码器
        
        Args:
            vocab_size: 词典大小
            embed_dim: 词嵌入维度，默认256
            hidden_dim: RNN隐藏状态维度，默认512
        """
        super(BasicRNNDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # 词嵌入层：Embedding(vocab_size, 256, padding_idx=0)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 6层基础RNN
        # input_size=256+512=768（词嵌入+CNN输出特征）
        # hidden_size=512, num_layers=6, batch_first=True, bidirectional=False
        self.rnn = nn.RNN(
            input_size=embed_dim + hidden_dim,  # 256 + 512 = 768
            hidden_size=hidden_dim,  # 512
            num_layers=6,  # 6层
            batch_first=True,
            bidirectional=False
        )
        
        # 隐藏状态初始化：CNN输出的512维特征 → Linear(512, 512) → tanh
        self.hidden_init = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出层：Linear(512, vocab_size)，无激活/归一化
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        # 词嵌入层：Xavier_uniform
        nn.init.xavier_uniform_(self.embedding.weight)
        # padding_idx=0的权重设为0
        self.embedding.weight.data[0].fill_(0)
        
        # RNN权重：默认PyTorch初始化（均匀分布）
        # 这里不手动初始化，使用PyTorch默认初始化
        
        # 隐藏状态初始化层：Xavier_uniform
        nn.init.xavier_uniform_(self.hidden_init.weight)
        nn.init.constant_(self.hidden_init.bias, 0)
        
        # 输出层：Xavier_uniform
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)
    
    def init_hidden(self, cnn_feat: torch.Tensor) -> torch.Tensor:
        """
        初始化RNN隐藏状态
        CNN输出的512维特征 → Linear(512, 512) → tanh → 作为6层RNN的初始隐藏状态
        
        Args:
            cnn_feat: CNN编码器输出的特征 (batch, 512)
        
        Returns:
            初始隐藏状态 (6, batch, 512) - 6层RNN，每层都需要初始状态
        """
        # (batch, 512) → (batch, 512)
        h0 = self.hidden_init(cnn_feat)
        h0 = torch.tanh(h0)
        
        # 扩展为6层RNN的初始状态
        # (batch, 512) → (1, batch, 512) → (6, batch, 512)
        h0 = h0.unsqueeze(0)  # (1, batch, 512)
        h0 = h0.repeat(6, 1, 1)  # (6, batch, 512)
        
        return h0
    
    def forward(self, word_ids: torch.Tensor, cnn_feat: torch.Tensor, 
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（训练模式：Teacher-Forcing）
        
        Args:
            word_ids: 词ID序列 (batch, seq_len)
            cnn_feat: CNN编码器输出的特征 (batch, 512)
            hidden: 初始隐藏状态 (6, batch, 512)，如果为None则自动初始化
        
        Returns:
            outputs: RNN输出 (batch, seq_len, vocab_size)
            hidden: 最终隐藏状态 (6, batch, 512)
        """
        batch_size, seq_len = word_ids.size()
        
        # 初始化隐藏状态
        if hidden is None:
            hidden = self.init_hidden(cnn_feat)  # (6, batch, 512)
        
        # 词嵌入
        # (batch, seq_len) → (batch, seq_len, 256)
        word_embeds = self.embedding(word_ids)
        
        # 将CNN特征与词嵌入拼接
        # CNN特征扩展: (batch, 512) → (batch, 1, 512) → (batch, seq_len, 512)
        cnn_feat_expanded = cnn_feat.unsqueeze(1).expand(batch_size, seq_len, self.hidden_dim)
        
        # 拼接: (batch, seq_len, 256) + (batch, seq_len, 512) → (batch, seq_len, 768)
        rnn_input = torch.cat([word_embeds, cnn_feat_expanded], dim=2)
        
        # 6层RNN前向传播
        # 输入: (batch, seq_len, 768), 隐藏状态: (6, batch, 512)
        # 输出: (batch, seq_len, 512), 隐藏状态: (6, batch, 512)
        rnn_output, hidden = self.rnn(rnn_input, hidden)
        
        # 输出投影
        # (batch, seq_len, 512) → (batch, seq_len, vocab_size)
        outputs = self.output_proj(rnn_output)
        
        return outputs, hidden


class FashionCaptionModel(nn.Module):
    """
    服饰描述生成模型：常规CNN编码器 + 6层基础RNN解码器
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512):
        """
        初始化模型
        
        Args:
            vocab_size: 词典大小
            embed_dim: 词嵌入维度，默认256
            hidden_dim: RNN隐藏状态维度，默认512
        """
        super(FashionCaptionModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # CNN编码器
        self.cnn_encoder = RegularCNNEncoder()
        
        # RNN解码器
        self.rnn_decoder = BasicRNNDecoder(vocab_size, embed_dim, hidden_dim)
        
        # 打印模型参数量
        self._print_model_info()
    
    def _print_model_info(self):
        """打印模型各部分参数量"""
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        cnn_params = count_parameters(self.cnn_encoder)
        embed_params = count_parameters(self.rnn_decoder.embedding)
        rnn_params = count_parameters(self.rnn_decoder.rnn)
        hidden_init_params = count_parameters(self.rnn_decoder.hidden_init)
        output_params = count_parameters(self.rnn_decoder.output_proj)
        total_params = count_parameters(self)
        
        print("="*60)
        print("模型参数量统计")
        print("="*60)
        print(f"CNN编码器: {cnn_params:,} 参数")
        print(f"词嵌入层: {embed_params:,} 参数")
        print(f"6层RNN: {rnn_params:,} 参数")
        print(f"隐藏状态初始化层: {hidden_init_params:,} 参数")
        print(f"输出层: {output_params:,} 参数")
        print(f"总参数量: {total_params:,} 参数")
        print("="*60)
    
    def forward(self, local_feat: torch.Tensor, caption: torch.Tensor) -> torch.Tensor:
        """
        前向传播（训练模式：Teacher-Forcing）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
            caption: 文本序列 (batch, seq_len)，包含<START>和<END>
        
        Returns:
            预测的词汇分布 (batch, seq_len, vocab_size)
        """
        # CNN编码
        cnn_feat = self.cnn_encoder(local_feat)  # (batch, 512)
        
        # RNN解码（Teacher-Forcing：输入是ground truth）
        # 输入序列去掉最后一个词（<END>），作为输入
        # 目标序列去掉第一个词（<START>），作为目标
        input_ids = caption[:, :-1]  # (batch, seq_len-1)
        outputs, _ = self.rnn_decoder(input_ids, cnn_feat)  # (batch, seq_len-1, vocab_size)
        
        return outputs
    
    def generate(self, local_feat: torch.Tensor, max_len: int = 20, 
                 start_idx: int = 2, end_idx: int = 3) -> torch.Tensor:
        """
        生成描述（推理模式：贪心解码）
        
        Args:
            local_feat: 局部特征 (batch, 2048, 7, 7)
            max_len: 最大生成长度，默认20
            start_idx: <START>标记索引，默认2
            end_idx: <END>标记索引，默认3
        
        Returns:
            生成的词ID序列 (batch, generated_len)
        """
        batch_size = local_feat.size(0)
        device = local_feat.device
        
        # CNN编码
        cnn_feat = self.cnn_encoder(local_feat)  # (batch, 512)
        
        # 初始化隐藏状态
        hidden = self.rnn_decoder.init_hidden(cnn_feat)  # (6, batch, 512)
        
        # 初始化输入：<START>标记
        input_ids = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=device)
        generated_ids = []
        
        # 逐词生成
        for _ in range(max_len):
            # 前向传播
            outputs, hidden = self.rnn_decoder(input_ids, cnn_feat, hidden)
            
            # 获取最后一个时间步的输出
            # (batch, 1, vocab_size) → (batch, vocab_size)
            logits = outputs[:, -1, :]
            
            # 贪心解码：选择概率最大的词
            next_word_ids = torch.argmax(logits, dim=1, keepdim=True)  # (batch, 1)
            generated_ids.append(next_word_ids)
            
            # 检查是否生成<END>
            if torch.all(next_word_ids == end_idx):
                break
            
            # 更新输入
            input_ids = next_word_ids
        
        # 拼接生成的序列
        generated_sequence = torch.cat(generated_ids, dim=1)  # (batch, generated_len)
        
        return generated_sequence


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
    print("模型测试")
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
    print("创建模型")
    print("="*60)
    model = FashionCaptionModel(vocab_size=vocab_size)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\n[OK] 模型已移动到设备: {device}")
    
    # 测试1: CNN编码器维度验证
    print("\n" + "="*60)
    print("测试1: CNN编码器维度验证")
    print("="*60)
    batch_size = 4
    local_feat = torch.randn(batch_size, 2048, 7, 7).to(device)
    print(f"输入局部特征形状: {local_feat.shape}")
    
    with torch.no_grad():
        cnn_feat = model.cnn_encoder(local_feat)
    print(f"CNN输出特征形状: {cnn_feat.shape}")
    assert cnn_feat.shape == (batch_size, 512), f"CNN输出维度错误: {cnn_feat.shape}"
    print("[OK] CNN编码器维度验证通过")
    
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
        seq = generated_sequences[i].cpu().tolist()
        # 转换为单词（如果词典可用）
        words = [idx2word.get(idx, f"<{idx}>") for idx in seq]
        print(f"  样本{i+1}: {' '.join(words)}")
    
    print("\n" + "="*60)
    print("[OK] 所有测试通过！")
    print("="*60)

