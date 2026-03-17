"""
训练Model1b: 常规CNN编码器 + 2层LSTM解码器
改进的训练策略：混合Teacher-Forcing、学习率调度、<END>权重提升
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from models.model1b_cnn_2layer_lstm import FashionCaptionModelLSTM, load_vocab
from utils.dataset import get_dataloader
from utils.data_loader import load_and_validate_dataset
from utils.text_processor import TextProcessor
from torch.utils.data import Dataset
import numpy as np


class WeightedCrossEntropyLoss(nn.Module):
    """
    加权交叉熵损失函数
    给<END>标记的损失权重提升1.5倍（避免提前截断）
    """
    
    def __init__(self, vocab_size: int, end_idx: int = 3, end_weight: float = 1.5, ignore_index: int = 0):
        """
        Args:
            vocab_size: 词典大小
            end_idx: <END>标记索引
            end_weight: <END>标记的权重倍数
            ignore_index: 忽略的索引（<PAD>）
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.vocab_size = vocab_size
        self.end_idx = end_idx
        self.end_weight = end_weight
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算加权交叉熵损失
        
        Args:
            inputs: 预测分布 (batch * seq_len, vocab_size)
            targets: 目标索引 (batch * seq_len,)
        
        Returns:
            加权损失值
        """
        # 标准交叉熵损失（reduction='none'）
        loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        
        # 对<END>标记的损失应用额外权重（1.5倍）
        end_mask = (targets == self.end_idx)
        loss = loss * (1.0 + (self.end_weight - 1.0) * end_mask.float())
        
        # 平均（忽略<PAD>）
        valid_mask = (targets != self.ignore_index)
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        return loss


class ExpandedFashionCaptionDataset(Dataset):
    """
    扩展的Fashion数据集：支持特征索引映射
    用于处理扩展后的数据（每个图像的所有描述）
    """
    
    def __init__(self, feature_file: str, caption_sequences: torch.Tensor, feature_indices: List[int]):
        """
        初始化数据集
        
        Args:
            feature_file: npz特征文件路径
            caption_sequences: 文本序列张量，形状为(N, max_len)
            feature_indices: 特征索引列表，长度等于caption_sequences的数量
        """
        self.feature_file = feature_file
        self.caption_sequences = caption_sequences
        self.feature_indices = feature_indices
        
        # 加载npz文件
        data = np.load(feature_file, allow_pickle=True)
        self.global_feats = data['global_feats']
        self.local_feats = data['local_feats']
        
        # 验证数据一致性
        num_sequences = caption_sequences.shape[0]
        num_indices = len(feature_indices)
        
        if num_sequences != num_indices:
            raise ValueError(
                f"文本序列数量({num_sequences})与特征索引数量({num_indices})不匹配！"
            )
        
        # 验证特征索引有效性
        max_feat_idx = max(feature_indices) if feature_indices else -1
        if max_feat_idx >= self.local_feats.shape[0]:
            raise ValueError(
                f"特征索引超出范围: 最大索引{max_feat_idx}，特征数量{self.local_feats.shape[0]}"
            )
        
        self.length = num_sequences
        
        print(f"成功加载扩展数据集: {feature_file}")
        print(f"  样本数量: {self.length}")
        print(f"  特征数量: {self.local_feats.shape[0]}")
        print(f"  文本序列形状: {self.caption_sequences.shape}")
    
    def __len__(self) -> int:
        """返回样本总数"""
        return self.length
    
    def __getitem__(self, idx: int) -> dict:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            包含以下键的字典：
            - "global_feat": 全局特征 (torch.float32, 形状(2048,))
            - "local_feat": 局部特征 (torch.float32, 形状(2048, 7, 7))
            - "caption": 文本序列 (torch.long, 形状(max_len,))
        """
        # 使用特征索引获取特征
        feat_idx = self.feature_indices[idx]
        global_feat = torch.from_numpy(self.global_feats[feat_idx]).float()  # (2048,)
        local_feat = torch.from_numpy(self.local_feats[feat_idx]).float()   # (2048, 7, 7)
        
        # 获取文本序列
        caption = self.caption_sequences[idx].long()  # (max_len,)
        
        return {
            "global_feat": global_feat,
            "local_feat": local_feat,
            "caption": caption
        }


def expand_captions_and_sequences(data_list: List[Dict], feature_indices: List[int], 
                                   processor: TextProcessor, max_len: int = 20) -> Tuple[List[int], torch.Tensor]:
    """
    扩展数据列表和序列：每个样本使用所有描述（而非仅第一个），提升数据多样性
    
    Args:
        data_list: 原始数据列表
        feature_indices: 每个数据项对应的特征索引
        processor: 文本处理器
        max_len: 最大序列长度
    
    Returns:
        expanded_indices: 扩展后的特征索引列表（每个图像的特征对应多个描述）
        expanded_sequences: 扩展后的文本序列张量
    """
    expanded_indices = []
    expanded_sequences = []
    
    for item, feat_idx in zip(data_list, feature_indices):
        captions = item.get('captions', [])
        if len(captions) == 0:
            continue
        
        # 为每个描述创建一个样本
        for caption in captions:
            if isinstance(caption, str):
                # 处理文本序列
                sequence = processor.text_to_sequence(caption, max_len=max_len)
                expanded_sequences.append(sequence)
                expanded_indices.append(feat_idx)  # 使用相同的特征索引
    
    expanded_sequences_tensor = torch.tensor(expanded_sequences, dtype=torch.long)
    return expanded_indices, expanded_sequences_tensor


def train_epoch(model, dataloader, criterion, optimizer, device, max_norm=1.0, teacher_forcing_ratio=0.8):
    """训练一个epoch（混合Teacher-Forcing）"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 移动到设备
        local_feat = batch['local_feat'].to(device)  # (batch, 2048, 7, 7)
        caption = batch['caption'].to(device)  # (batch, max_len)
        
        # 前向传播（混合Teacher-Forcing：80% Teacher-Forcing，20%自回归）
        outputs = model(local_feat, caption, teacher_forcing_ratio=teacher_forcing_ratio)
        
        # 准备目标：去掉第一个词（<START>）
        targets = caption[:, 1:]  # (batch, seq_len-1)
        
        # 计算加权损失（<END>标记权重1.5倍）
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（仅兜底，LSTM梯度更稳定）
        clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 打印进度
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            local_feat = batch['local_feat'].to(device)
            caption = batch['caption'].to(device)
            
            # 验证时使用100% Teacher-Forcing
            outputs = model(local_feat, caption, teacher_forcing_ratio=1.0)
            targets = caption[:, 1:]
            
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """主训练函数"""
    print("="*60)
    print("训练Model1b: 常规CNN编码器 + 2层LSTM解码器")
    print("="*60)
    
    # 配置参数
    vocab_file = PROJECT_ROOT / "dataset" / "vocab.json"
    train_feature_file = PROJECT_ROOT / "dataset" / "features" / "train_features.npz"
    val_feature_file = PROJECT_ROOT / "dataset" / "features" / "val_features.npz"
    batch_size = 8
    num_epochs = 20
    learning_rate = 1e-4
    weight_decay = 1e-5
    max_norm = 1.0  # 梯度裁剪
    max_len = 20
    teacher_forcing_ratio = 0.8  # 80% Teacher-Forcing，20%自回归
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载词典
    print("\n加载词典...")
    vocab_info = load_vocab(str(vocab_file))
    vocab_size = vocab_info['vocab_size']
    idx2word = vocab_info['idx2word']
    print(f"词典大小: {vocab_size}")
    
    # 加载数据集
    print("\n加载数据集...")
    train_data, val_data, test_data = load_and_validate_dataset(
        captions_file=str(PROJECT_ROOT / "dataset" / "captions.json"),
        images_dir=str(PROJECT_ROOT / "dataset" / "images"),
        random_seed=42
    )
    
    # 文本预处理
    print("\n文本预处理...")
    processor = TextProcessor(min_freq=3)
    processor.load_vocab(str(vocab_file))
    
    # 处理验证集文本序列（不扩展）
    val_sequences = processor.batch_process(val_data, max_len=max_len)
    
    # 扩展训练集：每个样本使用所有描述（提升数据多样性）
    print("\n扩展训练集（使用所有描述）...")
    train_feature_indices = list(range(len(train_data)))  # 原始特征索引
    train_expanded_indices, train_sequences = expand_captions_and_sequences(
        train_data, train_feature_indices, processor, max_len=max_len
    )
    print(f"原始训练集: {len(train_data)}条")
    print(f"扩展后训练集: {len(train_expanded_indices)}条（每个图像的所有描述）")
    
    print(f"训练集序列形状: {train_sequences.shape}")
    print(f"验证集序列形状: {val_sequences.shape}")
    
    # 创建Dataset和DataLoader
    print("\n创建DataLoader...")
    # 训练集使用扩展数据集（支持特征索引映射）
    train_dataset = ExpandedFashionCaptionDataset(
        str(train_feature_file), 
        train_sequences, 
        train_expanded_indices
    )
    # 验证集使用标准数据集
    from utils.dataset import FashionCaptionDataset
    val_dataset = FashionCaptionDataset(str(val_feature_file), val_sequences)
    
    train_loader = get_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows上设为0
        pin_memory=False
    )
    
    val_loader = get_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # 创建模型
    print("\n创建模型...")
    model = FashionCaptionModelLSTM(vocab_size=vocab_size)
    model = model.to(device)
    
    # 损失函数（加权交叉熵，<END>标记权重1.5倍）
    criterion = WeightedCrossEntropyLoss(vocab_size=vocab_size, end_idx=3, end_weight=1.5, ignore_index=0)
    
    # 优化器（Adam，weight_decay=1e-5抑制过拟合）
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 学习率调度器（ReduceLROnPlateau，监测val_loss）
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3, verbose=True)
    
    # 训练循环
    print("\n" + "="*60)
    print("开始训练（20轮，混合Teacher-Forcing，学习率调度）")
    print("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # 训练（混合Teacher-Forcing：80% Teacher-Forcing，20%自回归）
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, 
                                 max_norm=max_norm, teacher_forcing_ratio=teacher_forcing_ratio)
        print(f"训练损失: {train_loss:.4f}")
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        print(f"验证损失: {val_loss:.4f}")
        
        # 学习率调度（监测val_loss）
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = PROJECT_ROOT / "models" / "model1b_checkpoint.pt"
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'optimizer_state_dict': optimizer.state_dict(),
            }, str(model_save_path))
            print(f"[OK] 保存最佳模型（验证损失: {val_loss:.4f}）")
    
    # 生成示例
    print("\n" + "="*60)
    print("生成示例（温度采样temperature=0.7）")
    print("="*60)
    model.eval()
    
    # 从验证集取5个样本
    sample_batch = next(iter(val_loader))
    sample_local_feat = sample_batch['local_feat'][:5].to(device)
    sample_caption = sample_batch['caption'][:5]
    
    with torch.no_grad():
        generated_sequences = model.generate(sample_local_feat, max_len=25, temperature=0.7)
    
    print("\n生成的描述（后处理过滤特殊标记）:")
    for i in range(5):
        # 真实描述
        true_seq = sample_caption[i].tolist()
        true_words = model.postprocess_caption(torch.tensor(true_seq), idx2word)
        
        # 生成描述
        gen_seq = generated_sequences[i].cpu()
        gen_words = model.postprocess_caption(gen_seq, idx2word)
        
        print(f"\n样本{i+1}:")
        print(f"  真实: {' '.join(true_words) if true_words else '(空)'}")
        print(f"  生成: {' '.join(gen_words) if gen_words else '(空)'}")
    
    print("\n" + "="*60)
    print("[OK] 训练完成！")
    print("="*60)
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型已保存到: models/model1b_checkpoint.pt")


if __name__ == "__main__":
    main()

