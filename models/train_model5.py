"""
训练Model5: 全Transformer架构（局部表示+Transformer编码器→Transformer解码器）
复用前序模型的训练框架：混合Teacher-Forcing、学习率调度、损失函数
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
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from models.model5_full_transformer import FashionCaptionModelTransformer, load_vocab
from utils.dataset import get_dataloader, ExpandedFashionCaptionDataset
from utils.data_loader import load_and_validate_dataset
from utils.text_processor import TextProcessor
import random


class WeightedCrossEntropyLoss(nn.Module):
    """
    加权交叉熵损失函数
    给<END>标记的损失权重提升1.5倍（避免提前截断）
    与前序模型完全一致
    """
    
    def __init__(self, vocab_size: int, end_idx: int = 3, end_weight: float = 1.5, ignore_index: int = 0):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.vocab_size = vocab_size
        self.end_idx = end_idx
        self.end_weight = end_weight
        self.ignore_index = ignore_index
        
        # 使用register_buffer确保权重会自动移动到正确的设备
        weights = torch.ones(vocab_size)
        weights[end_idx] = end_weight
        self.register_buffer('weights', weights)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 确保权重在正确的设备上
        weights = self.weights.to(inputs.device)
        
        loss = F.cross_entropy(inputs, targets, weight=weights, ignore_index=self.ignore_index, reduction='none')
        end_mask = (targets == self.end_idx)
        loss = loss * (1.0 + (self.end_weight - 1.0) * end_mask.float())
        
        valid_mask = (targets != self.ignore_index)
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
        return loss


def train_epoch(model, dataloader, criterion, optimizer, device, max_norm=1.0, teacher_forcing_ratio=0.8):
    """
    训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        max_norm: 梯度裁剪的最大范数
        teacher_forcing_ratio: Teacher-Forcing比例
    
    Returns:
        平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        local_feat = batch['local_feat'].to(device)
        caption = batch['caption'].to(device)
        
        # 前向传播（混合Teacher-Forcing）
        outputs = model(local_feat, caption, teacher_forcing_ratio=teacher_forcing_ratio)
        
        # 计算损失
        targets = caption[:, 1:]  # 去掉<START>标记
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        clip_grad_norm_(model.parameters(), max_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_epoch(model, dataloader, criterion, device):
    """
    验证一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        平均损失
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            local_feat = batch['local_feat'].to(device)
            caption = batch['caption'].to(device)
            
            # 前向传播（使用Teacher-Forcing）
            outputs = model(local_feat, caption, teacher_forcing_ratio=1.0)
            
            # 计算损失
            targets = caption[:, 1:]  # 去掉<START>标记
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    """
    主训练函数
    复用前序模型的训练框架，仅修改模型调用逻辑
    """
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 路径配置
    PROJECT_ROOT = Path(__file__).parent.parent
    VOCAB_FILE = PROJECT_ROOT / "dataset" / "vocab.json"
    TRAIN_FEATURES = PROJECT_ROOT / "dataset" / "features" / "train_features.npz"
    VAL_FEATURES = PROJECT_ROOT / "dataset" / "features" / "val_features.npz"
    CAPTIONS_FILE = PROJECT_ROOT / "dataset" / "captions.json"
    IMAGES_DIR = PROJECT_ROOT / "dataset" / "images"
    CHECKPOINT_DIR = PROJECT_ROOT / "run" / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 训练配置
    BATCH_SIZE = 8
    MAX_LEN = 25
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 1.0
    TEACHER_FORCING_RATIO = 0.8
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载词汇表
    print("\n加载词汇表...")
    vocab_data = load_vocab(str(VOCAB_FILE))
    vocab_size = vocab_data['vocab_size']
    idx2word = vocab_data['idx2word']
    print(f"词汇表大小: {vocab_size}")
    
    # 加载数据
    print("\n加载数据...")
    train_data, val_data, test_data = load_and_validate_dataset(
        captions_file=str(CAPTIONS_FILE),
        images_dir=str(IMAGES_DIR),
        random_seed=42
    )
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    
    # 文本处理器
    print("\n初始化文本处理器...")
    text_processor = TextProcessor()
    text_processor.load_vocab(str(VOCAB_FILE))
    
    # 创建扩展训练数据集（使用所有caption）
    print("\n创建训练数据集（扩展所有caption）...")
    train_dataset = ExpandedFashionCaptionDataset(
        feature_file=str(TRAIN_FEATURES),
        all_captions_data=train_data,
        text_processor=text_processor,
        max_len=MAX_LEN
    )
    
    # 创建验证数据集
    print("\n创建验证数据集...")
    val_sequences = []
    for item in val_data:
        captions = item.get('captions', [])
        if captions:
            sequence = text_processor.text_to_sequence(captions[0], MAX_LEN)
            val_sequences.append(sequence)
    val_sequences = torch.tensor(val_sequences, dtype=torch.long)
    
    from utils.dataset import FashionCaptionDataset
    val_dataset = FashionCaptionDataset(
        feature_file=str(VAL_FEATURES),
        caption_sequences=val_sequences
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 创建模型
    print("\n创建模型...")
    model = FashionCaptionModelTransformer(vocab_size=vocab_size).to(device)
    
    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 损失函数
    criterion = WeightedCrossEntropyLoss(vocab_size=vocab_size).to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=3,
        verbose=True
    )
    
    # 训练循环
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            max_norm=MAX_GRAD_NORM,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO
        )
        
        # 验证
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch + 1} 结果:")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            checkpoint_path = CHECKPOINT_DIR / "model5_checkpoint.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'vocab_size': vocab_size,
            }, checkpoint_path)
            print(f"  [保存最佳模型] 验证损失: {val_loss:.4f}")
    
    print("\n" + "="*60)
    print("训练完成")
    print("="*60)
    print(f"最佳模型: Epoch {best_epoch}, 验证损失: {best_val_loss:.4f}")
    
    # 生成示例
    print("\n" + "="*60)
    print("生成示例")
    print("="*60)
    
    # 加载最佳模型
    checkpoint = torch.load(CHECKPOINT_DIR / "model5_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 选择一个验证样本
    sample_idx = 0
    sample = val_dataset[sample_idx]
    local_feat = sample['local_feat'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated_sequence = model.generate(local_feat, max_len=MAX_LEN, temperature=0.7)
    
    gen_words = model.postprocess_caption(generated_sequence[0].cpu(), idx2word)
    print(f"\n样本 {sample_idx + 1}:")
    print(f"  生成描述: {' '.join(gen_words) if gen_words else '(空)'}")


if __name__ == "__main__":
    main()

