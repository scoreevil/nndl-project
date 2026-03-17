"""
训练Model2（局部表示+自注意力→RNN+注意力）用于新数据集
新数据集特点：描述包含服饰和背景信息
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
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from models.model2_local_selfattn_attention_rnn import FashionCaptionModelAttention, load_vocab
from utils.dataset import get_dataloader, ExpandedFashionCaptionDataset
from utils.load_new_dataset import load_new_dataset_annotations
from utils.text_processor import TextProcessor
import random


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    标签平滑 + 加权交叉熵损失函数
    """
    
    def __init__(self, vocab_size: int, end_idx: int = 3, end_weight: float = 1.5, 
                 ignore_index: int = 0, smoothing: float = 0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.vocab_size = vocab_size
        self.end_idx = end_idx
        self.end_weight = end_weight
        self.ignore_index = ignore_index
        self.smoothing = smoothing
        
        weights = torch.ones(vocab_size)
        weights[end_idx] = end_weight
        self.register_buffer('weights', weights)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.size(0)
        vocab_size = inputs.size(1)
        device = inputs.device
        
        log_probs = F.log_softmax(inputs, dim=1)
        
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (vocab_size - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        valid_mask = (targets != self.ignore_index).unsqueeze(1).expand_as(smooth_targets)
        smooth_targets = smooth_targets * valid_mask.float()
        smooth_targets = smooth_targets / (smooth_targets.sum(dim=1, keepdim=True) + 1e-8)
        
        loss = -torch.sum(smooth_targets * log_probs, dim=1)
        
        weights = self.weights.to(device)
        end_mask = (targets == self.end_idx)
        loss = loss * weights[targets] * (1.0 + (self.end_weight - 1.0) * end_mask.float())
        
        if valid_mask[:, 0].sum() > 0:
            loss = loss[valid_mask[:, 0]].mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss


def train_epoch(model, dataloader, criterion, optimizer, device, max_norm=1.0, teacher_forcing_ratio=0.8):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        local_feat = batch['local_feat'].to(device)
        caption = batch['caption'].to(device)
        
        # 前向传播（混合Teacher-Forcing）
        outputs, _ = model(local_feat, caption, teacher_forcing_ratio=teacher_forcing_ratio, return_attn=False)
        
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
        
        # 打印进度
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            local_feat = batch['local_feat'].to(device)
            caption = batch['caption'].to(device)
            
            # 前向传播（验证时使用100% Teacher-Forcing）
            outputs, _ = model(local_feat, caption, teacher_forcing_ratio=1.0, return_attn=False)
            
            # 计算损失
            targets = caption[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def visualize_attention_heatmap(attn_weights: torch.Tensor, save_dir: Path, epoch: int, sample_idx: int = 0):
    """可视化注意力权重热力图并保存"""
    if attn_weights.dim() == 1:
        attn_weights = attn_weights.unsqueeze(0)
    
    seq_len, num_pixels = attn_weights.shape
    h, w = 7, 7
    
    attn_map = attn_weights.view(seq_len, h, w).cpu().numpy()
    
    fig, axes = plt.subplots(1, min(seq_len, 10), figsize=(min(seq_len, 10) * 2, 2))
    if seq_len == 1:
        axes = [axes]
    
    for i in range(min(seq_len, 10)):
        ax = axes[i]
        im = ax.imshow(attn_map[i], cmap='hot', interpolation='nearest')
        ax.set_title(f'Step {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'Epoch {epoch}, Sample {sample_idx}', fontsize=12)
    plt.tight_layout()
    
    save_path = save_dir / f"attention_epoch_{epoch}_sample_{sample_idx}.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] 注意力热力图已保存到: {save_path}")


def main():
    """主训练函数"""
    print("="*60)
    print("训练Model2（局部表示+自注意力→RNN+注意力）用于新数据集")
    print("新数据集特点：描述包含服饰和背景信息")
    print("="*60)
    
    # 配置参数（新数据集）
    vocab_file = PROJECT_ROOT / "newdataset" / "vocab.json"
    train_feature_file = PROJECT_ROOT / "newdataset" / "features" / "train_features.npz"
    val_feature_file = PROJECT_ROOT / "newdataset" / "features" / "val_features.npz"
    annotations_file = PROJECT_ROOT / "newdataset" / "annotations.json"
    images_dir = PROJECT_ROOT / "newdataset" / "images"
    
    # 模型超参数
    embed_dim = 256      # 词嵌入维度
    hidden_dim = 512    # LSTM隐藏状态维度
    
    # 训练超参数
    batch_size = 16      # 批次大小（新数据集较小，使用较小batch）
    num_epochs = 50      # 训练轮数
    learning_rate = 1e-4  # 学习率
    weight_decay = 1e-5  # 权重衰减
    max_norm = 1.0       # 梯度裁剪
    max_len = 25         # 最大生成长度
    teacher_forcing_ratio = 0.8  # Teacher-Forcing比例
    
    # 学习率调度器参数
    scheduler_patience = 10
    scheduler_factor = 0.7
    scheduler_min_lr = 1e-7
    
    # 早停机制参数
    early_stopping_patience = 15
    early_stopping_min_delta = 1e-5
    
    # 标签平滑参数
    label_smoothing = 0.1
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载词典
    print("\n加载词典...")
    vocab_info = load_vocab(str(vocab_file))
    vocab_size = vocab_info['vocab_size']
    idx2word = vocab_info['idx2word']
    print(f"词典大小: {vocab_size}")
    
    # 加载新数据集
    print("\n加载新数据集...")
    train_data, val_data, test_data = load_new_dataset_annotations(
        annotations_file=str(annotations_file),
        images_dir=str(images_dir),
        random_seed=42
    )
    
    # 文本预处理
    print("\n文本预处理...")
    processor = TextProcessor(min_freq=3)
    processor.load_vocab(str(vocab_file))
    
    # 扩展训练集：每个样本使用所有描述
    print("\n扩展训练集（每个样本使用所有描述）...")
    train_dataset = ExpandedFashionCaptionDataset(
        feature_file=str(train_feature_file),
        all_captions_data=train_data,
        text_processor=processor,
        max_len=max_len
    )
    val_dataset = ExpandedFashionCaptionDataset(
        feature_file=str(val_feature_file),
        all_captions_data=val_data,
        text_processor=processor,
        max_len=max_len
    )
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 创建DataLoader
    print("\n创建DataLoader...")
    train_loader = get_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = get_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # 创建模型（使用model2_local_selfattn_attention_rnn架构）
    print("\n创建模型...")
    print(f"  词嵌入维度: {embed_dim}")
    print(f"  LSTM隐藏状态维度: {hidden_dim}")
    model = FashionCaptionModelAttention(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim
    )
    model = model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")
    
    # 损失函数：标签平滑交叉熵损失
    print(f"\n使用标签平滑交叉熵损失（smoothing={label_smoothing}）")
    criterion = LabelSmoothingCrossEntropyLoss(
        vocab_size=vocab_size, 
        end_idx=3, 
        end_weight=2.0,
        ignore_index=0,
        smoothing=label_smoothing
    )
    criterion = criterion.to(device)
    
    # 优化器：AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8)
    
    # 学习率调度器：ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=scheduler_factor, 
        patience=scheduler_patience, 
        verbose=True, 
        min_lr=scheduler_min_lr
    )
    print(f"学习率调度器: ReduceLROnPlateau")
    print(f"  - factor: {scheduler_factor}")
    print(f"  - patience: {scheduler_patience}")
    print(f"  - min_lr: {scheduler_min_lr}")
    
    # 创建保存目录
    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    attention_dir = checkpoint_dir / "attention_visualizations_newdataset"
    attention_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    print(f"训练参数:")
    print(f"  - 训练轮数: {num_epochs}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 初始学习率: {learning_rate}")
    print(f"  - Teacher-Forcing比例: {teacher_forcing_ratio}")
    print(f"  - 标签平滑: {label_smoothing}")
    print(f"  - 早停patience: {early_stopping_patience}")
    print("="*60)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, max_norm, teacher_forcing_ratio)
        print(f"训练损失: {train_loss:.4f}")
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        print(f"验证损失: {val_loss:.4f}")
        
        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        improved = False
        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            improved = True
            
            checkpoint_path = checkpoint_dir / "model2_newdataset_checkpoint.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'embed_dim': embed_dim,
                'hidden_dim': hidden_dim,
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'learning_rate': current_lr,
            }, str(checkpoint_path))
            print(f"[OK] 最佳模型已保存到: {checkpoint_path} (验证损失: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"[INFO] 验证损失未改善 ({epochs_without_improvement}/{early_stopping_patience})")
        
        # 早停机制
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n{'='*60}")
            print(f"[早停] 验证损失连续{early_stopping_patience}个epoch未改善，提前停止训练")
            print(f"最佳验证损失: {best_val_loss:.4f}")
            print(f"{'='*60}")
            break
        
        # 注意力权重可视化（每5轮）
        if (epoch + 1) % 5 == 0:
            print("\n生成注意力热力图...")
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                sample_local_feat = sample_batch['local_feat'][:1].to(device)
                sample_caption = sample_batch['caption'][:1].to(device)
                
                outputs, attn_weights = model(sample_local_feat, sample_caption, 
                                             teacher_forcing_ratio=1.0, return_attn=True)
                
                if attn_weights is not None:
                    visualize_attention_heatmap(
                        attn_weights[0],
                        attention_dir,
                        epoch + 1,
                        sample_idx=0
                    )
    
    # 生成示例
    print("\n" + "="*60)
    print("生成示例（包含注意力可视化）")
    print("="*60)
    model.eval()
    
    sample_batch = next(iter(val_loader))
    sample_local_feat = sample_batch['local_feat'][:5].to(device)
    sample_caption = sample_batch['caption'][:5]
    
    with torch.no_grad():
        generated_sequences, attn_weights = model.generate(
            sample_local_feat, max_len=max_len, temperature=0.7, return_attn=True
        )
    
    print("\n生成的描述（包含注意力热力图）:")
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
        
        # 保存注意力热力图
        if attn_weights is not None:
            visualize_attention_heatmap(
                attn_weights[i],
                attention_dir,
                epoch=999,
                sample_idx=i
            )
    
    print("\n" + "="*60)
    print("[OK] 训练完成！")
    print("="*60)
    print(f"训练配置总结:")
    print(f"  - 总训练轮数: {num_epochs}")
    print(f"  - 实际训练轮数: {epoch + 1}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 初始学习率: {learning_rate}")
    print(f"  - 标签平滑: {label_smoothing}")
    print(f"  - Teacher-Forcing比例: {teacher_forcing_ratio}")
    print(f"\n最佳模型:")
    print(f"  - 最佳验证损失: {best_val_loss:.4f}")
    print(f"  - 模型已保存到: models/checkpoints/model2_newdataset_checkpoint.pt")
    print(f"  - 注意力热力图已保存到: {attention_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

