"""
训练Model2增强版: 多层自注意力编码器 + 增强型LSTM解码器 + 加性注意力
改进的训练策略：混合Teacher-Forcing、学习率调度、注意力权重可视化
目标：METEOR ≥ 0.7
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
matplotlib.use('Agg')  # 非交互式后端，用于保存图像
import matplotlib.pyplot as plt

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from models.model2_enhanced import FashionCaptionModelEnhanced, load_vocab
from utils.dataset import get_dataloader, ExpandedFashionCaptionDataset
from utils.data_loader import load_and_validate_dataset
from utils.text_processor import TextProcessor
import random


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    标签平滑 + 加权交叉熵损失函数
    结合标签平滑（提升泛化）和加权损失（<END>权重提升）
    """
    
    def __init__(self, vocab_size: int, end_idx: int = 3, end_weight: float = 1.5, 
                 ignore_index: int = 0, smoothing: float = 0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.vocab_size = vocab_size
        self.end_idx = end_idx
        self.end_weight = end_weight
        self.ignore_index = ignore_index
        self.smoothing = smoothing
        
        # 使用register_buffer确保权重会自动移动到正确的设备
        weights = torch.ones(vocab_size)
        weights[end_idx] = end_weight
        self.register_buffer('weights', weights)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播：标签平滑交叉熵损失
        
        Args:
            inputs: 模型输出 (batch * seq_len, vocab_size)
            targets: 真实标签 (batch * seq_len,)
        
        Returns:
            损失值
        """
        batch_size = inputs.size(0)
        vocab_size = inputs.size(1)
        device = inputs.device
        
        # 计算标准交叉熵损失
        log_probs = F.log_softmax(inputs, dim=1)  # (batch * seq_len, vocab_size)
        
        # 创建标签平滑的目标分布
        # 真实标签：1-smoothing，其他所有标签：smoothing/(vocab_size-1)
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (vocab_size - 1))
        
        # 为每个样本的真实标签分配更高概率
        # 使用scatter_高效赋值
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # 将padding标记的平滑概率设为0（只在有效位置计算损失）
        valid_mask = (targets != self.ignore_index).unsqueeze(1).expand_as(smooth_targets)
        smooth_targets = smooth_targets * valid_mask.float()
        
        # 归一化（确保概率和为1）
        smooth_targets = smooth_targets / (smooth_targets.sum(dim=1, keepdim=True) + 1e-8)
        
        # 计算交叉熵损失（使用平滑标签）
        loss = -torch.sum(smooth_targets * log_probs, dim=1)  # (batch * seq_len,)
        
        # 应用权重（<END>标记权重提升）
        weights = self.weights.to(device)
        end_mask = (targets == self.end_idx)
        loss = loss * weights[targets] * (1.0 + (self.end_weight - 1.0) * end_mask.float())
        
        # 忽略padding标记
        if valid_mask[:, 0].sum() > 0:
            loss = loss[valid_mask[:, 0]].mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    加权交叉熵损失函数（备用，不使用标签平滑时）
    给<END>标记的损失权重提升1.5倍（避免提前截断）
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
        # 确保权重在正确的设备上（register_buffer应该已经处理了，但为了安全起见）
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
    """
    验证模型
    
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
    """
    可视化注意力权重热力图并保存
    
    Args:
        attn_weights: 注意力权重 (seq_len, 49)
        save_dir: 保存目录
        epoch: 当前epoch
        sample_idx: 样本索引
    """
    if attn_weights.dim() == 1:
        attn_weights = attn_weights.unsqueeze(0)
    
    seq_len, num_pixels = attn_weights.shape
    h, w = 7, 7
    
    # 将注意力权重重塑为7×7
    attn_map = attn_weights.view(seq_len, h, w).cpu().numpy()
    
    # 创建子图
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
    
    # 保存图像
    save_path = save_dir / f"attention_epoch_{epoch}_sample_{sample_idx}.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] 注意力热力图已保存到: {save_path}")


def main():
    """主训练函数"""
    print("="*60)
    print("训练Model2增强版: 多层自注意力编码器 + 增强型LSTM解码器 + 加性注意力")
    print("目标：METEOR ≥ 0.7")
    print("="*60)
    
    # 配置参数（增强版推荐配置）
    vocab_file = PROJECT_ROOT / "dataset" / "vocab.json"
    train_feature_file = PROJECT_ROOT / "dataset" / "features" / "train_features.npz"
    val_feature_file = PROJECT_ROOT / "dataset" / "features" / "val_features.npz"
    
    # 模型超参数（增强版）
    embed_dim = 512      # 词嵌入维度（原256）
    hidden_dim = 768     # LSTM隐藏状态维度（原512）
    num_layers = 3       # LSTM层数（原2）
    dropout = 0.3        # Dropout比例（原0.1）
    
    # 训练超参数（优化版：目标METEOR ≥ 0.7）
    batch_size = 32      # 批次大小（根据GPU内存调整，如果内存不足可降为16）
    num_epochs = 80      # 训练轮数（增加以获得更好效果，充分训练）
    learning_rate = 5e-5  # 学习率（优化：稍低的学习率更稳定，达到0.7目标）
    weight_decay = 1e-5  # 权重衰减
    max_norm = 1.0       # 梯度裁剪
    max_len = 30         # 最大生成长度
    teacher_forcing_ratio = 0.9  # 90% Teacher-Forcing（高TF比例有助于学习）
    
    # 学习率调度器参数（优化）- 使用余弦退火预热
    use_cosine_scheduler = True  # 使用余弦退火替代ReduceLROnPlateau
    warmup_epochs = 5  # 前5个epoch线性增加学习率
    scheduler_patience = 10  # 如果使用ReduceLROnPlateau，增加patience
    scheduler_factor = 0.7  # 每次降低更多（0.7倍）
    scheduler_min_lr = 5e-7  # 最小学习率
    
    # 早停机制参数（防止过拟合）- 增加patience以充分训练
    early_stopping_patience = 20  # 如果验证loss连续20个epoch不下降，提前停止（增加以充分训练）
    early_stopping_min_delta = 1e-5  # 最小改进阈值
    
    # 标签平滑参数（提升泛化能力）- 降低标签平滑以提升模型自信度
    label_smoothing = 0.05  # 标签平滑系数（0.05表示95%置信度给真实标签，降低以提升性能）
    
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
    
    # 处理验证集文本序列
    val_sequences = processor.batch_process(val_data, max_len=max_len)
    
    # 扩展训练集：每个样本使用所有描述（提升数据多样性）
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
        num_workers=0,  # Windows兼容性
        pin_memory=False
    )
    val_loader = get_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # 创建增强版模型
    print("\n创建增强版模型...")
    print(f"  词嵌入维度: {embed_dim}")
    print(f"  LSTM隐藏状态维度: {hidden_dim}")
    print(f"  LSTM层数: {num_layers}")
    print(f"  Dropout: {dropout}")
    model = FashionCaptionModelEnhanced(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    model = model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")
    
    # 损失函数：标签平滑交叉熵损失（提升泛化能力，有助于达到0.7目标）
    print(f"\n使用标签平滑交叉熵损失（smoothing={label_smoothing}）")
    criterion = LabelSmoothingCrossEntropyLoss(
        vocab_size=vocab_size, 
        end_idx=3, 
        end_weight=2.0,  # 增加<END>标记权重（从1.5到2.0），避免提前截断
        ignore_index=0,
        smoothing=label_smoothing
    )
    criterion = criterion.to(device)  # 确保损失函数在正确的设备上
    
    # 优化器：AdamW（比Adam更稳定）+ weight_decay
    # 注意：AdamW中weight_decay的作用更明确，推荐用于大模型训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8)
    
    # 学习率调度器：使用余弦退火或ReduceLROnPlateau
    if use_cosine_scheduler:
        try:
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            # 余弦退火调度器：更平滑的学习率衰减
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_epochs,  # 余弦退火的周期
                eta_min=scheduler_min_lr
            )
            # 预热调度器：前warmup_epochs个epoch线性增加学习率
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,  # 从10%的学习率开始
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            # 组合调度器：先预热，然后余弦退火
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            print(f"学习率调度器: SequentialLR (预热 + 余弦退火)")
            print(f"  - 预热epochs: {warmup_epochs} (学习率从{learning_rate * 0.1:.2e}线性增加到{learning_rate:.2e})")
            print(f"  - 余弦退火: T_max={num_epochs - warmup_epochs}, eta_min={scheduler_min_lr}")
        except ImportError:
            # 如果PyTorch版本不支持SequentialLR，使用ReduceLROnPlateau
            print(f"[WARN] PyTorch版本不支持SequentialLR，使用ReduceLROnPlateau")
            use_cosine_scheduler = False
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=scheduler_factor, 
                patience=scheduler_patience, 
                verbose=True, 
                min_lr=scheduler_min_lr
            )
    if not use_cosine_scheduler:
        # ReduceLROnPlateau调度器（备用）
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=scheduler_factor, 
            patience=scheduler_patience, 
            verbose=True, 
            min_lr=scheduler_min_lr
        )
        print(f"学习率调度器: ReduceLROnPlateau")
        print(f"  - factor: {scheduler_factor} (每次降低{scheduler_factor}倍)")
        print(f"  - patience: {scheduler_patience} (等待{scheduler_patience}个epoch)")
        print(f"  - min_lr: {scheduler_min_lr}")
    
    # 创建保存目录
    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    attention_dir = checkpoint_dir / "attention_visualizations"
    attention_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    print("\n" + "="*60)
    print("开始训练（优化版配置，目标METEOR ≥ 0.7）")
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
    epochs_without_improvement = 0  # 早停计数器
    
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
        if use_cosine_scheduler:
            scheduler.step()  # 余弦退火调度器不需要val_loss
        else:
            scheduler.step(val_loss)  # ReduceLROnPlateau需要val_loss
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")
        
        # 保存最佳模型（改进：只有明显改进才保存）
        improved = False
        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0  # 重置计数器
            improved = True
            
            checkpoint_path = checkpoint_dir / "model2_enhanced_checkpoint.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'embed_dim': embed_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout,
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'learning_rate': current_lr,
            }, str(checkpoint_path))
            print(f"[OK] 最佳模型已保存到: {checkpoint_path} (验证损失: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"[INFO] 验证损失未改善 ({epochs_without_improvement}/{early_stopping_patience})")
        
        # 早停机制（如果验证loss连续多个epoch不下降，提前停止）
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n{'='*60}")
            print(f"[早停] 验证损失连续{early_stopping_patience}个epoch未改善，提前停止训练")
            print(f"最佳验证损失: {best_val_loss:.4f}")
            print(f"{'='*60}")
            break
        
        # 注意力权重可视化（每5轮打印1个样本的注意力热力图）
        if (epoch + 1) % 5 == 0:
            print("\n生成注意力热力图...")
            model.eval()
            with torch.no_grad():
                # 从验证集取1个样本
                sample_batch = next(iter(val_loader))
                sample_local_feat = sample_batch['local_feat'][:1].to(device)  # 只取1个样本
                sample_caption = sample_batch['caption'][:1].to(device)
                
                # 前向传播（返回注意力权重）
                outputs, attn_weights = model(sample_local_feat, sample_caption, 
                                             teacher_forcing_ratio=1.0, return_attn=True)
                
                if attn_weights is not None:
                    # 可视化注意力权重（取第一个样本）
                    visualize_attention_heatmap(
                        attn_weights[0],  # (seq_len-1, 49)
                        attention_dir,
                        epoch + 1,
                        sample_idx=0
                    )
    
    # 生成示例（包含注意力可视化）
    print("\n" + "="*60)
    print("生成示例（包含注意力可视化）")
    print("="*60)
    model.eval()
    
    # 从验证集取5个样本
    sample_batch = next(iter(val_loader))
    sample_local_feat = sample_batch['local_feat'][:5].to(device)
    sample_caption = sample_batch['caption'][:5]
    
    with torch.no_grad():
        generated_sequences, attn_weights = model.generate(
            sample_local_feat, max_len=max_len, temperature=0.8, return_attn=True
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
                attn_weights[i],  # (generated_len, 49)
                attention_dir,
                epoch=999,  # 使用999表示最终生成示例
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
    print(f"  - 模型已保存到: models/checkpoints/model2_enhanced_checkpoint.pt")
    print(f"  - 注意力热力图已保存到: {attention_dir}")
    print(f"\n下一步:")
    print(f"  - 评估模型性能: bash run/evaluate_model.sh")
    print(f"  - 目标METEOR ≥ 0.7")
    print("="*60)


if __name__ == "__main__":
    main()

