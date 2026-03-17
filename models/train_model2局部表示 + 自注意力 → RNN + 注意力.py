"""
训练Model2: 增强型CNN编码器（局部自注意力）+ 注意力增强LSTM解码器
改进的训练策略：混合Teacher-Forcing、学习率调度、注意力权重可视化
"""
import os
# 修复OpenMP错误：设置环境变量允许重复的OpenMP库
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LambdaLR
import sys
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

from models.model2_local_selfattn_attention_rnn import FashionCaptionModelAttention, load_vocab, visualize_attention
from utils.dataset import get_dataloader, ExpandedFashionCaptionDataset
from utils.data_loader import load_and_validate_dataset
from utils.text_processor import TextProcessor
import random


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    标签平滑交叉熵损失函数（改进版）
    结合加权损失和标签平滑，提升模型泛化能力
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
        计算标签平滑交叉熵损失
        
        Args:
            inputs: 模型输出 (batch * seq_len, vocab_size)
            targets: 目标标签 (batch * seq_len,)
        
        Returns:
            损失值
        """
        # 确保权重在正确的设备上
        weights = self.weights.to(inputs.device)
        
        # 标签平滑
        log_probs = F.log_softmax(inputs, dim=-1)  # (batch * seq_len, vocab_size)
        
        # 创建平滑后的目标分布
        batch_size = targets.size(0)
        true_dist = torch.zeros_like(log_probs)  # (batch * seq_len, vocab_size)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # 排除ignore_index和end_idx
        
        # 设置真实标签的概率
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # 处理ignore_index
        ignore_mask = (targets == self.ignore_index)
        true_dist[ignore_mask] = 0.0
        
        # 计算KL散度（等价于交叉熵）
        loss = -torch.sum(true_dist * log_probs, dim=-1)  # (batch * seq_len,)
        
        # 应用权重（<END>标记权重提升）
        end_mask = (targets == self.end_idx)
        loss = loss * (1.0 + (self.end_weight - 1.0) * end_mask.float())
        
        # 只计算有效位置的损失
        valid_mask = (targets != self.ignore_index)
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        return loss


def train_epoch(model, dataloader, criterion, optimizer, device, max_norm=1.0, 
                teacher_forcing_ratio=0.8, epoch=0, num_epochs=20):
    """
    训练一个epoch（改进版：动态Teacher Forcing）
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        max_norm: 梯度裁剪的最大范数
        teacher_forcing_ratio: Teacher-Forcing初始比例
        epoch: 当前epoch
        num_epochs: 总epoch数
    
    Returns:
        平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # 动态调整Teacher Forcing比例：从高到低线性衰减
    # 早期epoch使用高比例，后期epoch使用低比例
    current_tf_ratio = teacher_forcing_ratio * (1.0 - epoch / num_epochs * 0.5)  # 从teacher_forcing_ratio衰减到0.5*teacher_forcing_ratio
    current_tf_ratio = max(current_tf_ratio, 0.5)  # 最低不低于0.5
    
    for batch_idx, batch in enumerate(dataloader):
        local_feat = batch['local_feat'].to(device)
        caption = batch['caption'].to(device)
        
        # 前向传播（动态Teacher-Forcing）
        outputs, _ = model(local_feat, caption, teacher_forcing_ratio=current_tf_ratio, return_attn=False)
        
        # 计算损失（梯度累积：除以accumulation_steps）
        targets = caption[:, 1:]  # 去掉<START>标记
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1)) / accumulation_steps
        
        # 反向传播（累积梯度）
        loss.backward()
        
        total_loss += loss.item() * accumulation_steps  # 乘以accumulation_steps恢复原始loss值
        
        # 梯度累积：每accumulation_steps步更新一次参数
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪
            clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()  # 清零梯度，准备下一次累积
        
        num_batches += 1
        
        # 打印进度（减少打印频率以加速）
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item() * accumulation_steps:.4f}, TF Ratio: {current_tf_ratio:.3f}")
    
    
    # 如果最后还有未累积的梯度，进行更新
    if num_batches % accumulation_steps != 0:
        clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
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
    print("训练Model2: 增强型CNN编码器（局部自注意力）+ 注意力增强LSTM解码器")
    print("="*60)
    
    # 配置参数
    vocab_file = PROJECT_ROOT / "dataset" / "vocab.json"
    train_feature_file = PROJECT_ROOT / "dataset" / "features" / "train_features.npz"
    val_feature_file = PROJECT_ROOT / "dataset" / "features" / "val_features.npz"
    # 模型超参数（超大容量版 - 目标：至少一个指标50%+）
    embed_dim = 768  # 词嵌入维度（优化：512→768）
    hidden_dim = 1024  # LSTM隐藏状态维度（优化：768→1024）
    num_layers = 4  # LSTM层数（优化：3→4）
    dropout = 0.2  # Dropout比例（优化：0.15→0.2）
    cnn_output_dim = 1024  # CNN编码器输出维度（优化：768→1024）
    
    # 训练超参数（优化版）
    batch_size = 6  # 批次大小（优化：8→6，超大模型需要更小的batch_size）
    accumulation_steps = 3  # 梯度累积步数（模拟batch_size=18）
    num_epochs = 100  # 训练轮数（优化：40→100，更充分的训练）
    learning_rate = 3e-5  # 学习率（优化：4e-5→3e-5，超大模型需要更小的学习率）
    weight_decay = 5e-5  # 权重衰减（优化：1.5e-5→5e-5，更强的正则化）
    max_norm = 1.0  # 梯度裁剪（优化：0.8→1.0，对深度模型很重要）
    max_len = 35  # 最大生成长度（优化：25→35，允许生成更长的描述）
    teacher_forcing_ratio = 0.88  # Teacher-Forcing比例（优化：0.75→0.88，超大模型需要更多TF）
    early_stop_patience = 25  # 早停耐心（优化：8→25，更宽松的早停）
    skip_visualization = False  # 是否跳过注意力可视化
    
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
    
    # 创建模型（超大容量版）
    print("\n创建模型（超大容量版）...")
    print(f"  词嵌入维度: {embed_dim}")
    print(f"  LSTM隐藏状态维度: {hidden_dim}")
    print(f"  LSTM层数: {num_layers}")
    print(f"  Dropout: {dropout}")
    print(f"  CNN输出维度: {cnn_output_dim}")
    model = FashionCaptionModelAttention(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        cnn_output_dim=cnn_output_dim
    )
    model = model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")
    
    # 损失函数：标签平滑交叉熵损失（优化版）
    criterion = LabelSmoothingCrossEntropyLoss(
        vocab_size=vocab_size, 
        end_idx=3, 
        end_weight=2.5,  # 优化：1.5→2.5（更强的<END>权重，防止过早截断）
        ignore_index=0,
        smoothing=0.05  # 优化：0.12→0.05（降低标签平滑，提升模型自信度）
    )
    criterion = criterion.to(device)  # 确保损失函数在正确的设备上
    
    # 优化器：AdamW（优化版：更强的权重衰减）
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    
    # 学习率调度器：Warmup + CosineAnnealing + ReduceLROnPlateau
    # Warmup阶段：前8个epoch线性增长（优化：5→8，超大模型需要更长的预热）
    warmup_epochs = 8
    
    # CosineAnnealing调度器（在warmup后使用）
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=learning_rate * 0.01)
    
    # ReduceLROnPlateau作为辅助调度器
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=learning_rate * 0.0001)
    
    # 创建保存目录
    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    attention_dir = checkpoint_dir / "attention_visualizations"
    attention_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    best_val_loss = float('inf')
    no_improve_count = 0  # 用于早停
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # 训练（动态Teacher Forcing + 梯度累积）
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, max_norm, 
                                teacher_forcing_ratio, epoch, num_epochs, accumulation_steps)
        print(f"训练损失: {train_loss:.4f} (有效batch_size: {batch_size * accumulation_steps})")
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        print(f"验证损失: {val_loss:.4f}")
        
        # 学习率调度（Warmup + Cosine + Plateau）
        # Warmup阶段：前8个epoch线性增长学习率（优化：5→8）
        if epoch < warmup_epochs:
            warmup_start_lr = learning_rate * 0.1  # 从10%的学习率开始
            warmup_lr = warmup_start_lr + (learning_rate - warmup_start_lr) * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"学习率预热: {warmup_lr:.2e} (warmup epoch {epoch + 1}/{warmup_epochs})")
        else:
            # Warmup后使用CosineAnnealing
            scheduler_cosine.step()
        
        # ReduceLROnPlateau按验证损失更新（总是执行）
        scheduler_plateau.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0  # 重置计数器
            improved = True
            
            checkpoint_path = checkpoint_dir / "model2_checkpoint.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'embed_dim': embed_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout,
                'cnn_output_dim': cnn_output_dim,
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, str(checkpoint_path))
            print(f"[OK] 最佳模型已保存到: {checkpoint_path} (验证损失: {val_loss:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                print(f"\n[早停] 验证损失连续{early_stop_patience}个epoch未改善，停止训练")
                print(f"最佳验证损失: {best_val_loss:.4f}")
                break
        
        # 注意力权重可视化（每10轮打印1个样本的注意力热力图，减少频率以加速）
        if not skip_visualization and (epoch + 1) % 10 == 0:
            try:
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
            except Exception as e:
                print(f"[警告] 生成注意力热力图时出错: {e}")
                print("继续训练...")
                model.train()  # 确保模型回到训练模式
    
    # 生成示例（包含注意力可视化）
    print("\n" + "="*60)
    print("生成示例（包含注意力可视化）")
    print("="*60)
    try:
        model.eval()
        
        # 从验证集取5个样本
        sample_batch = next(iter(val_loader))
        sample_local_feat = sample_batch['local_feat'][:5].to(device)
        sample_caption = sample_batch['caption'][:5]
        
        with torch.no_grad():
            generated_sequences, attn_weights = model.generate(
                sample_local_feat, max_len=25, temperature=0.7, return_attn=True
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
                try:
                    visualize_attention_heatmap(
                        attn_weights[i],  # (generated_len, 49)
                        attention_dir,
                        epoch=999,  # 使用999表示最终生成示例
                        sample_idx=i
                    )
                except Exception as e:
                    print(f"[警告] 保存样本{i+1}的注意力热力图时出错: {e}")
    except Exception as e:
        print(f"[警告] 生成示例时出错: {e}")
        print("训练已完成，但示例生成失败")
    
    print("\n" + "="*60)
    print("[OK] 训练完成！")
    print("="*60)
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型已保存到: models/checkpoints/model2_checkpoint.pt")
    print(f"注意力热力图已保存到: {attention_dir}")


if __name__ == "__main__":
    main()

