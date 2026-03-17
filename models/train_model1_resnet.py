"""
训练Model1_ResNet: ResNet编码器 + LSTM解码器 + 注意力机制（改进版）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import sys
import os
from pathlib import Path
import math

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # 切换到项目根目录

from models.model1_resnet import FashionCaptionModelResNet, load_vocab
from utils.dataset import FashionCaptionDataset, get_dataloader
from utils.data_loader import load_and_validate_dataset
from utils.text_processor import build_vocab_and_process


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """标签平滑交叉熵损失（带<END>标记权重）"""
    
    def __init__(self, vocab_size: int, smoothing: float = 0.05, end_idx: int = 3, end_weight: float = 2.0, ignore_index: int = 0):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.end_idx = end_idx
        self.end_weight = end_weight
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch * seq_len, vocab_size)
            target: (batch * seq_len,)
        """
        log_probs = F.log_softmax(pred, dim=-1)
        
        # 创建平滑标签
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.ignore_index] = 0  # 忽略<PAD>
        
        # 创建mask（忽略<PAD>）
        mask = (target != self.ignore_index).float()
        
        # 计算权重（<END>标记权重更高）
        weights = torch.ones_like(target, dtype=torch.float)
        weights[target == self.end_idx] = self.end_weight
        
        # 计算损失
        loss = -torch.sum(true_dist * log_probs, dim=-1) * mask * weights
        loss = loss.sum() / (mask.sum() + 1e-8)
        
        return loss


def train_epoch(model, dataloader, criterion, optimizer, device, max_norm=1.0, 
                teacher_forcing_ratio=0.9, accumulation_steps=2):
    """
    训练一个epoch（带梯度累积）
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        max_norm: 梯度裁剪阈值
        teacher_forcing_ratio: Teacher-Forcing比例
        accumulation_steps: 梯度累积步数（模拟更大的batch_size）
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()  # 在epoch开始时清零梯度
    
    for batch_idx, batch in enumerate(dataloader):
        # 移动到设备
        local_feat = batch['local_feat'].to(device)  # (batch, 2048, 7, 7)
        caption = batch['caption'].to(device)  # (batch, max_len)
        
        # 前向传播（返回outputs和可选的attn_weights）
        outputs, _ = model(local_feat, caption, teacher_forcing_ratio=teacher_forcing_ratio)  # (batch, seq_len-1, vocab_size)
        
        # 准备目标：去掉第一个词（<START>）
        targets = caption[:, 1:]  # (batch, seq_len-1)
        
        # 计算损失（除以accumulation_steps，因为后面会累积）
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1)) / accumulation_steps
        
        # 反向传播（累积梯度）
        loss.backward()
        
        total_loss += loss.item() * accumulation_steps  # 乘以accumulation_steps恢复原始loss值
        
        # 每accumulation_steps步更新一次参数
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪
            clip_grad_norm_(model.parameters(), max_norm=max_norm)
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()  # 清零梯度，准备下一次累积
        
        num_batches += 1
        
        # 打印进度
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item() * accumulation_steps:.4f}")
    
    # 如果最后还有未累积的梯度，进行更新
    if num_batches % accumulation_steps != 0:
        clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """主训练函数"""
    print("="*60)
    print("训练Model1_ResNet: ResNet编码器 + 5层超大容量LSTM解码器 + 注意力机制（超大容量版）")
    print("="*60)
    print("🎯 目标：至少一个指标达到80%以上！")
    print("🚀 超大容量优化：embed_dim=768, hidden_dim=1024, 5层LSTM + 强化训练策略")
    print("="*60)
    
    # 配置参数（相对于项目根目录）- 优化版参数
    vocab_file = PROJECT_ROOT / "dataset" / "vocab.json"
    train_feature_file = PROJECT_ROOT / "dataset" / "features" / "train_features.npz"
    val_feature_file = PROJECT_ROOT / "dataset" / "features" / "val_features.npz"
    batch_size = 12  # 优化：16→12（超大容量模型需要更小的batch_size以节省GPU内存）
    accumulation_steps = 3  # 梯度累积步数（优化：2→3，模拟batch_size=36，更稳定的梯度）
    num_epochs = 120  # 优化：100→120（超大容量模型需要更多训练轮数，目标80%+）
    learning_rate = 4e-5  # 初始学习率（优化：5e-5→4e-5，超大模型需要更小的学习率）
    warmup_epochs = 8  # 学习率预热epochs（优化：5→8，超大模型需要更长的预热）
    max_norm = 1.0  # 梯度裁剪（保持1.0，对深度模型很重要）
    max_len = 45  # 优化：40→45（给生成更多空间，目标80%+）
    teacher_forcing_ratio = 0.9  # 优化：0.88→0.9（超大模型初期需要更多Teacher-Forcing）
    
    # ResNet配置
    resnet_type = 'resnet50'  # 可选: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    pretrained = True  # 是否使用预训练权重
    
    # 模型超参数（超大容量版 - 目标：至少一个指标80%+）
    embed_dim = 768  # 词嵌入维度（优化：512→768，更强的词表示能力）
    hidden_dim = 1024  # LSTM隐藏状态维度（优化：768→1024，最大化模型容量，目标80%+）
    num_layers = 5  # LSTM层数（5层深度LSTM）
    dropout = 0.4  # Dropout比例（防止过拟合）
    
    # 学习率调度器配置（针对超大容量模型优化）
    use_scheduler = True
    scheduler_patience = 10  # 优化：8→10（超大模型需要更耐心的学习率调度）
    scheduler_factor = 0.5  # 学习率衰减因子
    scheduler_min_lr = 1e-7  # 优化：5e-7→1e-7（更小的最小学习率，确保超大模型充分收敛）
    
    # 早停配置（针对超大容量模型优化）
    use_early_stopping = True
    early_stopping_patience = 30  # 优化：25→30（超大模型需要更多epoch才能充分训练，更宽松的早停）
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    print(f"ResNet类型: {resnet_type}")
    print(f"使用预训练权重: {pretrained}")
    
    # 加载词典
    print("\n加载词典...")
    vocab_info = load_vocab(str(vocab_file))
    vocab_size = vocab_info['vocab_size']
    print(f"词典大小: {vocab_size}")
    
    # 加载数据集和文本序列
    print("\n加载数据集...")
    train_data, val_data, test_data = load_and_validate_dataset(
        captions_file=str(PROJECT_ROOT / "dataset" / "captions.json"),
        images_dir=str(PROJECT_ROOT / "dataset" / "images"),
        random_seed=42
    )
    
    # 文本预处理（如果序列文件不存在）
    sequences_file = PROJECT_ROOT / "run" / "text_sequences.pt"
    if sequences_file.exists():
        print(f"\n加载文本序列: {sequences_file}")
        sequences_data = torch.load(str(sequences_file))
        train_sequences = sequences_data['train_sequences']
        val_sequences = sequences_data['val_sequences']
    else:
        print("\n文本预处理...")
        _, train_sequences, val_sequences, _ = build_vocab_and_process(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            max_len=max_len,
            min_freq=3
        )
    
    # 创建Dataset和DataLoader
    print("\n创建DataLoader...")
    train_dataset = FashionCaptionDataset(str(train_feature_file), train_sequences)
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
    
    # 创建模型（改进版）
    print("\n创建模型（改进版：LSTM + 注意力）...")
    model = FashionCaptionModelResNet(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        resnet_type=resnet_type,
        pretrained=pretrained,
        num_layers=num_layers,
        dropout=dropout
    )
    model = model.to(device)
    
    # 损失函数（标签平滑 + <END>权重）- 针对超大容量模型优化
    criterion = LabelSmoothingCrossEntropyLoss(
        vocab_size=vocab_size,
        smoothing=0.05,  # 优化：0.08→0.05（超大模型需要更自信的预测，目标80%+）
        end_idx=3,
        end_weight=2.5,  # 优化：3.5→2.5（降低end_weight，配合生成时的min_length和end_penalty使用）
        ignore_index=0
    )
    
    # 优化器（AdamW，更强的权重衰减）- 针对超大容量模型
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # 优化：5e-5→1e-4（超大模型需要更强的正则化，目标80%+）
    
    # 学习率调度器（结合warmup和ReduceLROnPlateau）
    # 注意：warmup在训练循环中手动实现，因为PyTorch的调度器不支持ReduceLROnPlateau和warmup直接组合
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
            verbose=True
        )
    
    # 训练循环（带梯度累积、学习率预热、早停和学习率调度）
    print("\n" + "="*60)
    print("开始训练（超大容量版：目标80%+指标）")
    print("="*60)
    print(f"模型容量: embed_dim={embed_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")
    print(f"有效batch_size: {batch_size} × {accumulation_steps} = {batch_size * accumulation_steps}")
    print(f"训练轮数: {num_epochs} epochs")
    print(f"学习率预热: {warmup_epochs} epochs (1e-7 → {learning_rate:.2e})")
    print(f"权重衰减: 1e-4（更强的正则化）")
    print("="*60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # 学习率预热（前warmup_epochs个epoch线性增加学习率）
        if epoch < warmup_epochs:
            warmup_lr = 1e-7 + (learning_rate - 1e-7) * (epoch + 1) / warmup_epochs  # 优化：从1e-7开始（更小的起始学习率）
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"学习率预热: {warmup_lr:.2e} (warmup epoch {epoch + 1}/{warmup_epochs})")
        else:
            print(f"当前学习率: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 训练（带梯度累积）
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, max_norm, 
                                 teacher_forcing_ratio, accumulation_steps)
        print(f"训练损失: {train_loss:.4f} (有效batch_size: {batch_size * accumulation_steps})")
        
        # 验证（每个epoch都验证）
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                local_feat = batch['local_feat'].to(device)
                caption = batch['caption'].to(device)
                
                outputs, _ = model(local_feat, caption, teacher_forcing_ratio=1.0)
                targets = caption[:, 1:]
                
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        print(f"验证损失: {avg_val_loss:.4f}")
        
        # 学习率调度（warmup期间不调度）
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step(avg_val_loss)
        
        # 早停检查
        if use_early_stopping:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                best_model_path = checkpoint_dir / "model1_resnet_best.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vocab_size': vocab_size,
                    'embed_dim': embed_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'resnet_type': resnet_type,
                    'pretrained': pretrained,
                    'epoch': epoch + 1,
                    'val_loss': avg_val_loss,
                }, str(best_model_path))
                print(f"✅ 保存最佳模型 (验证损失: {avg_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"验证损失未改善 ({patience_counter}/{early_stopping_patience})")
                
                if patience_counter >= early_stopping_patience:
                    print(f"\n早停触发！最佳验证损失: {best_val_loss:.4f}")
                    # 加载最佳模型
                    checkpoint = torch.load(best_model_path)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    break
    
    # 保存最终模型
    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = checkpoint_dir / "model1_resnet_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'resnet_type': resnet_type,
        'pretrained': pretrained,
        'epoch': epoch + 1,
        'val_loss': avg_val_loss,
    }, str(model_save_path))
    print(f"\n[OK] 最终模型已保存到: {model_save_path}")
    
    # 生成示例
    print("\n" + "="*60)
    print("生成示例")
    print("="*60)
    model.eval()
    idx2word = vocab_info['idx2word']
    
    # 从验证集取3个样本
    sample_batch = next(iter(val_loader))
    sample_local_feat = sample_batch['local_feat'][:3].to(device)
    sample_caption = sample_batch['caption'][:3]
    
    with torch.no_grad():
        generated_sequences, _ = model.generate(sample_local_feat, max_len=max_len, beam_size=5)
    
    print("\n生成的描述（使用Beam Search）:")
    for i in range(3):
        # 真实描述
        true_seq = sample_caption[i].tolist()
        true_words = [idx2word.get(idx, f"<{idx}>") for idx in true_seq if idx not in [0, 2, 3]]
        print(f"\n样本{i+1}:")
        print(f"  真实: {' '.join(true_words)}")
        
        # 生成描述
        gen_seq = generated_sequences[i].cpu()
        gen_words = model.postprocess_caption(gen_seq, idx2word)
        print(f"  生成: {' '.join(gen_words) if gen_words else '(空)'}")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)


if __name__ == "__main__":
    main()
