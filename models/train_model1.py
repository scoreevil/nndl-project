"""
训练Model1: 常规CNN编码器 + 6层基础RNN解码器
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # 切换到项目根目录

from models.model1_regular_cnn_6layer_rnn import FashionCaptionModel, load_vocab
from utils.dataset import FashionCaptionDataset, get_dataloader
from utils.data_loader import load_and_validate_dataset
from utils.text_processor import build_vocab_and_process


def train_epoch(model, dataloader, criterion, optimizer, device, max_norm=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 移动到设备
        local_feat = batch['local_feat'].to(device)  # (batch, 2048, 7, 7)
        caption = batch['caption'].to(device)  # (batch, max_len)
        
        # 前向传播
        outputs = model(local_feat, caption)  # (batch, seq_len-1, vocab_size)
        
        # 准备目标：去掉第一个词（<START>）
        targets = caption[:, 1:]  # (batch, seq_len-1)
        
        # 计算损失
        # outputs: (batch, seq_len-1, vocab_size) → (batch * (seq_len-1), vocab_size)
        # targets: (batch, seq_len-1) → (batch * (seq_len-1),)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（缓解6层RNN梯度爆炸问题）
        clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 打印进度
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """主训练函数"""
    print("="*60)
    print("训练Model1: 常规CNN编码器 + 6层基础RNN解码器")
    print("="*60)
    
    # 配置参数（相对于项目根目录）
    vocab_file = PROJECT_ROOT / "dataset" / "vocab.json"
    train_feature_file = PROJECT_ROOT / "dataset" / "features" / "train_features.npz"
    val_feature_file = PROJECT_ROOT / "dataset" / "features" / "val_features.npz"
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-4
    max_norm = 1.0  # 梯度裁剪
    max_len = 20
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
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
    
    # 创建模型
    print("\n创建模型...")
    model = FashionCaptionModel(vocab_size=vocab_size)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略<PAD>标记
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, max_norm)
        print(f"训练损失: {train_loss:.4f}")
        
        # 验证（可选）
        if (epoch + 1) % 2 == 0:  # 每2个epoch验证一次
            model.eval()
            val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    local_feat = batch['local_feat'].to(device)
                    caption = batch['caption'].to(device)
                    
                    outputs = model(local_feat, caption)
                    targets = caption[:, 1:]
                    
                    loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                    val_loss += loss.item()
                    num_val_batches += 1
            
            avg_val_loss = val_loss / num_val_batches
            print(f"验证损失: {avg_val_loss:.4f}")
    
    # 保存模型
    model_save_path = PROJECT_ROOT / "models" / "model1_checkpoint.pt"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'epoch': num_epochs,
    }, str(model_save_path))
    print(f"\n[OK] 模型已保存到: {model_save_path}")
    
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
        generated_sequences = model.generate(sample_local_feat, max_len=max_len)
    
    print("\n生成的描述:")
    for i in range(3):
        # 真实描述
        true_seq = sample_caption[i].tolist()
        true_words = [idx2word.get(idx, f"<{idx}>") for idx in true_seq if idx != 0]
        print(f"\n样本{i+1}:")
        print(f"  真实: {' '.join(true_words)}")
        
        # 生成描述
        gen_seq = generated_sequences[i].cpu().tolist()
        gen_words = [idx2word.get(idx, f"<{idx}>") for idx in gen_seq]
        print(f"  生成: {' '.join(gen_words)}")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)


if __name__ == "__main__":
    main()

