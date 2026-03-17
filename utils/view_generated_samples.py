"""
查看模型生成的样本
用于诊断模型生成质量问题
"""
import torch
import sys
import os
from pathlib import Path
import json

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from models.model2_enhanced import FashionCaptionModelEnhanced, load_vocab
from utils.dataset import FashionCaptionDataset, get_dataloader
from utils.data_loader import load_and_validate_dataset


def view_samples(model_checkpoint_path: str, val_feature_file: str, 
                 val_sequences_file: str, vocab_file: str,
                 num_samples: int = 20, batch_size: int = 8,
                 max_len: int = 30, device: str = None):
    """
    查看模型生成的样本
    
    Args:
        model_checkpoint_path: 模型checkpoint路径
        val_feature_file: 验证集特征文件
        val_sequences_file: 验证集序列文件
        vocab_file: 词典文件
        num_samples: 要查看的样本数量
        batch_size: 批次大小
        max_len: 最大生成长度
        device: 设备
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    print("="*60)
    print("查看模型生成的样本")
    print("="*60)
    print(f"模型checkpoint: {model_checkpoint_path}")
    print(f"设备: {device}")
    
    # 加载词典
    print(f"\n加载词典...")
    vocab_info = load_vocab(vocab_file)
    idx2word = vocab_info['idx2word']
    vocab_size = vocab_info['vocab_size']
    print(f"词典大小: {vocab_size}")
    
    # 加载模型
    print(f"\n加载模型...")
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    
    # 从checkpoint获取模型参数
    embed_dim = checkpoint.get('embed_dim', 512)
    hidden_dim = checkpoint.get('hidden_dim', 768)
    num_layers = checkpoint.get('num_layers', 3)
    dropout = checkpoint.get('dropout', 0.3)
    
    print(f"模型参数: embed_dim={embed_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}")
    
    model = FashionCaptionModelEnhanced(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"[OK] 模型已加载")
    
    # 加载验证集数据
    print(f"\n加载验证集数据...")
    val_sequences_data = torch.load(val_sequences_file, map_location='cpu')
    val_sequences = val_sequences_data['val_sequences']
    
    val_dataset = FashionCaptionDataset(val_feature_file, val_sequences)
    val_loader = get_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 加载参考描述
    print(f"\n加载参考描述...")
    train_data, val_data, test_data = load_and_validate_dataset(
        captions_file=str(PROJECT_ROOT / "dataset" / "captions.json"),
        images_dir=str(PROJECT_ROOT / "dataset" / "images"),
        random_seed=42
    )
    print(f"验证集参考描述数: {len(val_data)}")
    
    # 生成样本
    print(f"\n{'='*60}")
    print(f"生成并查看前 {num_samples} 个样本")
    print(f"{'='*60}\n")
    
    generated_samples = []
    reference_samples = []
    
    with torch.no_grad():
        sample_count = 0
        for batch_idx, batch in enumerate(val_loader):
            if sample_count >= num_samples:
                break
            
            local_feat = batch['local_feat'].to(device)
            batch_size_current = local_feat.size(0)
            
            # 生成描述
            generated_sequences, _ = model.generate(
                local_feat, 
                max_len=max_len, 
                temperature=0.8, 
                return_attn=False
            )
            
            # 处理每个样本
            for i in range(batch_size_current):
                if sample_count >= num_samples:
                    break
                
                # 获取生成的序列
                gen_seq = generated_sequences[i].cpu()
                gen_words = model.postprocess_caption(gen_seq, idx2word)
                gen_text = ' '.join(gen_words) if gen_words else ''
                
                # 获取参考描述
                global_idx = batch_idx * batch_size + i
                if global_idx < len(val_data):
                    ref_item = val_data[global_idx]
                    ref_text = ref_item.get('caption', '') if isinstance(ref_item, dict) else ''
                else:
                    ref_text = ''
                
                # 获取原始序列（用于对比）
                if global_idx < len(val_sequences):
                    orig_seq = val_sequences[global_idx]
                    orig_words = [idx2word.get(int(idx), f"<{int(idx)}>") for idx in orig_seq.cpu().tolist() if int(idx) not in [0, 1, 2, 3]]
                    orig_text = ' '.join(orig_words)
                else:
                    orig_text = ''
                
                generated_samples.append(gen_text)
                reference_samples.append(ref_text)
                
                # 打印样本
                print(f"样本 {sample_count + 1}:")
                print(f"  原始序列: {orig_text[:100]}..." if len(orig_text) > 100 else f"  原始序列: {orig_text}")
                print(f"  生成描述: {gen_text if gen_text else '(空)'}")
                print(f"  参考描述: {ref_text if ref_text else '(无)'}")
                print(f"  生成长度: {len(gen_words)} 词")
                print(f"  生成序列: {gen_seq.tolist()[:20]}..." if len(gen_seq) > 20 else f"  生成序列: {gen_seq.tolist()}")
                print()
                
                sample_count += 1
            
            if sample_count >= num_samples:
                break
    
    # 统计信息
    print(f"{'='*60}")
    print(f"统计信息:")
    print(f"{'='*60}")
    empty_count = sum(1 for cap in generated_samples if not cap or cap.strip() == "")
    avg_gen_len = sum(len(cap.split()) for cap in generated_samples) / len(generated_samples) if generated_samples else 0
    avg_ref_len = sum(len(ref.split()) for ref in reference_samples) / len(reference_samples) if reference_samples else 0
    
    print(f"总样本数: {len(generated_samples)}")
    print(f"空生成数量: {empty_count} ({100*empty_count/len(generated_samples):.1f}%)")
    print(f"平均生成长度: {avg_gen_len:.1f} 词")
    print(f"平均参考长度: {avg_ref_len:.1f} 词")
    
    # 检查是否有重复生成
    unique_gens = set(generated_samples)
    print(f"唯一生成数量: {len(unique_gens)} / {len(generated_samples)}")
    
    # 检查最常见的生成
    from collections import Counter
    gen_counter = Counter(generated_samples)
    print(f"\n最常见的生成（前5个）:")
    for gen, count in gen_counter.most_common(5):
        print(f"  '{gen[:50]}...' (出现 {count} 次)" if len(gen) > 50 else f"  '{gen}' (出现 {count} 次)")
    
    print(f"\n{'='*60}")
    print(f"[OK] 样本查看完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='查看模型生成的样本')
    parser.add_argument('--model', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--val_features', type=str, default='dataset/features/val_features.npz', help='验证集特征文件')
    parser.add_argument('--val_sequences', type=str, default='run/text_sequences.pt', help='验证集序列文件')
    parser.add_argument('--vocab', type=str, default='dataset/vocab.json', help='词典文件')
    parser.add_argument('--num_samples', type=int, default=20, help='要查看的样本数量')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--max_len', type=int, default=30, help='最大生成长度')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    view_samples(
        model_checkpoint_path=args.model,
        val_feature_file=args.val_features,
        val_sequences_file=args.val_sequences,
        vocab_file=args.vocab,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_len=args.max_len,
        device=args.device
    )
