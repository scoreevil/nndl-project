"""
自研模型推理：为测试集生成描述
支持Model1/RNN、Model2/LSTM、Model3/注意力RNN、Model5/全Transformer
"""
import torch
import json
import time
from pathlib import Path
from typing import List, Dict
import numpy as np
from .dataset import FashionCaptionDataset
from .model_evaluator import ModelEvaluator


def load_model(model_type: str, checkpoint_path: str, vocab_file: str, device: str = None):
    """
    加载模型
    
    Args:
        model_type: 模型类型 ("model1", "model1b", "model2", "model5")
        checkpoint_path: checkpoint文件路径
        vocab_file: 词典文件路径
        device: 设备
        
    Returns:
        (model, idx2word, vocab_size)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # 加载词典
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
    vocab_size = vocab_data['vocab_size']
    
    # 加载模型checkpoint（先加载到CPU，避免设备不匹配问题）
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 添加项目根目录到路径，以便导入models
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    if model_type == "model5":
        from models.model5_full_transformer import FashionCaptionModelTransformer
        model = FashionCaptionModelTransformer(vocab_size=vocab_size)
    elif model_type == "model2":
        from models.model2_local_selfattn_attention_rnn import FashionCaptionModelAttention
        model = FashionCaptionModelAttention(vocab_size=vocab_size)
    elif model_type == "model1b":
        from models.model1b_cnn_2layer_lstm import FashionCaptionModelLSTM
        model = FashionCaptionModelLSTM(vocab_size=vocab_size)
    elif model_type == "model1":
        from models.model1_regular_cnn_6layer_rnn import FashionCaptionModel
        model = FashionCaptionModel(vocab_size=vocab_size)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载模型权重（确保所有tensor都在CPU上，避免设备不匹配）
    state_dict = checkpoint['model_state_dict']
    # 强制将所有tensor移动到CPU（无论原始设备是什么）
    state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    # 将模型移动到目标设备
    model = model.to(device)
    model.eval()
    
    # 验证模型设备（调试用）
    if torch.cuda.is_available() and device.type == 'cuda':
        first_param = next(model.parameters())
        if first_param.device.type != 'cuda':
            print(f"警告: 模型参数设备不匹配，重新移动到 {device}")
            model = model.to(device)
    
    return model, idx2word, vocab_size


def generate_captions(
    model_type: str,
    checkpoint_path: str,
    test_feature_file: str,
    test_sequences_file: str,
    vocab_file: str,
    output_file: str,
    num_samples: int = 100,
    max_len: int = 25,
    device: str = None
):
    """
    为测试集生成描述
    
    Args:
        model_type: 模型类型
        checkpoint_path: checkpoint文件路径
        test_feature_file: 测试集特征文件
        test_sequences_file: 测试集序列文件
        vocab_file: 词典文件路径
        output_file: 输出文件路径
        num_samples: 样本数量
        max_len: 最大生成长度
        device: 设备
    """
    print("="*60)
    print(f"使用 {model_type.upper()} 生成描述")
    print("="*60)
    
    # 设置设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # 加载模型
    print("\n加载模型...")
    model, idx2word, vocab_size = load_model(model_type, checkpoint_path, vocab_file, str(device))
    print(f"[OK] 模型已加载到设备: {device}")
    
    # 加载测试集
    print("\n加载测试集...")
    test_sequences_data = torch.load(test_sequences_file, map_location='cpu')
    # 检查文件结构，支持不同的键名
    if isinstance(test_sequences_data, dict):
        if 'test_sequences' in test_sequences_data:
            test_sequences = test_sequences_data['test_sequences']
        elif 'val_sequences' in test_sequences_data:
            # 如果没有test_sequences，使用val_sequences
            print("警告: 未找到test_sequences，使用val_sequences")
            test_sequences = test_sequences_data['val_sequences']
        else:
            raise KeyError(f"序列文件中未找到test_sequences或val_sequences，可用键: {list(test_sequences_data.keys())}")
    else:
        raise ValueError(f"序列文件格式错误，期望字典，得到: {type(test_sequences_data)}")
    test_dataset = FashionCaptionDataset(test_feature_file, test_sequences)
    
    # 限制样本数量
    num_samples = min(num_samples, len(test_dataset))
    print(f"测试集样本数: {len(test_dataset)}, 将处理: {num_samples}")
    
    # 生成描述
    print("\n开始生成描述...")
    results = []
    start_time = time.time()
    
    with torch.no_grad():
        for idx in range(num_samples):
            sample = test_dataset[idx]
            local_feat = sample['local_feat'].unsqueeze(0).to(device)
            
            sample_start = time.time()
            
            # 根据模型类型调用不同的generate方法
            if model_type == "model5":
                generated_sequence = model.generate(local_feat, max_len=max_len, temperature=0.7)
            elif model_type == "model2":
                generated_sequence, _ = model.generate(local_feat, max_len=max_len, 
                                                       temperature=0.7, return_attn=False)
            elif model_type == "model1b":
                generated_sequence = model.generate(local_feat, max_len=max_len, temperature=0.7)
            else:  # model1
                generated_sequence = model.generate(local_feat, max_len=max_len)
            
            sample_time = (time.time() - sample_start) * 1000  # 转换为毫秒
            
            # 转换为文本
            gen_seq = generated_sequence[0].cpu()
            if hasattr(model, 'postprocess_caption'):
                gen_words = model.postprocess_caption(gen_seq, idx2word)
            else:
                gen_words = []
                for word_idx in gen_seq.tolist():
                    if word_idx == 3:  # <END>
                        break
                    if word_idx not in [0, 1, 2, 3]:  # 不是特殊标记
                        word = idx2word.get(word_idx, f"<{word_idx}>")
                        gen_words.append(word)
            
            caption = ' '.join(gen_words) if gen_words else ''
            sample_id = f"{idx+1:03d}"
            
            results.append({
                'sample_id': sample_id,
                'caption': caption,
                'time_ms': sample_time
            })
            
            if (idx + 1) % 10 == 0:
                print(f"  已处理 {idx + 1}/{num_samples} 个样本")
    
    total_time = time.time() - start_time
    avg_time = total_time / num_samples * 1000 if num_samples > 0 else 0
    
    print(f"\n生成完成:")
    print(f"  总样本数: {num_samples}")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  平均耗时: {avg_time:.2f} ms/样本")
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"{result['sample_id']}\t{result['caption']}\n")
    
    print(f"\n结果已保存到: {output_path}")
    print("="*60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自研模型推理生成描述")
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["model1", "model1b", "model2", "model5"],
                       help="模型类型")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="模型checkpoint文件路径")
    parser.add_argument("--test_feature_file", type=str,
                       default="dataset/features/test_features.npz",
                       help="测试集特征文件")
    parser.add_argument("--test_sequences_file", type=str,
                       default="run/text_sequences.pt",
                       help="测试集序列文件（默认: run/text_sequences.pt）")
    parser.add_argument("--vocab_file", type=str, default="dataset/vocab.json",
                       help="词典文件路径")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="输出目录")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="样本数量")
    parser.add_argument("--max_len", type=int, default=25,
                       help="最大生成长度")
    parser.add_argument("--device", type=str, default=None,
                       help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 确定输出文件名
    output_files = {
        "model1": "results/self_model1_generated.txt",
        "model1b": "results/self_model2_generated.txt",  # Model1b对应Model2/LSTM
        "model2": "results/self_model3_generated.txt",  # Model2对应Model3/注意力RNN
        "model5": "results/self_model5_generated.txt"
    }
    
    output_file = output_files.get(args.model_type, 
                                   f"results/self_{args.model_type}_generated.txt")
    
    generate_captions(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        test_feature_file=args.test_feature_file,
        test_sequences_file=args.test_sequences_file,
        vocab_file=args.vocab_file,
        output_file=output_file,
        num_samples=args.num_samples,
        max_len=args.max_len,
        device=args.device
    )


if __name__ == "__main__":
    main()

