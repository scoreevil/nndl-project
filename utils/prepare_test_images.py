"""
准备测试集图像：从测试集中选择100个代表性样本，调整到224x224并保存
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from .data_loader import load_and_validate_dataset
from .dataset import FashionCaptionDataset


def prepare_test_images(
    captions_file: str = "dataset/captions.json",
    images_dir: str = "dataset/images",
    test_feature_file: str = "dataset/features/test_features.npz",
    test_sequences_file: str = "run/text_sequences.pt",
    output_dir: str = "dataset/test_images_224x224",
    num_samples: int = 100,
    target_size: tuple = (224, 224)
):
    """
    准备测试集图像
    
    Args:
        captions_file: captions.json文件路径
        images_dir: 图像目录路径
        test_feature_file: 测试集特征文件
        test_sequences_file: 测试集序列文件
        output_dir: 输出目录
        num_samples: 样本数量，默认100
        target_size: 目标尺寸，默认(224, 224)
    """
    print("="*60)
    print("准备测试集图像")
    print("="*60)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载测试集数据
    print("\n加载测试集数据...")
    _, _, test_data = load_and_validate_dataset(
        captions_file=captions_file,
        images_dir=images_dir,
        random_seed=42
    )
    
    # 加载测试集特征和序列
    print("\n加载测试集特征...")
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
    
    print(f"测试集总样本数: {len(test_data)}")
    print(f"测试集特征数: {len(test_dataset)}")
    
    # 选择代表性样本（均匀采样）
    num_available = min(len(test_data), len(test_dataset), num_samples)
    if num_available < num_samples:
        print(f"警告: 可用样本数({num_available})少于请求数({num_samples})，将使用所有可用样本")
    
    # 均匀采样索引
    indices = np.linspace(0, min(len(test_data), len(test_dataset)) - 1, num_available, dtype=int)
    
    print(f"\n开始处理 {len(indices)} 个样本...")
    
    success_count = 0
    failed_samples = []
    
    for idx, sample_idx in enumerate(indices):
        try:
            # 获取图像路径
            if sample_idx < len(test_data):
                img_path = Path(test_data[sample_idx]['img_path'])
            else:
                # 如果test_data不够，尝试从images目录查找
                # 这里需要根据实际情况调整
                print(f"警告: 样本索引 {sample_idx} 超出test_data范围")
                continue
            
            if not img_path.exists():
                print(f"警告: 图像文件不存在: {img_path}")
                failed_samples.append(sample_idx)
                continue
            
            # 加载并调整图像
            img = Image.open(img_path)
            img = img.convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # 保存为PNG格式，命名为sample_001.png ~ sample_100.png
            output_filename = f"sample_{idx+1:03d}.png"
            output_filepath = output_path / output_filename
            img.save(output_filepath, 'PNG')
            
            success_count += 1
            if (idx + 1) % 10 == 0:
                print(f"  已处理 {idx + 1}/{len(indices)} 个样本")
        
        except Exception as e:
            print(f"  处理样本 {sample_idx} 时出错: {e}")
            failed_samples.append(sample_idx)
    
    print(f"\n成功处理 {success_count}/{len(indices)} 个样本")
    if failed_samples:
        print(f"失败的样本索引: {failed_samples}")
    
    # 保存样本索引映射（用于后续评测）
    sample_mapping = {
        'indices': indices.tolist(),
        'failed_samples': failed_samples,
        'output_dir': str(output_path)
    }
    mapping_file = output_path / 'sample_mapping.json'
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(sample_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"\n样本映射已保存到: {mapping_file}")
    print(f"图像已保存到: {output_path}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="准备测试集图像")
    parser.add_argument("--captions_file", type=str, default="dataset/captions.json",
                       help="captions.json文件路径")
    parser.add_argument("--images_dir", type=str, default="dataset/images",
                       help="图像目录路径")
    parser.add_argument("--test_feature_file", type=str, 
                       default="dataset/features/test_features.npz",
                       help="测试集特征文件")
    parser.add_argument("--test_sequences_file", type=str,
                       default="run/text_sequences.pt",
                       help="测试集序列文件（默认: run/text_sequences.pt）")
    parser.add_argument("--output_dir", type=str, 
                       default="dataset/test_images_224x224",
                       help="输出目录")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="样本数量")
    
    args = parser.parse_args()
    
    prepare_test_images(
        captions_file=args.captions_file,
        images_dir=args.images_dir,
        test_feature_file=args.test_feature_file,
        test_sequences_file=args.test_sequences_file,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )

