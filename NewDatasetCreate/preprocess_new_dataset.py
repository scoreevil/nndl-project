"""
新数据集预处理脚本
使用与text_processor.py相同的预处理流程，min_freq=3（出现3次就计入词表）
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.text_processor import TextProcessor


def extract_captions_from_annotations(annotations: List[Dict]) -> List[str]:
    """
    从标注数据中提取所有描述文本
    
    Args:
        annotations: 标注数据列表
        
    Returns:
        描述文本列表（用于构建词表）
    """
    captions = []
    
    for ann in annotations:
        use_lmm_as_first = ann.get('use_lmm_as_first', False)
        
        # 根据use_lmm_as_first决定第一个描述
        if use_lmm_as_first:
            first_desc = ann.get('lmm_clothing_desc', '')
        else:
            first_desc = ann.get('self_desc', '')
        
        # 合并所有描述（第一个描述 + LMMs描述 + 背景描述）
        merged_desc = first_desc
        if not use_lmm_as_first and ann.get('lmm_clothing_desc'):
            merged_desc += f" {ann['lmm_clothing_desc']}"
        elif use_lmm_as_first and ann.get('self_desc'):
            merged_desc += f" {ann['self_desc']}"
        
        if ann.get('lmm_bg1'):
            merged_desc += f" {ann['lmm_bg1']}"
        if ann.get('lmm_bg2'):
            merged_desc += f" {ann['lmm_bg2']}"
        
        # 只添加非空描述
        if merged_desc.strip():
            captions.append(merged_desc.strip())
    
    return captions


def convert_annotations_to_data_format(annotations: List[Dict]) -> List[Dict]:
    """
    将标注数据转换为与原始数据集兼容的格式（包含'captions'字段）
    
    Args:
        annotations: 标注数据列表
        
    Returns:
        转换后的数据列表，每个元素包含'captions'字段
    """
    data_list = []
    
    for ann in annotations:
        use_lmm_as_first = ann.get('use_lmm_as_first', False)
        
        # 根据use_lmm_as_first决定第一个描述
        if use_lmm_as_first:
            first_desc = ann.get('lmm_clothing_desc', '')
        else:
            first_desc = ann.get('self_desc', '')
        
        # 合并所有描述（第一个描述 + LMMs描述 + 背景描述）
        merged_desc = first_desc
        if not use_lmm_as_first and ann.get('lmm_clothing_desc'):
            merged_desc += f" {ann['lmm_clothing_desc']}"
        elif use_lmm_as_first and ann.get('self_desc'):
            merged_desc += f" {ann['self_desc']}"
        
        if ann.get('lmm_bg1'):
            merged_desc += f" {ann['lmm_bg1']}"
        if ann.get('lmm_bg2'):
            merged_desc += f" {ann['lmm_bg2']}"
        
        # 转换为标准格式（包含captions字段）
        data_item = {
            'image_path': ann.get('image_path', ''),
            'box_coords': ann.get('box_coords', []),
            'captions': [merged_desc.strip()] if merged_desc.strip() else [],
            'self_desc': ann.get('self_desc', ''),
            'lmm_clothing_desc': ann.get('lmm_clothing_desc', ''),
            'lmm_bg1': ann.get('lmm_bg1', ''),
            'lmm_bg2': ann.get('lmm_bg2', ''),
            'use_lmm_as_first': use_lmm_as_first,
            'status': ann.get('status', 'pending')
        }
        data_list.append(data_item)
    
    return data_list


def preprocess_new_dataset(
    annotations_file: str,
    vocab_file: str,
    sequences_file: str,
    max_len: int = 25,
    min_freq: int = 3
) -> Tuple[TextProcessor, torch.Tensor]:
    """
    预处理新数据集
    
    Args:
        annotations_file: 标注JSON文件路径
        vocab_file: 输出词表文件路径
        sequences_file: 输出序列文件路径
        max_len: 最大序列长度
        min_freq: 词的最小出现频率（默认3，出现3次就计入词表）
        
    Returns:
        processor: 文本处理器对象
        sequences: 文本序列张量 (N, max_len)
    """
    print("="*60)
    print("新数据集预处理")
    print("="*60)
    
    # 步骤1: 加载标注数据
    print("\n" + "="*50)
    print("步骤1: 加载标注数据")
    print("="*50)
    annotations_path = Path(annotations_file)
    if not annotations_path.exists():
        raise FileNotFoundError(f"标注文件不存在: {annotations_file}")
    
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"加载了 {len(annotations)} 条标注记录")
    
    # 步骤2: 提取所有描述（用于构建词表）
    print("\n" + "="*50)
    print("步骤2: 提取所有描述")
    print("="*50)
    all_captions = extract_captions_from_annotations(annotations)
    print(f"提取了 {len(all_captions)} 条有效描述")
    
    # 统计有效标注数量
    valid_count = sum(1 for ann in annotations 
                     if ann.get('box_coords') and 
                     (ann.get('self_desc') or ann.get('lmm_clothing_desc')))
    print(f"有效标注数量: {valid_count} (有选框且有描述)")
    
    # 步骤3: 创建文本处理器并构建词表
    print("\n" + "="*50)
    print("步骤3: 构建词表 (min_freq={})".format(min_freq))
    print("="*50)
    processor = TextProcessor(min_freq=min_freq)
    processor.build_vocab(all_captions)
    
    # 保存词表
    vocab_path = Path(vocab_file)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    processor.save_vocab(vocab_file)
    print(f"词表已保存到: {vocab_file}")
    
    # 步骤4: 打印序列化示例
    print("\n" + "="*50)
    print("步骤4: 序列化示例")
    print("="*50)
    if len(all_captions) > 0:
        processor.print_example(all_captions[0], max_len=max_len)
    
    # 步骤5: 转换数据格式并批量处理
    print("\n" + "="*50)
    print("步骤5: 批量处理数据集")
    print("="*50)
    data_list = convert_annotations_to_data_format(annotations)
    sequences = processor.batch_process(data_list, max_len=max_len)
    print(f"序列形状: {sequences.shape}")
    
    # 步骤6: 保存序列
    print("\n" + "="*50)
    print("步骤6: 保存序列")
    print("="*50)
    sequences_path = Path(sequences_file)
    sequences_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sequences, sequences_file)
    print(f"序列已保存到: {sequences_file}")
    
    # 统计信息
    print("\n" + "="*50)
    print("预处理统计信息")
    print("="*50)
    print(f"总标注数: {len(annotations)}")
    print(f"有效描述数: {len(all_captions)}")
    print(f"词表大小: {processor.vocab_size}")
    print(f"序列形状: {sequences.shape}")
    print(f"最大序列长度: {max_len}")
    print(f"最小词频: {min_freq}")
    
    return processor, sequences


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='预处理新数据集')
    parser.add_argument(
        '--annotations_file',
        type=str,
        default='NewDatasetCreate/annotations.json',
        help='标注JSON文件路径'
    )
    parser.add_argument(
        '--vocab_file',
        type=str,
        default='newdataset/vocab.json',
        help='输出词表文件路径'
    )
    parser.add_argument(
        '--sequences_file',
        type=str,
        default='newdataset/text_sequences.pt',
        help='输出序列文件路径'
    )
    parser.add_argument(
        '--max_len',
        type=int,
        default=25,
        help='最大序列长度'
    )
    parser.add_argument(
        '--min_freq',
        type=int,
        default=3,
        help='词的最小出现频率（出现min_freq次就计入词表）'
    )
    
    args = parser.parse_args()
    
    # 执行预处理
    processor, sequences = preprocess_new_dataset(
        annotations_file=args.annotations_file,
        vocab_file=args.vocab_file,
        sequences_file=args.sequences_file,
        max_len=args.max_len,
        min_freq=args.min_freq
    )
    
    print("\n" + "="*60)
    print("✓ 新数据集预处理完成！")
    print("="*60)

