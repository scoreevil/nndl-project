"""
新数据集加载模块
用于加载NewDatasetCreate/annotations.json格式的数据
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple
import random


def load_new_dataset_annotations(
    annotations_file: str,
    images_dir: str,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    加载新数据集的标注文件（annotations.json格式）
    
    Args:
        annotations_file: annotations.json文件路径
        images_dir: 图片根目录路径
        random_seed: 随机种子，默认42
        
    Returns:
        train_data: 训练集数据列表
        val_data: 验证集数据列表
        test_data: 测试集数据列表
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 使用pathlib处理路径
    annotations_path = Path(annotations_file)
    images_path = Path(images_dir)
    
    # 读取annotations.json文件
    try:
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到文件: {annotations_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析错误: {e}")
    except Exception as e:
        raise Exception(f"读取文件时发生错误: {e}")
    
    # 检查图片目录是否存在
    if not images_path.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")
    if not images_path.is_dir():
        raise ValueError(f"路径不是目录: {images_dir}")
    
    # 处理标注数据，转换为标准格式
    valid_data = []
    
    for ann in annotations:
        image_path_str = ann.get('image_path', '')
        if not image_path_str:
            continue
        
        # 处理路径（可能是绝对路径或相对路径）
        image_path = Path(image_path_str)
        if not image_path.is_absolute():
            # 如果是相对路径，相对于images_dir
            image_path = images_path / image_path.name
        
        # 检查图片是否存在
        if not image_path.exists():
            continue
        
        # 提取描述（合并所有描述）
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
            valid_data.append({
                "img_path": str(image_path),
                "captions": [merged_desc.strip()]
            })
    
    # 打乱数据
    random.shuffle(valid_data)
    
    # 按7:2:1比例划分数据集
    total_samples = len(valid_data)
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.2)
    
    train_data = valid_data[:train_size]
    val_data = valid_data[train_size:train_size + val_size]
    test_data = valid_data[train_size + val_size:]
    
    # 打印各数据集的样本数量
    print(f"总有效样本数: {total_samples}条")
    print(f"训练集: {len(train_data)}条")
    print(f"验证集: {len(val_data)}条")
    print(f"测试集: {len(test_data)}条")
    
    return train_data, val_data, test_data

