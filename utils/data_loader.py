"""
DeepFashion-MultiModal数据集加载与有效性校验模块
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import random


def load_and_validate_dataset(
    captions_file: str,
    images_dir: str,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    加载并验证DeepFashion-MultiModal数据集
    
    Args:
        captions_file: captions.json文件路径
        images_dir: 图片根目录路径
        random_seed: 随机种子，默认42
    
    Returns:
        train_data: 训练集数据列表
        val_data: 验证集数据列表
        test_data: 测试集数据列表
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 使用pathlib处理路径，兼容Windows/Linux
    captions_path = Path(captions_file)
    images_path = Path(images_dir)
    
    # 读取captions.json文件
    try:
        with open(captions_path, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到文件: {captions_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析错误: {e}")
    except Exception as e:
        raise Exception(f"读取文件时发生错误: {e}")
    
    # 检查图片目录是否存在
    if not images_path.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")
    if not images_path.is_dir():
        raise ValueError(f"路径不是目录: {images_dir}")
    
    # 处理不同格式的captions数据
    valid_data = []
    
    # 格式1: 字典格式 {"image_id.jpg": "caption text"}
    if isinstance(captions_data, dict):
        for image_id, caption in captions_data.items():
            # 检查caption是否为空
            if not caption or (isinstance(caption, str) and caption.strip() == ""):
                continue
            
            # 构建图片完整路径
            img_path = images_path / image_id
            
            # 检查图片是否存在
            if not img_path.exists():
                continue
            
            # 将单个字符串转换为列表格式（统一输出格式）
            captions_list = [caption] if isinstance(caption, str) else caption
            
            # 确保captions是列表且不为空
            if isinstance(captions_list, list) and len(captions_list) > 0:
                valid_data.append({
                    "img_path": str(img_path),
                    "captions": captions_list
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


if __name__ == "__main__":
    # 测试代码
    try:
        captions_file = "dataset/captions.json"
        images_dir = "dataset/images"
        
        train_data, val_data, test_data = load_and_validate_dataset(
            captions_file=captions_file,
            images_dir=images_dir,
            random_seed=42
        )
        
        # 打印前几个样本示例
        print("\n训练集前3个样本示例:")
        for i, sample in enumerate(train_data[:3]):
            print(f"样本{i+1}:")
            print(f"  图片路径: {sample['img_path']}")
            print(f"  描述数量: {len(sample['captions'])}")
            print(f"  第一个描述: {sample['captions'][0][:100]}...")
            print()
            
    except Exception as e:
        print(f"错误: {e}")