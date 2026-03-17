"""
为新数据集提取图像特征
"""
import sys
import os
from pathlib import Path
from typing import List, Dict
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from utils.load_new_dataset import load_new_dataset_annotations
from utils.feature_extractor import extract_features_for_datasets


def main():
    """为新数据集提取特征"""
    print("="*60)
    print("新数据集特征提取")
    print("="*60)
    
    # 配置路径
    annotations_file = PROJECT_ROOT / "newdataset" / "annotations.json"
    images_dir = PROJECT_ROOT / "newdataset" / "images"
    output_dir = PROJECT_ROOT / "newdataset" / "features"
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    print("\n加载数据集...")
    train_data, val_data, test_data = load_new_dataset_annotations(
        annotations_file=str(annotations_file),
        images_dir=str(images_dir),
        random_seed=42
    )
    
    # 提取特征
    print("\n开始提取图像特征...")
    extract_features_for_datasets(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        output_dir=str(output_dir),
        batch_size=32,
        device=None  # 自动选择设备
    )
    
    print("\n" + "="*60)
    print("✓ 特征提取完成！")
    print("="*60)
    print(f"特征文件保存在: {output_dir}")
    print(f"  - train_features.npz")
    print(f"  - val_features.npz")
    print(f"  - test_features.npz")


if __name__ == "__main__":
    main()

