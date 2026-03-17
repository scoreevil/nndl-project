#!/usr/bin/env python3
"""
检查缺失的图片文件
比较captions.json中记录的图片和实际存在的图片，找出缺失的文件
"""
import json
import os
from pathlib import Path
from typing import Set, List

def check_missing_images(captions_file: str, images_dir: str) -> List[str]:
    """
    检查缺失的图片文件
    
    Args:
        captions_file: captions.json文件路径
        images_dir: 图片目录路径
    
    Returns:
        missing_files: 缺失的文件列表
    """
    print("="*60)
    print("检查缺失的图片文件")
    print("="*60)
    
    # 读取captions.json
    print(f"\n读取 {captions_file}...")
    with open(captions_file, 'r', encoding='utf-8') as f:
        captions_data = json.load(f)
    
    # 提取所有图片文件名
    if isinstance(captions_data, dict):
        expected_images = set(captions_data.keys())
    elif isinstance(captions_data, list):
        expected_images = set()
        for item in captions_data:
            if isinstance(item, dict):
                img_name = item.get('image', item.get('image_id', item.get('img_path', '')))
                if img_name:
                    # 如果是完整路径，只取文件名
                    img_name = os.path.basename(img_name)
                    expected_images.add(img_name)
    else:
        raise ValueError(f"未知的captions.json格式: {type(captions_data)}")
    
    print(f"captions.json中记录的图片数量: {len(expected_images)}")
    
    # 检查实际存在的图片
    print(f"\n检查 {images_dir} 目录...")
    images_path = Path(images_dir)
    if not images_path.exists():
        print(f"错误: 目录不存在: {images_dir}")
        return []
    
    existing_images = set()
    for img_file in images_path.glob("*.jpg"):
        existing_images.add(img_file.name)
    for img_file in images_path.glob("*.jpeg"):
        existing_images.add(img_file.name)
    for img_file in images_path.glob("*.png"):
        existing_images.add(img_file.name)
    
    print(f"实际存在的图片数量: {len(existing_images)}")
    
    # 找出缺失的文件
    missing_images = expected_images - existing_images
    print(f"\n缺失的图片数量: {len(missing_images)}")
    
    if len(missing_images) > 0:
        print(f"\n缺失文件列表（前20个）:")
        for i, img_name in enumerate(sorted(missing_images)[:20], 1):
            print(f"  {i}. {img_name}")
        if len(missing_images) > 20:
            print(f"  ... 还有 {len(missing_images) - 20} 个文件")
        
        # 保存到文件
        missing_file = Path(images_dir).parent / "missing_images.txt"
        with open(missing_file, 'w', encoding='utf-8') as f:
            for img_name in sorted(missing_images):
                f.write(f"{img_name}\n")
        print(f"\n缺失文件列表已保存到: {missing_file}")
    else:
        print("\n✅ 所有图片文件都存在！")
    
    return sorted(missing_images)

if __name__ == "__main__":
    import sys
    
    captions_file = "dataset/captions.json"
    images_dir = "dataset/images"
    
    if len(sys.argv) > 1:
        captions_file = sys.argv[1]
    if len(sys.argv) > 2:
        images_dir = sys.argv[2]
    
    missing = check_missing_images(captions_file, images_dir)
    print(f"\n总计缺失: {len(missing)} 个文件")
