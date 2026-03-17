"""
图像处理工具函数
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import os


def crop_and_pad(image_path: str, box_coords: Tuple[int, int, int, int], 
                 output_size: Tuple[int, int] = (224, 224),
                 output_dir: Optional[str] = None) -> Tuple[str, Image.Image]:
    """
    裁剪图像选框区域并填充至指定尺寸
    
    Args:
        image_path: 原始图像路径
        box_coords: 选框坐标 (x1, y1, x2, y2)
        output_size: 输出图像尺寸，默认(224, 224)
        output_dir: 输出目录
        
    Returns:
        (输出图像路径, PIL图像对象)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box_coords
    
    # 修正坐标
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"无效的选框坐标: ({x1}, {y1}, {x2}, {y2})")
    
    # 裁剪
    cropped = img[y1:y2, x1:x2]
    
    # 计算缩放比例
    crop_h, crop_w = cropped.shape[:2]
    target_w, target_h = output_size
    
    scale = min(target_w / crop_w, target_h / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    
    # 调整大小
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 创建白色背景
    padded = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    
    # 居中放置
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    
    padded[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
    
    # 转换为PIL图像
    pil_image = Image.fromarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
    
    # 生成输出路径
    if output_dir is None:
        from .config import TEMP_CROPPED_DIR
        output_dir = TEMP_CROPPED_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    original_name = Path(image_path).stem
    output_path = Path(output_dir) / f"{original_name}_cropped.jpg"
    
    pil_image.save(output_path, "JPEG", quality=95)
    
    return str(output_path), pil_image


def get_image_info(image_path: str) -> dict:
    """
    获取图像信息
    
    Args:
        image_path: 图像路径
        
    Returns:
        图像信息字典
    """
    try:
        img = Image.open(image_path)
        return {
            "width": img.size[0],
            "height": img.size[1],
            "mode": img.mode,
            "format": img.format
        }
    except Exception as e:
        raise ValueError(f"无法读取图像信息: {e}")


def validate_box_coords(image_path: str, box_coords: Tuple[int, int, int, int]) -> bool:
    """
    验证选框坐标是否有效
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box_coords
        
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return False
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return False
        
        return True
    except Exception:
        return False

