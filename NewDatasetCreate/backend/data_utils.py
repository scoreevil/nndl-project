"""
数据管理工具函数
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import os


def load_annotations(json_path: str) -> List[Dict]:
    """加载标注数据"""
    if not os.path.exists(json_path):
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'annotations' in data:
            return data['annotations']
        else:
            return []
    except Exception as e:
        print(f"加载标注数据失败: {e}")
        return []


def save_annotation(annotation_data: Dict, json_path: str) -> None:
    """保存单张图的标注数据"""
    annotations = load_annotations(json_path)
    
    image_path = annotation_data.get('image_path')
    if not image_path:
        raise ValueError("annotation_data 必须包含 'image_path' 字段")
    
    found = False
    for i, ann in enumerate(annotations):
        if ann.get('image_path') == image_path:
            annotations[i] = annotation_data
            found = True
            break
    
    if not found:
        annotations.append(annotation_data)
    
    os.makedirs(Path(json_path).parent, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)


def get_annotation_status(json_path: str) -> Dict[str, bool]:
    """获取所有图像的标注状态"""
    annotations = load_annotations(json_path)
    status = {}
    
    for ann in annotations:
        image_path = ann.get('image_path')
        if image_path:
            has_box = 'box_coords' in ann and ann['box_coords'] is not None
            use_lmm_as_first = ann.get('use_lmm_as_first', False)
            # 根据use_lmm_as_first判断第一个描述是否存在
            first_desc = ann.get('lmm_clothing_desc') if use_lmm_as_first else ann.get('self_desc')
            has_desc = bool(first_desc)
            status[image_path] = has_box and has_desc
    
    return status


def export_dataset(json_path: str, npz_path: str) -> None:
    """将JSON标注记录导出为NPZ格式"""
    annotations = load_annotations(json_path)
    
    valid_annotations = []
    for ann in annotations:
        use_lmm_as_first = ann.get('use_lmm_as_first', False)
        # 根据use_lmm_as_first判断第一个描述
        first_desc = ann.get('lmm_clothing_desc') if use_lmm_as_first else ann.get('self_desc')
        
        if (ann.get('box_coords') and 
            first_desc and 
            ann.get('image_path')):
            valid_annotations.append(ann)
    
    if len(valid_annotations) == 0:
        raise ValueError("没有有效的标注数据可导出")
    
    image_paths = []
    box_coords_list = []
    self_descs = []
    lmm_clothing_descs = []
    lmm_bg1_descs = []
    lmm_bg2_descs = []
    merged_descs = []
    
    for ann in valid_annotations:
        image_paths.append(ann.get('image_path', ''))
        box_coords_list.append(ann.get('box_coords', [0, 0, 0, 0]))
        self_descs.append(ann.get('self_desc', ''))
        lmm_clothing_descs.append(ann.get('lmm_clothing_desc', ''))
        lmm_bg1_descs.append(ann.get('lmm_bg1', ''))
        lmm_bg2_descs.append(ann.get('lmm_bg2', ''))
        
        merged = ann.get('self_desc', '')
        if ann.get('lmm_clothing_desc'):
            merged += f" {ann['lmm_clothing_desc']}"
        if ann.get('lmm_bg1'):
            merged += f" {ann['lmm_bg1']}"
        if ann.get('lmm_bg2'):
            merged += f" {ann['lmm_bg2']}"
        merged_descs.append(merged.strip())
    
    os.makedirs(Path(npz_path).parent, exist_ok=True)
    np.savez(
        npz_path,
        image_paths=np.array(image_paths, dtype=object),
        box_coords=np.array(box_coords_list, dtype=np.int32),
        self_desc=np.array(self_descs, dtype=object),
        lmm_clothing_desc=np.array(lmm_clothing_descs, dtype=object),
        lmm_bg1=np.array(lmm_bg1_descs, dtype=object),
        lmm_bg2=np.array(lmm_bg2_descs, dtype=object),
        merged_desc=np.array(merged_descs, dtype=object)
    )


def get_annotation_count(json_path: str) -> int:
    """获取已标注的数量"""
    status = get_annotation_status(json_path)
    return sum(1 for v in status.values() if v)

