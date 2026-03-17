"""
DeepFashion-MultiModal数据集图像特征提取模块
使用预训练ResNet50提取全局特征和局部特征
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm


class ResNetFeatureExtractor:
    """使用ResNet50提取图像全局特征和局部特征"""
    
    def __init__(self, device: Optional[str] = None, batch_size: int = 32):
        """
        初始化特征提取器
        
        Args:
            device: 设备类型 ('cuda' 或 'cpu')，如果为None则自动选择
            batch_size: 批量处理大小
        """
        # 自动选择设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 加载预训练ResNet50
        try:
            # 新版本torchvision使用weights参数
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except AttributeError:
            # 旧版本torchvision使用pretrained参数
            self.model = models.resnet50(pretrained=True)
        
        # 冻结所有权重
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 设置为评估模式
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # 移除最后两层（avgpool和fc），用于提取局部特征
        self.local_feature_model = nn.Sequential(*list(self.model.children())[:-2])
        
        # 移除fc层，保留avgpool，用于提取全局特征
        self.global_feature_model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.batch_size = batch_size
        
        # 图像预处理：Resize到224x224，归一化（ImageNet均值和标准差）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_image(self, img_path: str) -> Optional[torch.Tensor]:
        """
        加载并预处理单张图片
        
        Args:
            img_path: 图片路径
            
        Returns:
            预处理后的图片张量，如果加载失败返回None
        """
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor
        except Exception as e:
            print(f"警告: 无法加载图片 {img_path}: {e}")
            return None
    
    def extract_features_batch(self, img_paths: List[str]) -> tuple:
        """
        批量提取特征
        
        Args:
            img_paths: 图片路径列表
            
        Returns:
            global_feats: 全局特征 (batch_size, 2048)
            local_feats: 局部特征 (batch_size, 2048, 7, 7)
        """
        # 加载图片
        img_tensors = []
        valid_indices = []
        
        for i, img_path in enumerate(img_paths):
            img_tensor = self.load_image(img_path)
            if img_tensor is not None:
                img_tensors.append(img_tensor)
                valid_indices.append(i)
        
        if len(img_tensors) == 0:
            return None, None
        
        # 堆叠为batch
        batch = torch.stack(img_tensors).to(self.device)
        
        # 提取特征（关闭梯度计算）
        with torch.no_grad():
            # 提取局部特征（去掉最后两层）
            local_feats = self.local_feature_model(batch)  # (batch, 2048, 7, 7)
            
            # 提取全局特征（去掉fc层，使用avgpool）
            global_feats = self.global_feature_model(batch)  # (batch, 2048, 1, 1)
            global_feats = global_feats.squeeze(-1).squeeze(-1)  # (batch, 2048)
        
        return global_feats.cpu().numpy(), local_feats.cpu().numpy(), valid_indices
    
    def extract_features_from_data(
        self,
        data_list: List[Dict],
        output_file: str
    ):
        """
        从数据列表中提取特征并保存到npz文件
        
        Args:
            data_list: 数据列表，每个元素包含 'img_path' 和 'captions'
            output_file: 输出npz文件路径
        """
        total_samples = len(data_list)
        print(f"\n开始提取特征，共 {total_samples} 张图片...")
        
        # 收集所有图片路径和ID
        img_paths = [item['img_path'] for item in data_list]
        img_ids = [Path(item['img_path']).name for item in data_list]
        
        # 存储所有特征
        all_global_feats = []
        all_local_feats = []
        valid_img_ids = []
        
        # 批量处理
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="提取特征进度"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_samples)
            batch_paths = img_paths[start_idx:end_idx]
            batch_ids = img_ids[start_idx:end_idx]
            
            # 提取特征
            global_feats, local_feats, valid_indices = self.extract_features_batch(batch_paths)
            
            if global_feats is not None and local_feats is not None:
                # 只保存成功加载的图片特征
                # valid_indices是batch内的索引，需要映射回原始batch_ids
                for feat_idx, batch_idx_in_batch in enumerate(valid_indices):
                    all_global_feats.append(global_feats[feat_idx])
                    all_local_feats.append(local_feats[feat_idx])
                    valid_img_ids.append(batch_ids[batch_idx_in_batch])
            
            # 打印进度
            processed = len(all_global_feats)
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                print(f"已处理 {processed}/{total_samples} 张图片")
        
        # 转换为numpy数组
        all_global_feats = np.array(all_global_feats)
        all_local_feats = np.array(all_local_feats)
        
        print(f"\n成功提取 {len(all_global_feats)} 张图片的特征")
        print(f"全局特征形状: {all_global_feats.shape}")
        print(f"局部特征形状: {all_local_feats.shape}")
        
        # 保存到npz文件
        # 确保文件扩展名为.npz
        if not output_file.endswith('.npz'):
            output_file = output_file.replace('.h5', '.npz')
            if not output_file.endswith('.npz'):
                output_file = output_file + '.npz'
        
        print(f"\n保存特征到 {output_file}...")
        
        np.savez_compressed(
            output_file,
            global_feats=all_global_feats.astype(np.float32),
            local_feats=all_local_feats.astype(np.float32),
            img_ids=np.array(valid_img_ids, dtype=object)
        )
        
        print(f"特征已成功保存到 {output_file}")
        print(f"  - global_feats: {all_global_feats.shape}")
        print(f"  - local_feats: {all_local_feats.shape}")
        print(f"  - img_ids: {len(valid_img_ids)} 个")


def extract_features_for_datasets(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    output_dir: str = "features",
    batch_size: int = 32,
    device: Optional[str] = None
):
    """
    为训练集、验证集、测试集提取特征
    
    Args:
        train_data: 训练集数据列表
        val_data: 验证集数据列表
        test_data: 测试集数据列表
        output_dir: 输出目录
        batch_size: 批量大小
        device: 设备类型
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 创建特征提取器
    extractor = ResNetFeatureExtractor(device=device, batch_size=batch_size)
    
    # 提取训练集特征
    print("\n" + "="*50)
    print("提取训练集特征")
    print("="*50)
    extractor.extract_features_from_data(
        train_data,
        str(output_path / "train_features.npz")
    )
    
    # 提取验证集特征
    print("\n" + "="*50)
    print("提取验证集特征")
    print("="*50)
    extractor.extract_features_from_data(
        val_data,
        str(output_path / "val_features.npz")
    )
    
    # 提取测试集特征
    print("\n" + "="*50)
    print("提取测试集特征")
    print("="*50)
    extractor.extract_features_from_data(
        test_data,
        str(output_path / "test_features.npz")
    )
    
    print("\n" + "="*50)
    print("所有特征提取完成！")
    print("="*50)


if __name__ == "__main__":
    # 测试代码
    from data_loader import load_and_validate_dataset
    
    # 加载数据集
    print("加载数据集...")
    train_data, val_data, test_data = load_and_validate_dataset(
        captions_file="dataset/captions.json",
        images_dir="dataset/images",
        random_seed=42
    )
    
    # 提取特征（使用小批量测试）
    print("\n开始提取特征...")
    extract_features_for_datasets(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        output_dir="dataset/features",
        batch_size=32,
        device=None  # 自动选择设备
    )