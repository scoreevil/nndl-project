"""
DeepFashion-MultiModal数据集的PyTorch Dataset和DataLoader封装
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict
import numpy as np


class FashionCaptionDataset(Dataset):
    """Fashion图像-文本数据集"""
    
    def __init__(self, feature_file: str, caption_sequences: torch.Tensor):
        """
        初始化数据集
        
        Args:
            feature_file: npz特征文件路径（如train_features.npz）
            caption_sequences: 文本序列张量，形状为(N, max_len)，dtype=torch.long
        """
        self.feature_file = feature_file
        self.caption_sequences = caption_sequences
        
        # 加载npz文件
        try:
            data = np.load(feature_file, allow_pickle=True)
            self.global_feats = data['global_feats']
            self.local_feats = data['local_feats']
            
            # 验证数据一致性
            num_features = self.global_feats.shape[0]
            num_captions = caption_sequences.shape[0]
            
            if num_features != num_captions:
                raise ValueError(
                    f"特征数量({num_features})与文本序列数量({num_captions})不匹配！"
                )
            
            self.length = num_features
            
            print(f"成功加载数据集: {feature_file}")
            print(f"  样本数量: {self.length}")
            print(f"  全局特征形状: {self.global_feats.shape}")
            print(f"  局部特征形状: {self.local_feats.shape}")
            print(f"  文本序列形状: {self.caption_sequences.shape}")
        except Exception as e:
            raise RuntimeError(f"无法打开npz文件 {feature_file}: {e}")
    
    def __len__(self) -> int:
        """返回样本总数"""
        return self.length
    
    def __getitem__(self, idx: int) -> dict:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            包含以下键的字典：
            - "global_feat": 全局特征 (torch.float32, 形状(2048,))
            - "local_feat": 局部特征 (torch.float32, 形状(2048, 7, 7))
            - "caption": 文本序列 (torch.long, 形状(max_len,))
        """
        # 从npz文件读取特征
        global_feat = torch.from_numpy(self.global_feats[idx]).float()  # (2048,)
        local_feat = torch.from_numpy(self.local_feats[idx]).float()   # (2048, 7, 7)
        
        # 获取文本序列
        caption = self.caption_sequences[idx].long()  # (max_len,)
        
        return {
            "global_feat": global_feat,
            "local_feat": local_feat,
            "caption": caption
        }


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    构建DataLoader
    
    Args:
        dataset: Dataset实例
        batch_size: 批量大小
        shuffle: 是否打乱数据
        num_workers: 多线程读取的进程数（可根据CPU核心数调整）
        pin_memory: 是否使用pin_memory（加速GPU传输）
    
    Returns:
        DataLoader实例
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # 不丢弃最后一个不完整的batch
    )
    
    return dataloader


def create_dataloaders(
    train_feature_file: str,
    val_feature_file: str,
    test_feature_file: str,
    train_sequences: torch.Tensor,
    val_sequences: torch.Tensor,
    test_sequences: torch.Tensor,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> tuple:
    """
    创建训练、验证、测试集的DataLoader
    
    Args:
        train_feature_file: 训练集特征文件路径（.npz格式）
        val_feature_file: 验证集特征文件路径（.npz格式）
        test_feature_file: 测试集特征文件路径（.npz格式）
        train_sequences: 训练集文本序列张量
        val_sequences: 验证集文本序列张量
        test_sequences: 测试集文本序列张量
        batch_size: 批量大小
        num_workers: 多线程读取的进程数
        pin_memory: 是否使用pin_memory
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 创建Dataset
    train_dataset = FashionCaptionDataset(train_feature_file, train_sequences)
    val_dataset = FashionCaptionDataset(val_feature_file, val_sequences)
    test_dataset = FashionCaptionDataset(test_feature_file, test_sequences)
    
    # 创建DataLoader
    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不打乱
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = get_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不打乱
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


class ExpandedFashionCaptionDataset(Dataset):
    """
    扩展的Fashion数据集：支持每个图像的所有描述
    用于处理扩展后的数据（每个图像的所有描述）
    """
    
    def __init__(self, feature_file: str, all_captions_data: List[Dict], text_processor, max_len: int = 20):
        """
        初始化扩展数据集
        
        Args:
            feature_file: npz特征文件路径
            all_captions_data: 数据列表，每个元素包含'captions'字段（描述列表）
            text_processor: 文本处理器对象（TextProcessor）
            max_len: 最大序列长度
        """
        self.feature_file = feature_file
        self.all_captions_data = all_captions_data
        self.text_processor = text_processor
        self.max_len = max_len
        
        # 加载npz文件
        try:
            data = np.load(feature_file, allow_pickle=True)
            self.global_feats = data['global_feats']
            self.local_feats = data['local_feats']
        except Exception as e:
            raise RuntimeError(f"无法打开npz文件 {feature_file}: {e}")
        
        # 构建特征索引到描述索引的映射
        # 每个图像可能有多个描述，需要为每个描述创建一个样本
        self.feature_indices = []  # 特征索引列表
        self.caption_indices = []  # 描述索引列表（在captions列表中的索引）
        
        for feat_idx, item in enumerate(all_captions_data):
            captions = item.get('captions', [])
            if len(captions) == 0:
                # 如果没有描述，创建一个空描述样本
                self.feature_indices.append(feat_idx)
                self.caption_indices.append(-1)  # -1表示空描述
            else:
                # 为每个描述创建一个样本
                for caption_idx in range(len(captions)):
                    self.feature_indices.append(feat_idx)
                    self.caption_indices.append(caption_idx)
        
        self.length = len(self.feature_indices)
        
        print(f"成功加载扩展数据集: {feature_file}")
        print(f"  原始图像数量: {len(all_captions_data)}")
        print(f"  扩展后样本数量: {self.length}")
        print(f"  特征数量: {self.local_feats.shape[0]}")
    
    def __len__(self) -> int:
        """返回样本总数"""
        return self.length
    
    def __getitem__(self, idx: int) -> dict:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            包含以下键的字典：
            - "global_feat": 全局特征 (torch.float32, 形状(2048,))
            - "local_feat": 局部特征 (torch.float32, 形状(2048, 7, 7))
            - "caption": 文本序列 (torch.long, 形状(max_len,))
        """
        # 获取特征索引
        feat_idx = self.feature_indices[idx]
        
        # 从npz文件读取特征
        global_feat = torch.from_numpy(self.global_feats[feat_idx]).float()  # (2048,)
        local_feat = torch.from_numpy(self.local_feats[feat_idx]).float()   # (2048, 7, 7)
        
        # 获取描述并转换为序列
        caption_idx = self.caption_indices[idx]
        if caption_idx == -1:
            # 空描述，创建PAD填充序列
            caption = torch.zeros(self.max_len, dtype=torch.long)
            caption[0] = 2  # <START>
            caption[1] = 3  # <END>
        else:
            item = self.all_captions_data[feat_idx]
            captions = item.get('captions', [])
            caption_text = captions[caption_idx]
            # 使用text_processor将文本转换为序列
            caption_seq = self.text_processor.text_to_sequence(caption_text, max_len=self.max_len)
            caption = torch.tensor(caption_seq, dtype=torch.long)
        
        return {
            "global_feat": global_feat,
            "local_feat": local_feat,
            "caption": caption
        }


if __name__ == "__main__":
    # 测试代码
    from utils.data_loader import load_and_validate_dataset
    from utils.text_processor import build_vocab_and_process
    
    print("="*50)
    print("测试Dataset和DataLoader")
    print("="*50)
    
    # 加载数据集
    print("\n1. 加载数据集...")
    train_data, val_data, test_data = load_and_validate_dataset(
        captions_file="dataset/captions.json",
        images_dir="dataset/images",
        random_seed=42
    )
    
    # 文本预处理（使用小批量测试）
    print("\n2. 文本预处理...")
    processor, train_seq, val_seq, test_seq = build_vocab_and_process(
        train_data=train_data[:100],  # 只用前100个样本测试
        val_data=val_data[:20],
        test_data=test_data[:20],
        max_len=20,
        min_freq=3
    )
    
    # 注意：这里需要先提取特征才能测试Dataset
    # 假设特征文件已存在
    print("\n3. 创建Dataset和DataLoader...")
    try:
        train_dataset = FashionCaptionDataset(
            feature_file="dataset/features/train_features.npz",
            caption_sequences=train_seq
        )
        
        train_loader = get_dataloader(
            dataset=train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,  # Windows上可能不支持多进程，设为0
            pin_memory=False  # CPU模式下设为False
        )
        
        # 迭代一个batch
        print("\n4. 迭代一个batch...")
        for batch in train_loader:
            print(f"global_feat: {batch['global_feat'].shape}")
            print(f"local_feat: {batch['local_feat'].shape}")
            print(f"caption: {batch['caption'].shape}")
            print(f"\n数据类型验证:")
            print(f"  global_feat dtype: {batch['global_feat'].dtype} (应为torch.float32)")
            print(f"  local_feat dtype: {batch['local_feat'].dtype} (应为torch.float32)")
            print(f"  caption dtype: {batch['caption'].dtype} (应为torch.int64/long)")
            break
        
        print("\n✓ Dataset和DataLoader测试通过！")
        
    except FileNotFoundError:
        print("\n⚠ 特征文件不存在，请先运行特征提取程序")
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()