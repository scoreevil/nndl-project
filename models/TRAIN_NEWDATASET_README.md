# 新数据集训练说明

## 概述

本训练脚本基于 `model2_local_selfattn_attention_rnn.py` 的架构，用于训练新数据集。新数据集的特点：
- **描述包含服饰和背景信息**：合并了第一个描述、LMMs描述和背景描述
- **词表大小**：723（min_freq=3）
- **数据集规模**：1000条标注（训练集700，验证集200，测试集100）

## 模型架构

使用 `FashionCaptionModelAttention` 模型，包含：
1. **增强型CNN编码器**：局部特征自注意力层（8头自注意力）
2. **注意力增强LSTM解码器**：标准注意力机制（LSTM隐藏状态与局部特征的注意力）

## 文件结构

```
newdataset/
├── annotations.json          # 标注文件
├── images/                   # 图像目录
├── vocab.json                # 词表文件（min_freq=3）
├── text_sequences.pt         # 文本序列文件
└── features/
    ├── train_features.npz    # 训练集特征
    ├── val_features.npz      # 验证集特征
    └── test_features.npz     # 测试集特征
```

## 使用步骤

### 1. 预处理数据集（如果尚未完成）

```bash
cd /root/autodl-tmp
python3 NewDatasetCreate/preprocess_new_dataset.py \
    --annotations_file NewDatasetCreate/annotations.json \
    --vocab_file newdataset/vocab.json \
    --sequences_file newdataset/text_sequences.pt \
    --max_len 25 \
    --min_freq 3
```

### 2. 提取图像特征（如果尚未完成）

```bash
cd /root/autodl-tmp
python3 utils/extract_new_dataset_features.py
```

### 3. 开始训练

**方式1：使用训练脚本（推荐）**
```bash
cd /root/autodl-tmp
bash run/train_model2_newdataset.sh
```

**方式2：直接运行Python脚本**
```bash
cd /root/autodl-tmp
python3 models/train_model2_newdataset.py
```

## 训练参数

- **模型超参数**：
  - 词嵌入维度：256
  - LSTM隐藏状态维度：512
  
- **训练超参数**：
  - 批次大小：16
  - 训练轮数：50
  - 初始学习率：1e-4
  - Teacher-Forcing比例：0.8
  - 标签平滑：0.1
  - 最大序列长度：25

- **优化策略**：
  - 优化器：AdamW
  - 学习率调度：ReduceLROnPlateau
  - 早停机制：patience=15
  - 梯度裁剪：max_norm=1.0

## 输出文件

训练完成后会生成：
- **模型检查点**：`models/checkpoints/model2_newdataset_checkpoint.pt`
- **注意力热力图**：`models/checkpoints/attention_visualizations_newdataset/`

## 描述格式

新数据集的描述格式为合并后的文本：
```
第一个描述 [另一个描述] 背景描述1 背景描述2
```

例如：
```
Black short-sleeved t-shirt paired with blue jeans and white sneakers A casual indoor setting in a modern cafe Glass windows and wooden benches visible in the background.
```

## 注意事项

1. **数据划分**：自动按7:2:1比例划分训练/验证/测试集
2. **特征提取**：需要先提取图像特征才能训练
3. **词表**：使用min_freq=3构建词表（出现3次及以上的词才计入）
4. **描述合并**：根据`use_lmm_as_first`决定第一个描述来源，然后合并所有描述

## 与原始数据集的差异

| 项目 | 原始数据集 | 新数据集 |
|------|-----------|---------|
| 描述内容 | 仅服饰描述 | 服饰+背景描述 |
| 词表大小 | ~5000 | 723 |
| 数据集规模 | 较大 | 1000条 |
| 描述来源 | 人工标注 | LMMs生成+人工修正 |

