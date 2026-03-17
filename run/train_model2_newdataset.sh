#!/bin/bash
# 训练Model2（局部表示+自注意力→RNN+注意力）用于新数据集

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="/root/autodl-tmp"
cd "$PROJECT_ROOT"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}训练Model2（新数据集）${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查特征文件是否存在
echo -e "\n${YELLOW}[检查] 验证特征文件...${NC}"
if [ ! -f "newdataset/features/train_features.npz" ]; then
    echo -e "${RED}错误: 训练集特征文件不存在${NC}"
    echo -e "${YELLOW}正在提取特征...${NC}"
    python3 utils/extract_new_dataset_features.py
fi

if [ ! -f "newdataset/features/val_features.npz" ]; then
    echo -e "${RED}错误: 验证集特征文件不存在${NC}"
    echo -e "${YELLOW}正在提取特征...${NC}"
    python3 utils/extract_new_dataset_features.py
fi

# 检查词表文件
if [ ! -f "newdataset/vocab.json" ]; then
    echo -e "${RED}错误: 词表文件不存在${NC}"
    echo -e "${YELLOW}正在预处理数据集...${NC}"
    python3 NewDatasetCreate/preprocess_new_dataset.py \
        --annotations_file NewDatasetCreate/annotations.json \
        --vocab_file newdataset/vocab.json \
        --sequences_file newdataset/text_sequences.pt \
        --max_len 25 \
        --min_freq 3
fi

echo -e "${GREEN}✓ 所有必需文件已就绪${NC}"

# 开始训练
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}[训练] 开始训练模型${NC}"
echo -e "${GREEN}========================================${NC}"

python3 models/train_model2_newdataset.py

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ 训练完成！${NC}"
echo -e "${GREEN}========================================${NC}"

