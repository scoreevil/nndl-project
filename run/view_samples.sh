#!/bin/bash
# 快速查看模型生成的样本

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}查看模型生成的样本${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查模型文件
MODEL_FILE="models/checkpoints/model2_enhanced_checkpoint.pt"
if [ ! -f "$MODEL_FILE" ]; then
    # 尝试target_model文件夹
    MODEL_FILE="run/target_model/model2_enhanced_checkpoint.pt"
    if [ ! -f "$MODEL_FILE" ]; then
        echo -e "${RED}错误: 找不到模型文件${NC}"
        echo -e "${YELLOW}请将模型文件放到以下位置之一:${NC}"
        echo -e "  - models/checkpoints/model2_enhanced_checkpoint.pt"
        echo -e "  - run/target_model/model2_enhanced_checkpoint.pt"
        exit 1
    fi
fi

echo -e "${GREEN}使用模型文件: ${MODEL_FILE}${NC}"

# 检查数据文件
VOCAB_FILE="dataset/vocab.json"
VAL_FEATURES="dataset/features/val_features.npz"
VAL_SEQUENCES="run/text_sequences.pt"

if [ ! -f "$VOCAB_FILE" ]; then
    echo -e "${RED}错误: 找不到词典文件 ${VOCAB_FILE}${NC}"
    exit 1
fi

if [ ! -f "$VAL_FEATURES" ]; then
    echo -e "${RED}错误: 找不到特征文件 ${VAL_FEATURES}${NC}"
    exit 1
fi

if [ ! -f "$VAL_SEQUENCES" ]; then
    echo -e "${RED}错误: 找不到序列文件 ${VAL_SEQUENCES}${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] 数据文件检查通过${NC}"

# 运行查看脚本
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}开始查看生成的样本${NC}"
echo -e "${GREEN}========================================${NC}\n"

python utils/view_generated_samples.py \
    --model "$MODEL_FILE" \
    --val_features "$VAL_FEATURES" \
    --val_sequences "$VAL_SEQUENCES" \
    --vocab "$VOCAB_FILE" \
    --num_samples 20 \
    --batch_size 8 \
    --max_len 30

if [ $? -ne 0 ]; then
    echo -e "\n${RED}错误: 查看样本失败${NC}"
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}[OK] 样本查看完成${NC}"
echo -e "${GREEN}========================================${NC}"
