#!/bin/bash
# 训练Model2: 增强型CNN编码器（局部自注意力）+ 注意力增强LSTM解码器

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}训练Model2: 增强型CNN编码器（局部自注意力）+ 注意力增强LSTM解码器${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}项目根目录: ${PROJECT_ROOT}${NC}"

echo -e "\n${YELLOW}[步骤0] 检查Python环境...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到Python，请先安装Python${NC}"
    exit 1
fi
PYTHON_VERSION=$(python --version)
echo -e "${GREEN}[OK] Python版本: ${PYTHON_VERSION}${NC}"

echo -e "\n${YELLOW}[步骤0.5] 检查Python依赖...${NC}"
python -c "import torch; import torchvision; import numpy; import matplotlib; print('[OK] 所有依赖已安装')" || {
    echo -e "${RED}错误: 缺少必要的Python包，请运行: pip install torch torchvision numpy matplotlib${NC}"
    exit 1
}

echo -e "\n${YELLOW}[步骤0.6] 检查数据文件...${NC}"
VOCAB_FILE="dataset/vocab.json"
TRAIN_FEATURES="dataset/features/train_features.npz"
VAL_FEATURES="dataset/features/val_features.npz"
TEXT_SEQUENCES_FILE="run/text_sequences.pt"

if [ ! -f "$VOCAB_FILE" ]; then
    echo -e "${RED}错误: 找不到文件 ${VOCAB_FILE}${NC}"
    echo -e "${YELLOW}提示: 请先运行数据处理脚本: bash run/process_data.sh${NC}"
    exit 1
fi

if [ ! -f "$TRAIN_FEATURES" ]; then
    echo -e "${RED}错误: 找不到文件 ${TRAIN_FEATURES}${NC}"
    echo -e "${YELLOW}提示: 请先运行数据处理脚本: bash run/process_data.sh${NC}"
    exit 1
fi

if [ ! -f "$VAL_FEATURES" ]; then
    echo -e "${RED}错误: 找不到文件 ${VAL_FEATURES}${NC}"
    echo -e "${YELLOW}提示: 请先运行数据处理脚本: bash run/process_data.sh${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] 数据文件检查通过${NC}"

mkdir -p models/checkpoints
mkdir -p models/checkpoints/attention_visualizations

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}[步骤1] 开始训练模型${NC}"
echo -e "${GREEN}========================================${NC}"

# 设置OpenMP环境变量，修复libiomp5md.dll重复初始化错误
export KMP_DUPLICATE_LIB_OK=TRUE

python models/train_model2.py

if [ $? -ne 0 ]; then
    echo -e "\n${RED}错误: 训练失败${NC}"
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}[OK] 训练完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n模型检查点已保存到: models/checkpoints/model2_checkpoint.pt"
echo -e "注意力热力图已保存到: models/checkpoints/attention_visualizations/"
echo -e "\n可以使用以下命令加载模型进行推理："
echo -e "  python -c \"from models.model2_local_selfattn_attention_rnn import FashionCaptionModelAttention, load_vocab; ...\""