#!/bin/bash
# 训练Model1b: 常规CNN编码器 + 2层LSTM解码器

set -e  # 遇到错误立即退出

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"  # 切换到项目根目录

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}训练Model1b: 常规CNN编码器 + 2层LSTM解码器${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}项目根目录: ${PROJECT_ROOT}${NC}"

# 检查Python环境
echo -e "\n${YELLOW}[步骤0] 检查Python环境...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到Python，请先安装Python${NC}"
    exit 1
fi
PYTHON_VERSION=$(python --version)
echo -e "${GREEN}[OK] Python版本: ${PYTHON_VERSION}${NC}"

# 检查必要的Python包
echo -e "\n${YELLOW}[步骤0.5] 检查Python依赖...${NC}"
python -c "import torch; import torchvision; import numpy; print('[OK] 所有依赖已安装')" || {
    echo -e "${RED}错误: 缺少必要的Python包，请运行: pip install torch torchvision numpy${NC}"
    exit 1
}

# 检查必要的数据文件
echo -e "\n${YELLOW}[步骤0.6] 检查数据文件...${NC}"
VOCAB_FILE="dataset/vocab.json"
TRAIN_FEATURES="dataset/features/train_features.npz"
VAL_FEATURES="dataset/features/val_features.npz"

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

# 创建模型保存目录
mkdir -p models

# 运行训练脚本
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}[步骤1] 开始训练模型${NC}"
echo -e "${GREEN}========================================${NC}"

python models/train_model1b.py

if [ $? -ne 0 ]; then
    echo -e "\n${RED}错误: 训练失败${NC}"
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}[OK] 训练完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n模型检查点已保存到: models/model1b_checkpoint.pt"
echo -e "\n改进点："
echo -e "  - 2层LSTM替代6层基础RNN（解决梯度消失）"
echo -e "  - 混合Teacher-Forcing训练（80% Teacher-Forcing + 20% 自回归）"
echo -e "  - <END>标记权重提升1.5倍（避免提前截断）"
echo -e "  - 学习率调度（ReduceLROnPlateau）"
echo -e "  - 温度采样生成（temperature=0.7，提升多样性）"
echo -e "\n可以使用以下命令加载模型进行推理："
echo -e "  python -c \"from models.model1b_cnn_2layer_lstm import FashionCaptionModelLSTM, load_vocab; ...\""

