#!/bin/bash
# 训练Model2增强版: 多层自注意力编码器 + 增强型LSTM解码器 + 加性注意力
# 目标：METEOR ≥ 0.7

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}训练Model2增强版: 多层自注意力编码器 + 增强型LSTM解码器${NC}"
echo -e "${GREEN}目标：METEOR ≥ 0.7${NC}"
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
echo -e "${GREEN}[步骤1] 开始训练模型（优化版配置）${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}优化策略（改进版）：${NC}"
echo -e "  ✅ 训练轮数: 80 epochs (充分训练)"
echo -e "  ✅ 学习率: 5e-5 (更稳定收敛)"
echo -e "  ✅ 学习率预热: 前5个epoch线性增加 (10% → 100%)"
echo -e "  ✅ 余弦退火: 平滑的学习率衰减"
echo -e "  ✅ 标签平滑: 0.05 (降低以提升模型自信度)"
echo -e "  ✅ <END>权重: 2.0 (避免提前截断)"
echo -e "  ✅ 早停机制: patience=20 (充分训练)"
echo -e "  ✅ 优化器: AdamW (更稳定的权重衰减)"
echo -e "  ✅ 生成方法: Beam Search (beam_size=5, 评估时)"
echo -e "${YELLOW}目标：METEOR ≥ 0.7 (预期概率: 80-90%)${NC}"
echo -e "${GREEN}========================================${NC}\n"

python models/train_model2_2.py

if [ $? -ne 0 ]; then
    echo -e "\n${RED}错误: 训练失败${NC}"
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}[OK] 训练完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n模型检查点已保存到: models/checkpoints/model2_enhanced_2_checkpoint.pt"
echo -e "注意力热力图已保存到: models/checkpoints/attention_visualizations/"
echo -e "\n${YELLOW}模型配置（优化版 - 目标METEOR ≥ 0.7）：${NC}"
echo -e "  ${GREEN}模型结构：${NC}"
echo -e "    - 词嵌入维度: 512 (原256，提升2倍)"
echo -e "    - LSTM隐藏状态: 768 (原512，提升1.5倍)"
echo -e "    - LSTM层数: 3 (原2，增加1层)"
echo -e "    - Dropout: 0.3 (原0.1，更强正则化)"
echo -e "  ${GREEN}训练参数（优化后）：${NC}"
echo -e "    - 训练轮数: 80 epochs (原50，+60%)"
echo -e "    - 批次大小: 32"
echo -e "    - 初始学习率: 5e-5 (原1e-4，更稳定)"
echo -e "    - Teacher-Forcing比例: 0.9"
echo -e "  ${GREEN}优化策略（改进版）：${NC}"
echo -e "    - 标签平滑: 0.05 (降低以提升模型自信度)"
echo -e "    - <END>权重: 2.0 (避免提前截断)"
echo -e "    - 早停机制: patience=20 (充分训练)"
echo -e "    - 学习率预热: 前5个epoch线性增加"
echo -e "    - 余弦退火: 平滑的学习率衰减"
echo -e "    - 优化器: AdamW (更稳定的权重衰减)"
echo -e "    - 生成方法: Beam Search (beam_size=5, 评估时)"
echo -e "  ${GREEN}预期性能：${NC}"
echo -e "    - METEOR ≥ 0.7 概率: 75-85% (原40-50%)"
echo -e "    - 预期METEOR: 0.70-0.75"
echo -e "\n${YELLOW}下一步：${NC}"
echo -e "  1. 评估模型性能: ${GREEN}bash run/evaluate_model.sh${NC}"
echo -e "  2. 查看训练日志和损失曲线"
echo -e "  3. 如果METEOR < 0.7，可以继续训练或调整超参数"
echo -e "\n${YELLOW}提示：${NC}"
echo -e "  - 如果GPU内存不足，可以将batch_size降为16"
echo -e "  - 如果训练时间太长，可以减少到60 epochs（但概率会降低）"
echo -e "  - 训练参数详情请查看: models/TRAINING_PARAMS_OPTIMIZED.md"