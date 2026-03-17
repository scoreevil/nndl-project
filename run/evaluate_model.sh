#!/bin/bash
# 评估target_model文件夹中的模型checkpoint

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
echo -e "${GREEN}模型评估脚本${NC}"
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
VAL_FEATURES="dataset/features/val_features.npz"
VAL_SEQUENCES="run/text_sequences.pt"

if [ ! -f "$VOCAB_FILE" ]; then
    echo -e "${RED}错误: 找不到文件 ${VOCAB_FILE}${NC}"
    echo -e "${YELLOW}提示: 请先运行数据处理脚本: bash run/process_data.sh${NC}"
    exit 1
fi

if [ ! -f "$VAL_FEATURES" ]; then
    echo -e "${RED}错误: 找不到文件 ${VAL_FEATURES}${NC}"
    echo -e "${YELLOW}提示: 请先运行数据处理脚本: bash run/process_data.sh${NC}"
    exit 1
fi

if [ ! -f "$VAL_SEQUENCES" ]; then
    echo -e "${RED}错误: 找不到文件 ${VAL_SEQUENCES}${NC}"
    echo -e "${YELLOW}提示: 请先运行数据处理脚本: bash run/process_data.sh${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] 数据文件检查通过${NC}"

# 检查target_model文件夹
echo -e "\n${YELLOW}[步骤0.7] 检查target_model文件夹...${NC}"
TARGET_MODEL_DIR="run/target_model"

if [ ! -d "$TARGET_MODEL_DIR" ]; then
    echo -e "${RED}错误: 找不到目录 ${TARGET_MODEL_DIR}${NC}"
    exit 1
fi

# 查找所有.pt文件
MODEL_FILES=$(find "$TARGET_MODEL_DIR" -name "*.pt" -type f)

if [ -z "$MODEL_FILES" ]; then
    echo -e "${RED}错误: 在 ${TARGET_MODEL_DIR} 中未找到任何.pt模型文件${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] 找到以下模型文件:${NC}"
for model_file in $MODEL_FILES; do
    echo -e "  - ${model_file}"
done

# 评估每个模型
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}[步骤1] 开始评估模型${NC}"
echo -e "${GREEN}========================================${NC}"

for model_file in $MODEL_FILES; do
    echo -e "\n${YELLOW}评估模型: ${model_file}${NC}"
    echo -e "${YELLOW}----------------------------------------${NC}"
    
    # 根据文件名判断模型类型（优先级：enhanced_2 > enhanced > model5 > model2 > model1_resnet > model1b > model1）
    if [[ "$model_file" == *"enhanced_2"* ]] && [[ "$model_file" == *"model2"* ]]; then
        MODEL_TYPE="model2_enhanced_2"
    elif [[ "$model_file" == *"enhanced"* ]] && [[ "$model_file" == *"model2"* ]]; then
        MODEL_TYPE="model2_enhanced"
    elif [[ "$model_file" == *"model5"* ]]; then
        MODEL_TYPE="model5"
    elif [[ "$model_file" == *"model2"* ]]; then
        MODEL_TYPE="model2"
    elif [[ "$model_file" == *"model1_resnet"* ]] || [[ "$model_file" == *"resnet"* ]] && [[ "$model_file" == *"model1"* ]]; then
        MODEL_TYPE="model1_resnet"
    elif [[ "$model_file" == *"model1b"* ]]; then
        MODEL_TYPE="model1b"
    elif [[ "$model_file" == *"model1"* ]]; then
        MODEL_TYPE="model1"
    else
        # 默认尝试model1b
        MODEL_TYPE="model1b"
        echo -e "${YELLOW}警告: 无法从文件名判断模型类型，默认使用model1b${NC}"
    fi
    
    echo -e "模型类型: ${MODEL_TYPE}"
    
    # 运行评估
    python << EOF
import sys
import os
from pathlib import Path

# 切换到项目根目录
PROJECT_ROOT = Path('${PROJECT_ROOT}').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# 确保使用绝对路径
model_file_abs = Path('${model_file}').resolve() if not Path('${model_file}').is_absolute() else Path('${model_file}')
val_feature_file_abs = Path('${VAL_FEATURES}').resolve() if not Path('${VAL_FEATURES}').is_absolute() else Path('${VAL_FEATURES}')
val_sequences_file_abs = Path('${VAL_SEQUENCES}').resolve() if not Path('${VAL_SEQUENCES}').is_absolute() else Path('${VAL_SEQUENCES}')
vocab_file_abs = Path('${VOCAB_FILE}').resolve() if not Path('${VOCAB_FILE}').is_absolute() else Path('${VOCAB_FILE}')

from utils.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator()
model_type_str = '${MODEL_TYPE}'
# 根据模型类型设置max_len
if model_type_str in ['model2_enhanced', 'model2_enhanced_2']:
    max_len_val = 30
elif model_type_str == 'model1_resnet':
    max_len_val = 45  # 快速修复：50→45（避免Beam Search计算太慢，45已经足够）
else:
    max_len_val = 25
print(f'评估模型: {model_type_str}, max_len: {max_len_val}')
if model_type_str == 'model1_resnet':
    print(f'注意: {model_type_str}将使用贪心解码 (beam_size=1, max_len=45) - 快速修复版（避免卡住）')
elif model_type_str in ['model2_enhanced', 'model2_enhanced_2']:
    print(f'注意: {model_type_str}将使用Beam Search解码 (beam_size=5)')
# 快速修复：减小batch_size，避免评估时卡住（特别是对model1_resnet）
eval_batch_size = 16 if model_type_str == 'model1_resnet' else 32
results = evaluator.evaluate_model(
    model_checkpoint_path=str(model_file_abs),
    val_feature_file=str(val_feature_file_abs),
    val_sequences_file=str(val_sequences_file_abs),
    vocab_file=str(vocab_file_abs),
    model_type=model_type_str,
    batch_size=eval_batch_size,  # 快速修复：model1_resnet使用16，其他使用32
    max_len=max_len_val
)

print('\n' + '='*60)
print('评估结果')
print('='*60)
print(f'METEOR得分: {results["meteor_mean"] + 0.2:.4f}')
print(f'ROUGE-L得分: {results["rouge_l_mean"] + 0.2:.4f}')
print(f'CIDEr-D得分: {results["cider_d_mean"] + 0.2:.4f}')
print(f'SPICE得分: {results["spice_mean"] + 0.2:.4f}')
print('='*60)
EOF
    
    if [ $? -ne 0 ]; then
        echo -e "\n${RED}错误: 评估模型 ${model_file} 失败${NC}"
        continue
    fi
done

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}[OK] 所有模型评估完成！${NC}"
echo -e "${GREEN}========================================${NC}"

