#!/bin/bash
# DeepFashion-MultiModal数据集完整数据处理流程
# 包括：数据加载、特征提取、文本预处理

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

# 配置参数（相对于项目根目录）
CAPTIONS_FILE="dataset/captions.json"
IMAGES_DIR="dataset/images"
OUTPUT_DIR="dataset/features"
VOCAB_FILE="dataset/vocab.json"
BATCH_SIZE=32
MAX_LEN=20
MIN_FREQ=3
RANDOM_SEED=42

echo -e "${GREEN}项目根目录: ${PROJECT_ROOT}${NC}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}DeepFashion-MultiModal数据处理流程${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查Python环境
echo -e "\n${YELLOW}[步骤0] 检查Python环境...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到Python，请先安装Python${NC}"
    exit 1
fi
PYTHON_VERSION=$(python --version)
echo -e "${GREEN}✓ Python版本: ${PYTHON_VERSION}${NC}"

# 检查必要的Python包
echo -e "\n${YELLOW}[步骤0.5] 检查Python依赖...${NC}"
python -c "import torch; import torchvision; import numpy; print('✓ 所有依赖已安装')" || {
    echo -e "${RED}错误: 缺少必要的Python包，请运行: pip install torch torchvision numpy${NC}"
    exit 1
}

# 检查数据文件是否存在
echo -e "\n${YELLOW}[步骤0.6] 检查数据文件...${NC}"
if [ ! -f "$CAPTIONS_FILE" ]; then
    echo -e "${RED}错误: 找不到文件 ${CAPTIONS_FILE}${NC}"
    exit 1
fi
if [ ! -d "$IMAGES_DIR" ]; then
    echo -e "${RED}错误: 找不到目录 ${IMAGES_DIR}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 数据文件检查通过${NC}"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 步骤1: 数据加载和验证
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}[步骤1] 数据加载和验证${NC}"
echo -e "${GREEN}========================================${NC}"
python << EOF
import sys
import os
sys.path.insert(0, '${PROJECT_ROOT}')
os.chdir('${PROJECT_ROOT}')
from utils.data_loader import load_and_validate_dataset

print("开始加载数据集...")
train_data, val_data, test_data = load_and_validate_dataset(
    captions_file="${CAPTIONS_FILE}",
    images_dir="${IMAGES_DIR}",
    random_seed=${RANDOM_SEED}
)

print(f"\n数据集加载完成:")
print(f"  训练集: {len(train_data)}条")
print(f"  验证集: {len(val_data)}条")
print(f"  测试集: {len(test_data)}条")

# 保存数据列表（用于后续步骤）
import pickle
os.makedirs('run', exist_ok=True)
with open('run/temp_data.pkl', 'wb') as f:
    pickle.dump((train_data, val_data, test_data), f)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 数据加载失败${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 数据加载完成${NC}"

# 步骤2: 特征提取
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}[步骤2] 图像特征提取${NC}"
echo -e "${GREEN}========================================${NC}"
python << EOF
import sys
import os
sys.path.insert(0, '${PROJECT_ROOT}')
os.chdir('${PROJECT_ROOT}')
import pickle
from utils.feature_extractor import extract_features_for_datasets

# 加载数据
with open('run/temp_data.pkl', 'rb') as f:
    train_data, val_data, test_data = pickle.load(f)

print("开始提取图像特征...")
extract_features_for_datasets(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    output_dir="${OUTPUT_DIR}",
    batch_size=${BATCH_SIZE},
    device=None  # 自动选择设备
)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 特征提取失败${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 特征提取完成${NC}"

# 步骤3: 文本预处理
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}[步骤3] 文本预处理${NC}"
echo -e "${GREEN}========================================${NC}"
python << EOF
import sys
import os
sys.path.insert(0, '${PROJECT_ROOT}')
os.chdir('${PROJECT_ROOT}')
import pickle
from utils.text_processor import build_vocab_and_process

# 加载数据
with open('run/temp_data.pkl', 'rb') as f:
    train_data, val_data, test_data = pickle.load(f)

print("开始文本预处理...")
processor, train_sequences, val_sequences, test_sequences = build_vocab_and_process(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    max_len=${MAX_LEN},
    min_freq=${MIN_FREQ},
    vocab_file="${VOCAB_FILE}"
)

print(f"\n文本预处理完成:")
print(f"  词典大小: {processor.vocab_size}")
print(f"  训练集序列形状: {train_sequences.shape}")
print(f"  验证集序列形状: {val_sequences.shape}")
print(f"  测试集序列形状: {test_sequences.shape}")

# 保存序列（可选，用于后续训练）
import torch
torch.save({
    'train_sequences': train_sequences,
    'val_sequences': val_sequences,
    'test_sequences': test_sequences
}, 'run/text_sequences.pt')
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 文本预处理失败${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 文本预处理完成${NC}"

# 清理临时文件
echo -e "\n${YELLOW}[清理] 删除临时文件...${NC}"
rm -f run/temp_data.pkl
echo -e "${GREEN}✓ 临时文件已清理${NC}"

# 完成
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ 所有数据处理步骤完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n生成的文件:"
echo -e "  - ${OUTPUT_DIR}/train_features.npz"
echo -e "  - ${OUTPUT_DIR}/val_features.npz"
echo -e "  - ${OUTPUT_DIR}/test_features.npz"
echo -e "  - ${VOCAB_FILE}"
echo -e "  - run/text_sequences.pt (可选)"
echo -e "\n现在可以使用这些文件进行模型训练了！"

