#!/bin/bash
# 训练Model5: 全Transformer架构（局部表示+Transformer编码器→Transformer解码器）

set -e  # 遇到错误立即退出

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"  # 切换到项目根目录

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}训练Model5: 全Transformer架构${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查Python环境
echo -e "\n${YELLOW}[步骤0.1] 检查Python环境...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}[错误] 未找到Python，请先安装Python${NC}"
    exit 1
fi
PYTHON_VERSION=$(python --version 2>&1)
echo -e "${GREEN}[OK] ${PYTHON_VERSION}${NC}"

# 检查必要的Python包
echo -e "\n${YELLOW}[步骤0.2] 检查必要的Python包...${NC}"
python << EOF
import sys
required_packages = ['torch', 'numpy', 'pathlib']
missing_packages = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing_packages.append(pkg)

if missing_packages:
    print(f"缺少以下包: {', '.join(missing_packages)}")
    sys.exit(1)
else:
    print("所有必要的包都已安装")
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}[错误] 缺少必要的Python包，请先安装${NC}"
    exit 1
fi
echo -e "${GREEN}[OK] 依赖检查通过${NC}"

# 检查必要的数据文件
echo -e "\n${YELLOW}[步骤0.3] 检查数据文件...${NC}"
VOCAB_FILE="dataset/vocab.json"
TRAIN_FEATURES="dataset/features/train_features.npz"
VAL_FEATURES="dataset/features/val_features.npz"
CAPTIONS_FILE="dataset/captions.json"
IMAGES_DIR="dataset/images"

if [ ! -f "$VOCAB_FILE" ]; then
    echo -e "${RED}[错误] 未找到词汇表文件: $VOCAB_FILE${NC}"
    exit 1
fi

if [ ! -f "$TRAIN_FEATURES" ]; then
    echo -e "${RED}[错误] 未找到训练特征文件: $TRAIN_FEATURES${NC}"
    exit 1
fi

if [ ! -f "$VAL_FEATURES" ]; then
    echo -e "${RED}[错误] 未找到验证特征文件: $VAL_FEATURES${NC}"
    exit 1
fi

if [ ! -f "$CAPTIONS_FILE" ]; then
    echo -e "${RED}[错误] 未找到描述文件: $CAPTIONS_FILE${NC}"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo -e "${RED}[错误] 未找到图像目录: $IMAGES_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] 数据文件检查通过${NC}"

# 检查checkpoints目录
echo -e "\n${YELLOW}[步骤0.4] 检查checkpoints目录...${NC}"
CHECKPOINT_DIR="run/checkpoints"
mkdir -p "$CHECKPOINT_DIR"
echo -e "${GREEN}[OK] checkpoints目录已准备${NC}"

# 开始训练
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}[步骤1] 开始训练${NC}"
echo -e "${GREEN}========================================${NC}"

python models/train_model5.py

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}[完成] 训练成功完成${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "模型checkpoint已保存到: ${CHECKPOINT_DIR}/model5_checkpoint.pt"
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}[错误] 训练失败${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

