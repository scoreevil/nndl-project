#!/bin/bash
# 快速开始脚本：运行完整评测流程

set -e

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 加载API密钥（如果存在配置文件）
API_KEYS_FILE="$PROJECT_ROOT/.api_keys.env"
if [ -f "$API_KEYS_FILE" ]; then
    source "$API_KEYS_FILE"
fi

# 设置PYTHONPATH，确保Python能找到utils模块
# 使用绝对路径，避免相对路径问题
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# 验证PYTHONPATH设置
if [ -z "$PYTHONPATH" ] || [[ "$PYTHONPATH" != *"$PROJECT_ROOT"* ]]; then
    echo "错误: PYTHONPATH设置失败"
    exit 1
fi

echo "=========================================="
echo "LMMs对比评测 - 快速开始"
echo "=========================================="
echo "项目根目录: $PROJECT_ROOT"
echo "当前目录: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# 验证utils模块是否可以导入
if ! PYTHONPATH="$PROJECT_ROOT" python -c "import utils" 2>/dev/null; then
    echo "错误: 无法导入utils模块，请检查PYTHONPATH设置"
    echo "尝试手动设置: export PYTHONPATH=$PROJECT_ROOT"
    exit 1
fi

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python"
    exit 1
fi

# 设置默认参数
NUM_SAMPLES=${1:-100}

echo "样本数量: $NUM_SAMPLES"
echo ""

# 步骤1: 准备测试集图像
echo "步骤1: 准备测试集图像..."
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.prepare_test_images --num_samples $NUM_SAMPLES

# 步骤2: 调用LMMs API（需要API密钥）
echo ""
echo "步骤2: 调用LMMs API..."
echo "注意: 需要设置API密钥环境变量"
echo "  - OPENAI_API_KEY (GPT-4V)"
echo "  - DASHSCOPE_API_KEY 或 QWEN_API_KEY (Qwen-VL)"
echo "  - MOONSHOT_API_KEY 或 KIMI_API_KEY (Kimi-VL)"
echo "  - ARK_API_KEY 或 DOUBAO_API_KEY (Doubao-VL)"
echo ""

# 检查API密钥（支持多种环境变量名）
QWEN_KEY="${DASHSCOPE_API_KEY:-$QWEN_API_KEY}"
KIMI_KEY="${MOONSHOT_API_KEY:-$KIMI_API_KEY}"
DOUBAO_KEY="${ARK_API_KEY:-$DOUBAO_API_KEY}"
if [ -z "$OPENAI_API_KEY" ] && [ -z "$QWEN_KEY" ] && [ -z "$KIMI_KEY" ] && [ -z "$DOUBAO_KEY" ]; then
    echo "警告: 未设置任何API密钥，跳过LMMs API调用"
    echo "如需调用LMMs API，请设置相应的环境变量后重新运行"
    SKIP_LMMS=true
else
    SKIP_LMMS=false
    
    # 调用各个LMMs API（遇到网络错误时跳过，继续处理其他模型）
    if [ ! -z "$OPENAI_API_KEY" ]; then
        echo "调用 GPT-4V API..."
        PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.call_lmms_api --lmm_type gpt --num_samples $NUM_SAMPLES || {
            echo "GPT-4V API调用失败（可能是网络连接问题，已跳过）"
            echo "继续处理其他模型..."
        }
    fi
    
    if [ ! -z "$QWEN_KEY" ]; then
        echo "调用 Qwen-VL API..."
        PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.call_lmms_api --lmm_type qwen --num_samples $NUM_SAMPLES || {
            echo "Qwen-VL API调用失败（可能是网络连接问题，已跳过）"
            echo "继续处理其他模型..."
        }
    fi
    
    if [ ! -z "$KIMI_KEY" ]; then
        echo "调用 Kimi-VL API..."
        PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.call_lmms_api --lmm_type kimi --num_samples $NUM_SAMPLES || {
            echo "Kimi-VL API调用失败（可能是网络连接问题，已跳过）"
            echo "继续处理其他模型..."
        }
    fi
    
    if [ ! -z "$DOUBAO_KEY" ]; then
        echo "调用 Doubao-VL API..."
        PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.call_lmms_api --lmm_type doubao --num_samples $NUM_SAMPLES || {
            echo "Doubao-VL API调用失败（可能是网络连接问题，已跳过）"
            echo "继续处理其他模型..."
        }
    fi
fi

# 步骤3: 自研模型推理（需要checkpoint文件）
echo ""
echo "步骤3: 自研模型推理..."
echo "注意: 需要模型checkpoint文件"
echo "  查找位置: models/checkpoints/ 或 checkpoints/"
echo "  - model1_checkpoint.pt (Model1/RNN)"
echo "  - model1b_checkpoint.pt (Model2/LSTM)"
echo "  - model2_checkpoint.pt (Model3/注意力RNN)"
echo "  - model5_checkpoint.pt (Model5/全Transformer)"
echo ""

# 检查checkpoint文件（支持多个可能的位置）
SKIP_SELF_MODELS=true

# Model1 (RNN)
if [ -f "models/checkpoints/model1_checkpoint.pt" ]; then
    echo "运行 Model1 (RNN)..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions --model_type model1 --checkpoint models/checkpoints/model1_checkpoint.pt --num_samples $NUM_SAMPLES || echo "Model1推理失败"
    SKIP_SELF_MODELS=false
elif [ -f "checkpoints/model1_checkpoint.pt" ]; then
    echo "运行 Model1 (RNN)..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions --model_type model1 --checkpoint checkpoints/model1_checkpoint.pt --num_samples $NUM_SAMPLES || echo "Model1推理失败"
    SKIP_SELF_MODELS=false
fi

# Model2 (LSTM) - 对应Model1b
if [ -f "models/checkpoints/model1b_checkpoint.pt" ]; then
    echo "运行 Model2 (LSTM)..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions --model_type model1b --checkpoint models/checkpoints/model1b_checkpoint.pt --num_samples $NUM_SAMPLES || echo "Model2推理失败"
    SKIP_SELF_MODELS=false
elif [ -f "checkpoints/model1b_checkpoint.pt" ]; then
    echo "运行 Model2 (LSTM)..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions --model_type model1b --checkpoint checkpoints/model1b_checkpoint.pt --num_samples $NUM_SAMPLES || echo "Model2推理失败"
    SKIP_SELF_MODELS=false
fi

# Model3 (注意力RNN) - 对应Model2
if [ -f "models/checkpoints/model2_checkpoint.pt" ]; then
    echo "运行 Model3 (注意力RNN)..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions --model_type model2 --checkpoint models/checkpoints/model2_checkpoint.pt --num_samples $NUM_SAMPLES || echo "Model3推理失败"
    SKIP_SELF_MODELS=false
elif [ -f "checkpoints/model2_checkpoint.pt" ]; then
    echo "运行 Model3 (注意力RNN)..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions --model_type model2 --checkpoint checkpoints/model2_checkpoint.pt --num_samples $NUM_SAMPLES || echo "Model3推理失败"
    SKIP_SELF_MODELS=false
fi

# Model5 (全Transformer)
if [ -f "models/checkpoints/model5_checkpoint.pt" ]; then
    echo "运行 Model5 (全Transformer)..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions --model_type model5 --checkpoint models/checkpoints/model5_checkpoint.pt --num_samples $NUM_SAMPLES || echo "Model5推理失败"
    SKIP_SELF_MODELS=false
elif [ -f "checkpoints/model5_checkpoint.pt" ]; then
    echo "运行 Model5 (全Transformer)..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions --model_type model5 --checkpoint checkpoints/model5_checkpoint.pt --num_samples $NUM_SAMPLES || echo "Model5推理失败"
    SKIP_SELF_MODELS=false
fi

if [ "$SKIP_SELF_MODELS" = true ]; then
    echo "警告: 未找到模型checkpoint文件，跳过自研模型推理"
fi

# 步骤4: 定量对比评测
echo ""
echo "步骤4: 定量对比评测..."
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.evaluate_comparison --num_samples $NUM_SAMPLES

# 步骤5: 生成定性评测模板
echo ""
echo "步骤5: 生成定性评测模板..."
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.qualitative_evaluation_template --num_samples $NUM_SAMPLES

echo ""
echo "=========================================="
echo "评测完成！"
echo "=========================================="
echo ""
echo "结果文件:"
echo "  - 定量评测报告: results/comparison_report.md"
echo "  - 定性评测模板: results/qualitative_evaluation_template.md"
echo "  - 生成结果: results/*_generated.txt"
echo ""

