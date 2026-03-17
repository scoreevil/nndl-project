#!/bin/bash
# 生成所有可用模型的结果

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

# 设置PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

NUM_SAMPLES=${1:-100}

echo "=========================================="
echo "生成所有可用模型的结果"
echo "=========================================="
echo "项目根目录: $PROJECT_ROOT"
echo "样本数量: $NUM_SAMPLES"
echo ""

# 创建results目录
mkdir -p results

# 自研模型
echo "=== 自研模型 ==="

# Model1 (RNN)
if [ -f "models/checkpoints/model1_checkpoint.pt" ]; then
    echo "生成 Model1 (RNN) 描述..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions \
        --model_type model1 \
        --checkpoint models/checkpoints/model1_checkpoint.pt \
        --num_samples $NUM_SAMPLES || echo "Model1生成失败"
elif [ -f "checkpoints/model1_checkpoint.pt" ]; then
    echo "生成 Model1 (RNN) 描述..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions \
        --model_type model1 \
        --checkpoint checkpoints/model1_checkpoint.pt \
        --num_samples $NUM_SAMPLES || echo "Model1生成失败"
else
    echo "跳过 Model1: checkpoint文件不存在"
fi

# Model2 (LSTM)
if [ -f "models/checkpoints/model1b_checkpoint.pt" ]; then
    echo "生成 Model2 (LSTM) 描述..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions \
        --model_type model1b \
        --checkpoint models/checkpoints/model1b_checkpoint.pt \
        --num_samples $NUM_SAMPLES || echo "Model2生成失败"
elif [ -f "checkpoints/model1b_checkpoint.pt" ]; then
    echo "生成 Model2 (LSTM) 描述..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions \
        --model_type model1b \
        --checkpoint checkpoints/model1b_checkpoint.pt \
        --num_samples $NUM_SAMPLES || echo "Model2生成失败"
else
    echo "跳过 Model2: checkpoint文件不存在"
fi

# Model3 (注意力RNN)
if [ -f "models/checkpoints/model2_checkpoint.pt" ]; then
    echo "生成 Model3 (注意力RNN) 描述..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions \
        --model_type model2 \
        --checkpoint models/checkpoints/model2_checkpoint.pt \
        --num_samples $NUM_SAMPLES || echo "Model3生成失败"
elif [ -f "checkpoints/model2_checkpoint.pt" ]; then
    echo "生成 Model3 (注意力RNN) 描述..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions \
        --model_type model2 \
        --checkpoint checkpoints/model2_checkpoint.pt \
        --num_samples $NUM_SAMPLES || echo "Model3生成失败"
else
    echo "跳过 Model3: checkpoint文件不存在"
fi

# Model5 (全Transformer)
if [ -f "models/checkpoints/model5_checkpoint.pt" ]; then
    echo "生成 Model5 (全Transformer) 描述..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions \
        --model_type model5 \
        --checkpoint models/checkpoints/model5_checkpoint.pt \
        --num_samples $NUM_SAMPLES || echo "Model5生成失败"
elif [ -f "checkpoints/model5_checkpoint.pt" ]; then
    echo "生成 Model5 (全Transformer) 描述..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.generate_self_model_captions \
        --model_type model5 \
        --checkpoint checkpoints/model5_checkpoint.pt \
        --num_samples $NUM_SAMPLES || echo "Model5生成失败"
else
    echo "跳过 Model5: checkpoint文件不存在"
fi

# LMMs
echo ""
echo "=== LMMs ==="
echo "注意: 需要设置API密钥环境变量"
echo "  - DASHSCOPE_API_KEY 或 QWEN_API_KEY (Qwen-VL)"
echo "  - MOONSHOT_API_KEY 或 KIMI_API_KEY (Kimi-VL)"
echo "  - ARK_API_KEY 或 DOUBAO_API_KEY (Doubao-VL)"

if [ ! -z "$OPENAI_API_KEY" ]; then
    echo "生成 GPT-4V 描述..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.call_lmms_api \
        --lmm_type gpt --num_samples $NUM_SAMPLES || {
        echo "GPT-4V生成失败（可能是网络连接问题，已跳过）"
        echo "继续处理其他模型..."
    }
else
    echo "跳过 GPT-4V: 未设置OPENAI_API_KEY"
fi

# Qwen支持DASHSCOPE_API_KEY和QWEN_API_KEY两种环境变量
QWEN_KEY="${DASHSCOPE_API_KEY:-$QWEN_API_KEY}"
if [ ! -z "$QWEN_KEY" ]; then
    echo "生成 Qwen-VL 描述..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.call_lmms_api \
        --lmm_type qwen --num_samples $NUM_SAMPLES || {
        echo "Qwen-VL生成失败（可能是网络连接问题，已跳过）"
        echo "继续处理其他模型..."
    }
else
    echo "跳过 Qwen-VL: 未设置DASHSCOPE_API_KEY或QWEN_API_KEY"
fi

# Kimi支持MOONSHOT_API_KEY和KIMI_API_KEY两种环境变量
KIMI_KEY="${MOONSHOT_API_KEY:-$KIMI_API_KEY}"
if [ ! -z "$KIMI_KEY" ]; then
    echo "生成 Kimi-VL 描述..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.call_lmms_api \
        --lmm_type kimi --num_samples $NUM_SAMPLES || {
        echo "Kimi-VL生成失败（可能是网络连接问题，已跳过）"
        echo "继续处理其他模型..."
    }
else
    echo "跳过 Kimi-VL: 未设置MOONSHOT_API_KEY或KIMI_API_KEY"
fi

# Doubao支持ARK_API_KEY和DOUBAO_API_KEY两种环境变量
DOUBAO_KEY="${ARK_API_KEY:-$DOUBAO_API_KEY}"
if [ ! -z "$DOUBAO_KEY" ]; then
    echo "生成 Doubao-VL 描述..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m utils.call_lmms_api \
        --lmm_type doubao --num_samples $NUM_SAMPLES || {
        echo "Doubao-VL生成失败（可能是网络连接问题，已跳过）"
        echo "继续处理其他模型..."
    }
else
    echo "跳过 Doubao-VL: 未设置ARK_API_KEY或DOUBAO_API_KEY"
fi

echo ""
echo "=========================================="
echo "生成完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
ls -lh results/*_generated.txt 2>/dev/null || echo "  没有找到生成结果文件"
echo ""
