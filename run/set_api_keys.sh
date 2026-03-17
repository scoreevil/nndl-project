#!/bin/bash
# 设置API密钥环境变量

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# API密钥配置文件路径
API_KEYS_FILE="$PROJECT_ROOT/.api_keys.env"

if [ -f "$API_KEYS_FILE" ]; then
    echo "加载API密钥配置文件: $API_KEYS_FILE"
    source "$API_KEYS_FILE"
    echo "API密钥已设置"
    echo ""
    echo "已设置的API密钥:"
    [ ! -z "$OPENAI_API_KEY" ] && echo "  ✓ OPENAI_API_KEY"
    [ ! -z "$MOONSHOT_API_KEY" ] && echo "  ✓ MOONSHOT_API_KEY"
    [ ! -z "$KIMI_API_KEY" ] && echo "  ✓ KIMI_API_KEY"
    [ ! -z "$DASHSCOPE_API_KEY" ] && echo "  ✓ DASHSCOPE_API_KEY"
    [ ! -z "$QWEN_API_KEY" ] && echo "  ✓ QWEN_API_KEY"
    [ ! -z "$ARK_API_KEY" ] && echo "  ✓ ARK_API_KEY"
    [ ! -z "$DOUBAO_API_KEY" ] && echo "  ✓ DOUBAO_API_KEY"
else
    echo "警告: API密钥配置文件不存在: $API_KEYS_FILE"
    echo "请创建配置文件或手动设置环境变量"
fi

