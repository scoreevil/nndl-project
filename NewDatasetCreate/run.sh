#!/bin/bash
# 启动Flask服务器

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$(pwd)/..:${PYTHONPATH:-}"

echo "============================================================"
echo "启动服饰图像标注工具 - Web版本"
echo "============================================================"
echo ""
echo "服务器将在 http://localhost:5000 启动"
echo "在浏览器中打开上述地址即可使用"
echo ""
echo "如果远程访问，使用: http://your-server-ip:5000"
echo ""
echo "按 Ctrl+C 停止服务器"
echo "============================================================"
echo ""

# 从项目根目录运行，使用模块方式
python -m backend.app

