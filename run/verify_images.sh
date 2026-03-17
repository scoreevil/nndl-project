#!/bin/bash
# 验证图片同步结果

echo "=========================================="
echo "验证图片同步结果"
echo "=========================================="

cd /root/autodl-tmp

# 检查缺失文件
echo ""
echo "检查缺失文件..."
python3 utils/check_missing_images.py

# 统计文件数量
echo ""
echo "=========================================="
echo "文件统计"
echo "=========================================="
echo "实际图片数量: $(ls dataset/images/*.jpg 2>/dev/null | wc -l)"
echo "预期图片数量: $(python3 -c "import json; print(len(json.load(open('dataset/captions.json'))))")"
