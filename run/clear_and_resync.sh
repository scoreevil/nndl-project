#!/bin/bash
# 清空图片目录并准备重新同步

set -e

cd /root/autodl-tmp

echo "=========================================="
echo "清空图片目录"
echo "=========================================="
echo ""
echo "⚠️  警告：这将删除所有现有图片文件！"
echo "当前图片数量: $(ls dataset/images/*.jpg 2>/dev/null | wc -l)"
echo "占用空间: $(du -sh dataset/images/ | cut -f1)"
echo ""
read -p "确认删除所有图片文件？(yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "操作已取消"
    exit 0
fi

echo ""
echo "正在删除所有图片文件..."
rm -f dataset/images/*.jpg
rm -f dataset/images/*.jpeg
rm -f dataset/images/*.png

echo "✅ 删除完成！"
echo ""
echo "当前图片数量: $(ls dataset/images/*.jpg 2>/dev/null | wc -l)"
echo ""
echo "=========================================="
echo "下一步操作："
echo "=========================================="
echo "现在可以在VS Code中重新复制所有图片文件了"
echo "1. 打开本地的 images 目录"
echo "2. 全选所有文件（Ctrl+A 或 Cmd+A）"
echo "3. 复制到服务器的 dataset/images/ 目录"
echo "4. 这次不会有'文件已存在'的冲突了"
echo ""
