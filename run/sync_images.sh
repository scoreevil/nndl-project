#!/bin/bash
# 同步本地图片到服务器（只复制缺失的文件）

# 使用方法：
# 在本地机器上运行：rsync -av --progress /本地路径/dataset/images/ user@server:/root/autodl-tmp/dataset/images/
# 或者在服务器上运行：bash run/sync_images.sh

echo "=========================================="
echo "图片同步脚本"
echo "=========================================="
echo ""
echo "注意事项："
echo "1. 这个脚本用于从本地同步图片到服务器"
echo "2. 需要在本地机器上运行rsync命令"
echo "3. rsync会自动跳过已存在的相同文件，只复制新文件"
echo ""
echo "推荐命令（在本地机器上运行）："
echo ""
echo "rsync -av --progress --ignore-existing \\"
echo "  d:/Users/kq/Desktop/大学/大三上/nndl课设/NNDL/dataset/images/ \\"
echo "  root@your-server:/root/autodl-tmp/dataset/images/"
echo ""
echo "或者使用VS Code的Remote-SSH扩展，在本地终端运行："
echo ""
echo "rsync -av --progress \\"
echo "  /本地路径/dataset/images/ \\"
echo "  root@server-ip:/root/autodl-tmp/dataset/images/"
echo ""
echo "参数说明："
echo "  -a: 归档模式，保持文件属性"
echo "  -v: 详细输出"
echo "  --progress: 显示进度"
echo "  --ignore-existing: 跳过已存在的文件（不覆盖）"
echo "  --update: 只更新较新的文件"
echo ""
