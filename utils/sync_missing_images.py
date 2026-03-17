#!/usr/bin/env python3
"""
根据缺失文件列表，生成同步命令或脚本
帮助用户只复制缺失的图片文件
"""
import os
from pathlib import Path

def generate_sync_commands(missing_file: str, local_images_dir: str, remote_images_dir: str = "/root/autodl-tmp/dataset/images"):
    """
    根据缺失文件列表生成rsync命令
    
    Args:
        missing_file: missing_images.txt文件路径
        local_images_dir: 本地images目录路径
        remote_images_dir: 远程images目录路径
    """
    print("="*60)
    print("生成同步命令")
    print("="*60)
    
    # 读取缺失文件列表
    print(f"\n读取缺失文件列表: {missing_file}")
    with open(missing_file, 'r', encoding='utf-8') as f:
        missing_files = [line.strip() for line in f if line.strip()]
    
    print(f"缺失文件数量: {len(missing_files)}")
    
    # 按前缀分组（更高效）
    prefixes = {}
    for filename in missing_files:
        # 提取前缀，例如 "WOMEN-Blouses_Shirts"
        parts = filename.split('-')
        if len(parts) >= 2:
            prefix = '-'.join(parts[:2])  # WOMEN-Blouses_Shirts
        else:
            prefix = filename.split('_')[0]  # 备用方案
        
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(filename)
    
    print(f"\n文件前缀分组: {len(prefixes)} 个组")
    print(f"每组平均文件数: {len(missing_files) // len(prefixes)}")
    
    # 生成rsync命令（按前缀）
    print(f"\n{'='*60}")
    print("方法1: 使用rsync按前缀同步（推荐）")
    print(f"{'='*60}")
    print("\n在本地机器上运行以下命令（需要替换<服务器地址>）:\n")
    
    print("# 基本rsync命令（跳过已存在文件）")
    print(f"rsync -av --progress --ignore-existing \\")
    print(f'  "{local_images_dir}/" \\')
    print(f"  root@<服务器地址>:{remote_images_dir}/")
    print("\n# 示例（如果本地路径是Windows）:")
    print(f'rsync -av --progress --ignore-existing \\')
    print(f'  "d:/Users/kq/Desktop/大学/大三上/nndl课设/NNDL/dataset/images/" \\')
    print(f'  root@your-server-ip:{remote_images_dir}/')
    
    # 生成包含特定文件的rsync命令（使用--include）
    print(f"\n{'='*60}")
    print("方法2: 只复制缺失的文件（精确）")
    print(f"{'='*60}")
    print("\n注意：这个方法需要先生成包含规则文件\n")
    
    # 生成包含规则文件
    include_file = Path(missing_file).parent / "rsync_include.txt"
    with open(include_file, 'w', encoding='utf-8') as f:
        for filename in missing_files:
            f.write(f"+ {filename}\n")
        f.write("- *\n")  # 排除其他所有文件
    
    print(f"已生成rsync包含规则文件: {include_file}")
    print(f"\n使用以下命令（在本地机器上）:")
    print(f"rsync -av --progress --include-from={include_file.name} \\")
    print(f'  "{local_images_dir}/" \\')
    print(f"  root@<服务器地址>:{remote_images_dir}/")
    
    # 生成PowerShell脚本（Windows用户）
    ps_script = Path(missing_file).parent / "copy_missing_images.ps1"
    with open(ps_script, 'w', encoding='utf-8') as f:
        f.write("# PowerShell脚本：只复制缺失的图片文件\n")
        f.write(f"# 缺失文件数量: {len(missing_files)}\n\n")
        f.write("$LocalImagesDir = 'd:/Users/kq/Desktop/大学/大三上/nndl课设/NNDL/dataset/images'\n")
        f.write("$RemoteImagesDir = '/root/autodl-tmp/dataset/images'\n")
        f.write("$ServerAddress = 'your-server-ip'\n\n")
        f.write("$MissingFiles = Get-Content 'missing_images.txt'\n\n")
        f.write("Write-Host '开始复制缺失的图片文件...'\n")
        f.write("$Count = 0\n")
        f.write("foreach ($file in $MissingFiles) {\n")
        f.write("    $localPath = Join-Path $LocalImagesDir $file\n")
        f.write("    if (Test-Path $localPath) {\n")
        f.write("        scp $localPath root@$ServerAddress`:${RemoteImagesDir}/\n")
        f.write("        $Count++\n")
        f.write("        Write-Host \"已复制: $file ($Count/$($MissingFiles.Count))\"\n")
        f.write("    }\n")
        f.write("}\n")
        f.write("Write-Host \"完成！共复制 $Count 个文件\"\n")
    
    print(f"\n已生成PowerShell脚本: {ps_script}")
    print("（需要修改服务器地址后运行）")
    
    # 生成简单的bash脚本
    bash_script = Path(missing_file).parent / "copy_missing_images.sh"
    with open(bash_script, 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Bash脚本：只复制缺失的图片文件\n")
        f.write(f"# 缺失文件数量: {len(missing_files)}\n\n")
        f.write("LOCAL_IMAGES_DIR='d:/Users/kq/Desktop/大学/大三上/nndl课设/NNDL/dataset/images'\n")
        f.write("REMOTE_IMAGES_DIR='/root/autodl-tmp/dataset/images'\n")
        f.write("SERVER_ADDRESS='your-server-ip'\n\n")
        f.write("echo '开始复制缺失的图片文件...'\n")
        f.write("COUNT=0\n")
        f.write("while IFS= read -r file; do\n")
        f.write("    if [ -f \"$LOCAL_IMAGES_DIR/$file\" ]; then\n")
        f.write("        scp \"$LOCAL_IMAGES_DIR/$file\" root@$SERVER_ADDRESS:${REMOTE_IMAGES_DIR}/\n")
        f.write("        COUNT=$((COUNT + 1))\n")
        f.write("        echo \"已复制: $file ($COUNT/${len(missing_files)})\"\n")
        f.write("    fi\n")
        f.write("done < missing_images.txt\n")
        f.write("echo \"完成！共复制 $COUNT 个文件\"\n")
    
    os.chmod(bash_script, 0o755)
    print(f"已生成Bash脚本: {bash_script}")
    
    print(f"\n{'='*60}")
    print("推荐使用方法1（rsync）最简单高效")
    print("如果rsync不可用，可以使用生成的脚本")
    print(f"{'='*60}")

if __name__ == "__main__":
    import sys
    
    missing_file = "dataset/missing_images.txt"
    local_images_dir = "d:/Users/kq/Desktop/大学/大三上/nndl课设/NNDL/dataset/images"
    
    if len(sys.argv) > 1:
        missing_file = sys.argv[1]
    if len(sys.argv) > 2:
        local_images_dir = sys.argv[2]
    
    if not os.path.exists(missing_file):
        print(f"错误: 找不到文件 {missing_file}")
        print("请先运行: python3 utils/check_missing_images.py")
        sys.exit(1)
    
    generate_sync_commands(missing_file, local_images_dir)
