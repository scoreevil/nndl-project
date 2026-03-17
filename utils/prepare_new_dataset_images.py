"""
准备新数据集图像：从test/image中选择1000张图片，缩放为224×224，重命名为bg_fashion_0001.jpg格式
"""
import os
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def resize_image(image_path: str, target_size: tuple = (224, 224)) -> Image.Image:
    """
    调整图像大小
    
    Args:
        image_path: 图像路径
        target_size: 目标尺寸
        
    Returns:
        调整后的PIL Image对象
    """
    try:
        img = Image.open(image_path)
        # 转换为RGB模式（处理RGBA等格式）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # 使用高质量重采样
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        return img_resized
    except Exception as e:
        print(f"警告: 无法处理图像 {image_path}: {e}")
        return None


def prepare_new_dataset_images(
    source_dir: str = "test/image",
    output_dir: str = "newdataset/images",
    num_images: int = 1000,
    target_size: tuple = (224, 224),
    random_seed: int = 42,
    prefix: str = "bg_fashion"
):
    """
    准备新数据集图像
    
    Args:
        source_dir: 源图像目录
        output_dir: 输出目录
        num_images: 需要选择的图像数量
        target_size: 目标尺寸
        random_seed: 随机种子
        prefix: 输出文件名前缀
    """
    print("="*60)
    print("准备新数据集图像")
    print("="*60)
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    print(f"\n扫描源目录: {source_dir}")
    source_path = Path(source_dir)
    
    # 获取所有jpg文件
    image_files = list(source_path.glob("*.jpg"))
    image_files.extend(list(source_path.glob("*.JPG")))
    
    if len(image_files) == 0:
        print(f"错误: 在 {source_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 随机选择指定数量的图像
    if len(image_files) < num_images:
        print(f"警告: 源目录中只有 {len(image_files)} 张图像，少于请求的 {num_images} 张")
        selected_files = image_files
    else:
        selected_files = random.sample(image_files, num_images)
    
    print(f"选择了 {len(selected_files)} 张图像进行处理")
    
    # 处理图像
    print(f"\n开始处理图像（调整大小为 {target_size}）...")
    success_count = 0
    failed_count = 0
    
    for idx, img_file in enumerate(tqdm(selected_files, desc="处理图像")):
        try:
            # 调整图像大小
            img_resized = resize_image(str(img_file), target_size)
            
            if img_resized is None:
                failed_count += 1
                continue
            
            # 生成输出文件名
            output_filename = f"{prefix}_{idx+1:04d}.jpg"
            output_filepath = output_path / output_filename
            
            # 保存图像（使用高质量JPEG）
            img_resized.save(
                output_filepath,
                "JPEG",
                quality=95,
                optimize=True
            )
            success_count += 1
            
        except Exception as e:
            print(f"\n错误: 处理 {img_file} 时出错: {e}")
            failed_count += 1
            continue
    
    print("\n" + "="*60)
    print("处理完成")
    print("="*60)
    print(f"成功处理: {success_count} 张")
    print(f"失败: {failed_count} 张")
    print(f"输出目录: {output_path.absolute()}")
    print("="*60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="准备新数据集图像")
    parser.add_argument("--source_dir", type=str, default="test/image",
                       help="源图像目录")
    parser.add_argument("--output_dir", type=str, default="newdataset/images",
                       help="输出目录")
    parser.add_argument("--num_images", type=int, default=1000,
                       help="需要选择的图像数量")
    parser.add_argument("--target_size", type=int, nargs=2, default=[224, 224],
                       help="目标尺寸 (width height)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--prefix", type=str, default="bg_fashion",
                       help="输出文件名前缀")
    
    args = parser.parse_args()
    
    prepare_new_dataset_images(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        target_size=tuple(args.target_size),
        random_seed=args.random_seed,
        prefix=args.prefix
    )


if __name__ == "__main__":
    main()

