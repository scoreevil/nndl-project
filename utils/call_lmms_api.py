"""
调用LMMs API生成图像描述
"""
import os
import time
from pathlib import Path
from typing import List, Dict
from .lmm_api_client import create_lmm_client


# 统一的Prompt模板
UNIFIED_PROMPT = """Please describe the clothing in the image concisely in English. Requirements:
Include core attributes: style (e.g., tank top/dress), sleeve type (e.g., sleeves cut off), material (e.g., cotton), pattern (e.g., floral/solid color);
Keep the length within 25 words;
Output only the description text, without additional explanations or punctuation;
Avoid repetitive phrases, strictly match the clothing features in the image."""


def call_lmm_api(
    lmm_type: str,
    image_dir: str,
    output_file: str,
    prompt: str = UNIFIED_PROMPT,
    num_samples: int = 100,
    api_key: str = None
):
    """
    调用LMM API生成描述
    
    Args:
        lmm_type: LMM类型 ("gpt", "qwen", "kimi", "doubao")
        image_dir: 图像目录路径
        output_file: 输出文件路径
        prompt: 提示词
        num_samples: 样本数量
        api_key: API密钥（可选，会从环境变量读取）
    """
    print("="*60)
    print(f"调用 {lmm_type.upper()} API生成描述")
    print("="*60)
    
    # 创建客户端
    try:
        client = create_lmm_client(lmm_type, api_key)
    except Exception as e:
        print(f"错误: 无法创建{lmm_type}客户端: {e}")
        return
    
    # 加载图像路径
    image_path = Path(image_dir)
    if not image_path.exists():
        print(f"错误: 图像目录不存在: {image_dir}")
        return
    
    # 获取所有图像文件
    image_files = sorted([f for f in image_path.glob("sample_*.png")])
    if len(image_files) < num_samples:
        print(f"警告: 找到的图像文件数({len(image_files)})少于请求数({num_samples})")
        num_samples = len(image_files)
    
    print(f"\n开始处理 {num_samples} 个样本...")
    
    results = []
    start_time = time.time()
    consecutive_failures = 0  # 连续失败计数
    max_consecutive_failures = 2  # 连续失败阈值，超过此值则跳过该模型
    
    for idx, img_file in enumerate(image_files[:num_samples]):
        sample_id = f"{idx+1:03d}"
        print(f"\n处理样本 {sample_id}: {img_file.name}")
        
        # 调用API
        sample_start = time.time()
        caption = client.generate_caption(str(img_file), prompt)
        sample_time = (time.time() - sample_start) * 1000  # 转换为毫秒
        
        # 检查是否失败
        if caption == "generation_failed":
            consecutive_failures += 1
            print(f"  警告: 生成失败（连续失败 {consecutive_failures} 次）")
            
            # 如果连续失败次数过多，可能是网络问题，跳过该模型
            if consecutive_failures >= max_consecutive_failures:
                print(f"\n警告: 连续 {consecutive_failures} 次失败，可能是网络连接问题")
                print(f"跳过 {lmm_type.upper()}，继续处理其他模型")
                # 为剩余样本填充失败标记
                for remaining_idx in range(idx, num_samples):
                    remaining_id = f"{remaining_idx+1:03d}"
                    results.append({
                        'sample_id': remaining_id,
                        'caption': 'generation_failed',
                        'time_ms': 0
                    })
                break
        else:
            consecutive_failures = 0  # 重置连续失败计数
        
        # 保存结果
        results.append({
            'sample_id': sample_id,
            'caption': caption,
            'time_ms': sample_time
        })
        
        if caption != "generation_failed":
            print(f"  生成描述: {caption[:50]}..." if len(caption) > 50 else f"  生成描述: {caption}")
            print(f"  耗时: {sample_time:.2f} ms")
        
        # 避免API限流
        if idx < num_samples - 1:
            time.sleep(0.5)
    
    total_time = time.time() - start_time
    avg_time = total_time / num_samples * 1000 if num_samples > 0 else 0
    
    # 统计有效生成
    valid_count = sum(1 for r in results if r['caption'] != "generation_failed")
    success_rate = valid_count / num_samples * 100 if num_samples > 0 else 0
    
    print(f"\n处理完成:")
    print(f"  总样本数: {num_samples}")
    print(f"  有效生成: {valid_count}")
    print(f"  成功率: {success_rate:.2f}%")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  平均耗时: {avg_time:.2f} ms/样本")
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"{result['sample_id']}\t{result['caption']}\n")
    
    print(f"\n结果已保存到: {output_path}")
    print("="*60)


def main():
    """主函数：调用所有LMMs API"""
    import argparse
    
    parser = argparse.ArgumentParser(description="调用LMMs API生成描述")
    parser.add_argument("--lmm_type", type=str, required=True,
                       choices=["gpt", "qwen", "kimi", "doubao"],
                       help="LMM类型")
    parser.add_argument("--image_dir", type=str, 
                       default="dataset/test_images_224x224",
                       help="图像目录路径")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="输出目录")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="样本数量")
    parser.add_argument("--api_key", type=str, default=None,
                       help="API密钥（可选，会从环境变量读取）")
    
    args = parser.parse_args()
    
    # 确定输出文件名
    output_files = {
        "gpt": "results/lmm_gpt_generated.txt",
        "qwen": "results/lmm_qwen_generated.txt",
        "kimi": "results/lmm_kimi_generated.txt",
        "doubao": "results/lmm_doubao_generated.txt"
    }
    
    output_file = output_files.get(args.lmm_type, 
                                   f"results/lmm_{args.lmm_type}_generated.txt")
    
    call_lmm_api(
        lmm_type=args.lmm_type,
        image_dir=args.image_dir,
        output_file=output_file,
        num_samples=args.num_samples,
        api_key=args.api_key
    )


if __name__ == "__main__":
    main()

