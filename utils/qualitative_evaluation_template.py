"""
定性对比评测框架：人工评分模板
评分维度：语义准确性、属性完整性、流畅度
"""
import json
from pathlib import Path
from typing import List, Dict


QUALITATIVE_EVALUATION_TEMPLATE = """
# 定性对比评测评分表

## 评分说明
- 评分范围：0-5分，5分为最优
- 评分维度：
  1. **语义准确性**：生成描述与图片服饰特征的匹配度
  2. **属性完整性**：是否覆盖'款式/袖型/材质/图案'4类核心属性
  3. **流畅度**：英文表述的自然流畅程度

## 评分表

| 样本ID | 图像路径 | 参考描述 | Model1描述 | Model2描述 | Model3描述 | Model5描述 | GPT-4V描述 | Qwen-VL描述 | Kimi-VL描述 | Doubao-VL描述 |
|--------|---------|---------|-----------|-----------|-----------|-----------|-----------|------------|------------|--------------|
| 001 | sample_001.png | [参考描述] | [描述] | [描述] | [描述] | [描述] | [描述] | [描述] | [描述] | [描述] |
|      |          |         | 准确性:___ | 准确性:___ | 准确性:___ | 准确性:___ | 准确性:___ | 准确性:___ | 准确性:___ | 准确性:___ |
|      |          |         | 完整性:___ | 完整性:___ | 完整性:___ | 完整性:___ | 完整性:___ | 完整性:___ | 完整性:___ | 完整性:___ |
|      |          |         | 流畅度:___ | 流畅度:___ | 流畅度:___ | 流畅度:___ | 流畅度:___ | 流畅度:___ | 流畅度:___ | 流畅度:___ |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## 评分统计

| 模型 | 语义准确性(平均) | 属性完整性(平均) | 流畅度(平均) | 综合得分 |
|------|----------------|----------------|-------------|---------|
| Model1（RNN） | ___ | ___ | ___ | ___ |
| Model2（LSTM） | ___ | ___ | ___ | ___ |
| Model3（注意力RNN） | ___ | ___ | ___ | ___ |
| Model5（全Transformer） | ___ | ___ | ___ | ___ |
| GPT-4V | ___ | ___ | ___ | ___ |
| Qwen-VL | ___ | ___ | ___ | ___ |
| Kimi-VL | ___ | ___ | ___ | ___ |
| Doubao-VL | ___ | ___ | ___ | ___ |
"""


def generate_qualitative_template(
    results_dir: str = "results",
    image_dir: str = "dataset/test_images_224x224",
    captions_file: str = "dataset/captions.json",
    images_dir: str = "dataset/images",
    num_samples: int = 100,
    output_file: str = "results/qualitative_evaluation_template.md"
):
    """
    生成定性评测模板
    
    Args:
        results_dir: 结果目录
        image_dir: 测试图像目录
        captions_file: captions.json文件路径
        images_dir: 原始图像目录
        num_samples: 样本数量
        output_file: 输出文件路径
    """
    print("="*60)
    print("生成定性评测模板")
    print("="*60)
    
    # 加载所有生成结果
    model_files = {
        "Model1（RNN）": "results/self_model1_generated.txt",
        "Model2（LSTM）": "results/self_model2_generated.txt",
        "Model3（注意力RNN）": "results/self_model3_generated.txt",
        "Model5（全Transformer）": "results/self_model5_generated.txt",
        "GPT-4V": "results/lmm_gpt_generated.txt",
        "Qwen-VL": "results/lmm_qwen_generated.txt",
        "Kimi-VL": "results/lmm_kimi_generated.txt",
        "Doubao-VL": "results/lmm_doubao_generated.txt",
    }
    
    all_captions = {}
    for model_name, file_path in model_files.items():
        captions = {}
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        sample_id, caption = parts
                        captions[sample_id] = caption
        all_captions[model_name] = captions
    
    # 加载参考描述
    from .data_loader import load_and_validate_dataset
    _, _, test_data = load_and_validate_dataset(
        captions_file=captions_file,
        images_dir=images_dir,
        random_seed=42
    )
    
    num_samples = min(num_samples, len(test_data))
    
    # 生成Markdown模板
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 定性对比评测评分表\n\n")
        f.write("## 评分说明\n")
        f.write("- 评分范围：0-5分，5分为最优\n")
        f.write("- 评分维度：\n")
        f.write("  1. **语义准确性**：生成描述与图片服饰特征的匹配度\n")
        f.write("  2. **属性完整性**：是否覆盖\'款式/袖型/材质/图案\'4类核心属性\n")
        f.write("  3. **流畅度**：英文表述的自然流畅程度\n\n")
        f.write("## 评分表\n\n")
        
        # 表头
        f.write("| 样本ID | 图像路径 | 参考描述 | ")
        f.write(" | ".join(model_files.keys()))
        f.write(" |\n")
        f.write("|" + "|".join(["--------"] * (len(model_files) + 3)) + "|\n")
        
        # 数据行
        for idx in range(num_samples):
            sample_id = f"{idx+1:03d}"
            image_path = f"sample_{sample_id}.png"
            
            # 获取参考描述
            if idx < len(test_data):
                ref_captions = test_data[idx].get('captions', [])
                ref_desc = ref_captions[0] if ref_captions else ""
            else:
                ref_desc = ""
            
            # 获取各模型生成描述
            model_descriptions = []
            for model_name in model_files.keys():
                desc = all_captions.get(model_name, {}).get(sample_id, "")
                if desc == "generation_failed":
                    desc = "[生成失败]"
                model_descriptions.append(desc[:50] + "..." if len(desc) > 50 else desc)
            
            # 写入数据行
            f.write(f"| {sample_id} | {image_path} | {ref_desc[:50]}... | ")
            f.write(" | ".join(model_descriptions))
            f.write(" |\n")
            
            # 评分行
            f.write("| | | | ")
            f.write(" | ".join(["准确性:___ 完整性:___ 流畅度:___"] * len(model_files)))
            f.write(" |\n")
        
        f.write("\n## 评分统计\n\n")
        f.write("| 模型 | 语义准确性(平均) | 属性完整性(平均) | 流畅度(平均) | 综合得分 |\n")
        f.write("|------|----------------|----------------|-------------|---------|\n")
        
        for model_name in model_files.keys():
            f.write(f"| {model_name} | ___ | ___ | ___ | ___ |\n")
    
    print(f"\n模板已保存到: {output_path}")
    print("="*60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成定性评测模板")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="结果目录")
    parser.add_argument("--image_dir", type=str, 
                       default="dataset/test_images_224x224",
                       help="测试图像目录")
    parser.add_argument("--captions_file", type=str, default="dataset/captions.json",
                       help="captions.json文件路径")
    parser.add_argument("--images_dir", type=str, default="dataset/images",
                       help="原始图像目录")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="样本数量")
    parser.add_argument("--output_file", type=str,
                       default="results/qualitative_evaluation_template.md",
                       help="输出文件路径")
    
    args = parser.parse_args()
    
    generate_qualitative_template(
        results_dir=args.results_dir,
        image_dir=args.image_dir,
        captions_file=args.captions_file,
        images_dir=args.images_dir,
        num_samples=args.num_samples,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()

