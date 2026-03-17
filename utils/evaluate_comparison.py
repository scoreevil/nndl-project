"""
对比评测脚本：自研模型 vs LMMs
计算定量指标并生成对比报告
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from .model_evaluator import ModelEvaluator
from .data_loader import load_and_validate_dataset


def load_generated_captions(file_path: str) -> Dict[str, str]:
    """
    加载生成的描述文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        字典，key为sample_id，value为描述文本
    """
    captions = {}
    if not Path(file_path).exists():
        print(f"警告: 文件不存在: {file_path}")
        return captions
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                sample_id, caption = parts
                captions[sample_id] = caption
    
    return captions


def load_reference_captions(
    captions_file: str,
    images_dir: str,
    num_samples: int = 100
) -> List[List[str]]:
    """
    加载参考描述
    
    Args:
        captions_file: captions.json文件路径
        images_dir: 图像目录路径
        num_samples: 样本数量
        
    Returns:
        参考描述列表，每个元素是一个描述列表（可能有多个参考描述）
    """
    _, _, test_data = load_and_validate_dataset(
        captions_file=captions_file,
        images_dir=images_dir,
        random_seed=42
    )
    
    # 限制样本数量
    num_samples = min(num_samples, len(test_data))
    
    references = []
    for item in test_data[:num_samples]:
        captions = item.get('captions', [])
        if len(captions) > 0:
            references.append(captions)
        else:
            references.append([''])
    
    return references


def evaluate_model_results(
    generated_captions: Dict[str, str],
    reference_captions: List[List[str]],
    model_name: str
) -> Dict[str, float]:
    """
    评估模型生成结果
    
    Args:
        generated_captions: 生成的描述字典
        reference_captions: 参考描述列表
        model_name: 模型名称
        
    Returns:
        评估结果字典
    """
    evaluator = ModelEvaluator()
    
    # 准备候选和参考列表
    candidates = []
    references = []
    
    for idx in range(len(reference_captions)):
        sample_id = f"{idx+1:03d}"
        caption = generated_captions.get(sample_id, "")
        
        # 过滤失败样本
        if caption == "generation_failed" or not caption.strip():
            continue
        
        candidates.append(caption)
        references.append(reference_captions[idx])
    
    if len(candidates) == 0:
        print(f"警告: {model_name} 没有有效生成结果")
        return {
            'spice_mean': 0.0,
            'rouge_l_mean': 0.0,
            'meteor_mean': 0.0,
            'cider_d_mean': 0.0,
            'valid_count': 0,
            'total_count': len(reference_captions)
        }
    
    # 评估
    results = evaluator.evaluate(candidates, references)
    
    results['valid_count'] = len(candidates)
    results['total_count'] = len(reference_captions)
    
    return results


def calculate_inference_time(file_path: str) -> float:
    """
    从生成结果文件中提取平均推理时间（如果记录了时间）
    这里简化处理，返回0（实际可以从日志或单独的时间记录文件读取）
    
    Args:
        file_path: 文件路径
        
    Returns:
        平均推理时间（毫秒）
    """
    # 实际实现中可以从时间记录文件读取
    # 这里返回0，表示未记录
    return 0.0


def generate_comparison_report(
    results_dir: str = "results",
    captions_file: str = "dataset/captions.json",
    images_dir: str = "dataset/images",
    num_samples: int = 100,
    output_file: str = "results/comparison_report.md"
):
    """
    生成对比评测报告
    
    Args:
        results_dir: 结果目录
        captions_file: captions.json文件路径
        images_dir: 图像目录路径
        num_samples: 样本数量
        output_file: 输出文件路径
    """
    print("="*60)
    print("生成对比评测报告")
    print("="*60)
    
    # 加载参考描述
    print("\n加载参考描述...")
    reference_captions = load_reference_captions(
        captions_file, images_dir, num_samples
    )
    print(f"[OK] 加载了 {len(reference_captions)} 个参考描述")
    
    # 定义模型列表
    models = [
        # 自研模型
        {"name": "Model1（RNN）", "type": "self", "file": "results/self_model1_generated.txt"},
        {"name": "Model2（LSTM）", "type": "self", "file": "results/self_model2_generated.txt"},
        {"name": "Model3（注意力RNN）", "type": "self", "file": "results/self_model3_generated.txt"},
        {"name": "Model5（全Transformer）", "type": "self", "file": "results/self_model5_generated.txt"},
        # LMMs
        {"name": "GPT-4V", "type": "lmm", "file": "results/lmm_gpt_generated.txt"},
        {"name": "Qwen-VL", "type": "lmm", "file": "results/lmm_qwen_generated.txt"},
        {"name": "Kimi-VL", "type": "lmm", "file": "results/lmm_kimi_generated.txt"},
        {"name": "Doubao-VL", "type": "lmm", "file": "results/lmm_doubao_generated.txt"},
    ]
    
    # 评估所有模型
    print("\n开始评估模型...")
    
    # 检查是否有任何生成结果文件
    has_any_results = False
    for model_info in models:
        file_path = model_info["file"]
        if Path(file_path).exists():
            has_any_results = True
            break
    
    if not has_any_results:
        print("\n" + "="*60)
        print("警告: 没有找到任何生成结果文件！")
        print("="*60)
        print("\n请先运行生成步骤：")
        print("1. 自研模型推理: python -m utils.generate_self_model_captions --model_type <model> --checkpoint <path>")
        print("2. LMMs API调用: python -m utils.call_lmms_api --lmm_type <lmm>")
        print("\n或使用快速开始脚本: bash run/quick_start.sh")
        print("="*60)
        return
    
    all_results = []
    
    for model_info in models:
        model_name = model_info["name"]
        file_path = model_info["file"]
        
        print(f"\n评估 {model_name}...")
        
        # 加载生成结果
        generated_captions = load_generated_captions(file_path)
        
        if not generated_captions:
            print(f"  警告: {model_name} 没有生成结果，跳过")
            print(f"  提示: 需要先运行生成步骤生成 {file_path}")
            continue
        
        # 评估
        eval_results = evaluate_model_results(
            generated_captions, reference_captions, model_name
        )
        
        # 计算有效生成率
        valid_rate = (eval_results['valid_count'] / eval_results['total_count'] * 100) if eval_results['total_count'] > 0 else 0
        
        # 获取推理时间（如果有记录）
        inference_time = calculate_inference_time(file_path)
        
        all_results.append({
            'name': model_name,
            'type': model_info['type'],
            'spice': eval_results.get('spice_mean', 0.0),
            'rouge_l': eval_results.get('rouge_l_mean', 0.0),
            'meteor': eval_results.get('meteor_mean', 0.0),
            'cider_d': eval_results.get('cider_d_mean', 0.0),
            'valid_rate': valid_rate,
            'inference_time': inference_time
        })
        
        print(f"  SPICE: {eval_results.get('spice_mean', 0.0):.4f}")
        print(f"  ROUGE-L: {eval_results.get('rouge_l_mean', 0.0):.4f}")
        print(f"  METEOR: {eval_results.get('meteor_mean', 0.0):.4f}")
        print(f"  CIDEr-D: {eval_results.get('cider_d_mean', 0.0):.4f}")
        print(f"  有效生成率: {valid_rate:.2f}%")
    
    # 生成Markdown报告
    print("\n生成Markdown报告...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 自研模型 vs LMMs 对比评测报告\n\n")
        f.write("## 定量对比评测结果\n\n")
        f.write("| 模型类型 | 具体模型 | SPICE | ROUGE-L | METEOR | CIDEr | 有效生成率 | 推理耗时（ms/样本） |\n")
        f.write("|---------|---------|-------|---------|--------|--------|-----------|-------------------|\n")
        
        for result in all_results:
            model_type = "自研模型" if result['type'] == 'self' else "LMMs"
            f.write(f"| {model_type} | {result['name']} | "
                   f"{result['spice']:.4f} | {result['rouge_l']:.4f} | "
                   f"{result['meteor']:.4f} | {result['cider_d']:.4f} | "
                   f"{result['valid_rate']:.2f}% | "
                   f"{result['inference_time']:.2f} |\n")
        
        f.write("\n## 说明\n\n")
        f.write("- **SPICE**: 语义相似度指标\n")
        f.write("- **ROUGE-L**: 最长公共子序列指标\n")
        f.write("- **METEOR**: 考虑同义词的翻译质量指标\n")
        f.write("- **CIDEr**: 基于共识的图像描述评估指标（CIDEr-D）\n")
        f.write("- **有效生成率**: 成功生成描述的样本占比\n")
        f.write("- **推理耗时**: 每个样本的平均推理时间（毫秒）\n")
    
    print(f"\n报告已保存到: {output_path}")
    print("="*60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="对比评测脚本")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="结果目录")
    parser.add_argument("--captions_file", type=str, default="dataset/captions.json",
                       help="captions.json文件路径")
    parser.add_argument("--images_dir", type=str, default="dataset/images",
                       help="图像目录路径")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="样本数量")
    parser.add_argument("--output_file", type=str, 
                       default="results/comparison_report.md",
                       help="输出文件路径")
    
    args = parser.parse_args()
    
    generate_comparison_report(
        results_dir=args.results_dir,
        captions_file=args.captions_file,
        images_dir=args.images_dir,
        num_samples=args.num_samples,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()

