"""
完整评测流程：整合所有步骤
1. 准备测试集图像
2. 调用LMMs API生成描述
3. 自研模型推理生成描述
4. 定量对比评测
5. 生成定性评测模板
"""
import subprocess
import sys
import os
from pathlib import Path

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# 切换到项目根目录
os.chdir(PROJECT_ROOT)

# 添加项目根目录到Python路径
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_command(cmd: list, description: str):
    """
    运行命令
    
    Args:
        cmd: 命令列表
        description: 命令描述
    """
    print("\n" + "="*60)
    print(f"{description}")
    print("="*60)
    print(f"执行命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"错误: {description} 失败")
        print(f"错误输出: {result.stderr}")
        return False
    else:
        print(f"成功: {description}")
        if result.stdout:
            print(result.stdout)
        return True


def main():
    """主函数：运行完整评测流程"""
    import argparse
    
    parser = argparse.ArgumentParser(description="完整评测流程")
    parser.add_argument("--skip_prepare_images", action="store_true",
                       help="跳过准备图像步骤")
    parser.add_argument("--skip_lmms", action="store_true",
                       help="跳过LMMs API调用")
    parser.add_argument("--skip_self_models", action="store_true",
                       help="跳过自研模型推理")
    parser.add_argument("--lmms_only", type=str, nargs="+",
                       choices=["gpt", "qwen", "kimi", "doubao"],
                       help="只运行指定的LMMs")
    parser.add_argument("--self_models_only", type=str, nargs="+",
                       choices=["model1", "model1b", "model2", "model5"],
                       help="只运行指定的自研模型")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="样本数量")
    
    args = parser.parse_args()
    
    print("="*60)
    print("完整评测流程")
    print("="*60)
    
    # 步骤1: 准备测试集图像
    if not args.skip_prepare_images:
        success = run_command(
            [sys.executable, "-m", "utils.prepare_test_images", 
             "--num_samples", str(args.num_samples)],
            "步骤1: 准备测试集图像"
        )
        if not success:
            print("错误: 准备图像失败，退出")
            return
    
    # 步骤2: 调用LMMs API
    if not args.skip_lmms:
        lmms = args.lmms_only if args.lmms_only else ["gpt", "qwen", "kimi", "doubao"]
        
        for lmm_type in lmms:
            success = run_command(
                [sys.executable, "-m", "utils.call_lmms_api",
                 "--lmm_type", lmm_type,
                 "--num_samples", str(args.num_samples)],
                f"步骤2: 调用 {lmm_type.upper()} API"
            )
            if not success:
                print(f"警告: {lmm_type} API调用失败，继续处理其他模型")
    
    # 步骤3: 自研模型推理
    if not args.skip_self_models:
        # 需要指定checkpoint路径，这里使用默认路径（实际使用时需要根据实际情况调整）
        model_configs = {
            "model1": {
                "checkpoint": "checkpoints/model1_checkpoint.pt",
                "name": "Model1（RNN）"
            },
            "model1b": {
                "checkpoint": "checkpoints/model1b_checkpoint.pt",
                "name": "Model2（LSTM）"
            },
            "model2": {
                "checkpoint": "checkpoints/model2_checkpoint.pt",
                "name": "Model3（注意力RNN）"
            },
            "model5": {
                "checkpoint": "checkpoints/model5_checkpoint.pt",
                "name": "Model5（全Transformer）"
            }
        }
        
        models = args.self_models_only if args.self_models_only else list(model_configs.keys())
        
        for model_type in models:
            config = model_configs.get(model_type)
            if not config:
                print(f"警告: 未找到模型配置 {model_type}，跳过")
                continue
            
            checkpoint = config["checkpoint"]
            if not Path(checkpoint).exists():
                print(f"警告: checkpoint文件不存在: {checkpoint}，跳过 {config['name']}")
                continue
            
            success = run_command(
                [sys.executable, "-m", "utils.generate_self_model_captions",
                 "--model_type", model_type,
                 "--checkpoint", checkpoint,
                 "--num_samples", str(args.num_samples)],
                f"步骤3: {config['name']} 推理"
            )
            if not success:
                print(f"警告: {config['name']} 推理失败，继续处理其他模型")
    
    # 步骤4: 定量对比评测
    success = run_command(
        [sys.executable, "-m", "utils.evaluate_comparison",
         "--num_samples", str(args.num_samples)],
        "步骤4: 定量对比评测"
    )
    if not success:
        print("警告: 定量对比评测失败")
    
    # 步骤5: 生成定性评测模板
    success = run_command(
        [sys.executable, "-m", "utils.qualitative_evaluation_template",
         "--num_samples", str(args.num_samples)],
        "步骤5: 生成定性评测模板"
    )
    if not success:
        print("警告: 生成定性评测模板失败")
    
    print("\n" + "="*60)
    print("完整评测流程结束")
    print("="*60)
    print("\n结果文件:")
    print("  - 定量评测报告: results/comparison_report.md")
    print("  - 定性评测模板: results/qualitative_evaluation_template.md")
    print("  - 生成结果: results/*_generated.txt")


if __name__ == "__main__":
    main()

