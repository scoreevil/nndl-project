"""
Utils模块：包含数据准备、LMMs API调用、模型推理、评测等功能
"""

# 注意：不在这里导入所有模块，避免循环导入和初始化问题
# 使用时直接导入：from utils.module_name import function_name
# 或：import utils.module_name

__all__ = [
    'prepare_test_images',
    'call_lmms_api',
    'generate_self_model_captions',
    'evaluate_comparison',
    'qualitative_evaluation_template',
    'lmm_api_client'
]

