"""
验证配置文件
"""
from config import SELF_MODEL_CHECKPOINT, LMMS_CONFIG, IMAGE_DIR

print("="*60)
print("配置验证")
print("="*60)

# 验证模型路径
print(f"\n模型路径: {SELF_MODEL_CHECKPOINT}")
print(f"模型存在: {SELF_MODEL_CHECKPOINT.exists()}")

# 验证图像目录
print(f"\n图像目录: {IMAGE_DIR}")
print(f"图像目录存在: {IMAGE_DIR.exists()}")

# 验证LMMs API密钥
print("\nLMMs API密钥配置:")
for name, cfg in LMMS_CONFIG.items():
    key = cfg.get('api_key', '')
    if key:
        masked_key = key[:20] + "..." if len(key) > 20 else key
        print(f"  {name.upper()}: 已配置 ({masked_key})")
    else:
        print(f"  {name.upper()}: 未配置")

print("\n" + "="*60)

