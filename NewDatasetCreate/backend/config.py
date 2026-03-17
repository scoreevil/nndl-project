"""
配置文件：模型路径、API密钥、图像目录等
"""
import os
from pathlib import Path

# 项目根目录
# 支持从不同位置运行
_config_file = Path(__file__).resolve()
if _config_file.parent.name == 'backend':
    # 从backend目录运行
    PROJECT_ROOT = _config_file.parent.parent.parent
    NEWDATASET_DIR = _config_file.parent.parent
else:
    # 从其他位置运行
    PROJECT_ROOT = _config_file.parent.parent
    NEWDATASET_DIR = _config_file.parent

# 图像目录
IMAGE_DIR = PROJECT_ROOT / "newdataset" / "images"

# 自研模型配置
# 注意：模型文件名为 model2_checkpoint.pt（Model3/注意力RNN）
SELF_MODEL_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "model2_checkpoint.pt"
SELF_MODEL_CONFIG = {
    "vocab_size": 10000,
    "embed_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "ff_dim": 2048,
    "max_seq_len": 50
}

# LMMs API配置
LMMS_CONFIG = {
    "gpt": {
        "api_key": os.getenv("OPENAI_API_KEY", "sk-GXePMPQwBVHMFqCfaAE78fXtN8QIr3M4uUoX0ymciCEwM3Af"),
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "enabled": True
    },
    "qwen": {
        "api_key": os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY", "sk-cc8358b29dcd4a158f94f1f80f2f0ca8"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-plus",
        "enabled": True
    },
    "kimi": {
        "api_key": os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY", "sk-hVoVZ80U3N3y5r5Xr9t0yAylE79bQyEjYK4bNYzZRykkq8Gm"),
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-32k-vision-preview",
        "enabled": True
    },
    "doubao": {
        "api_key": os.getenv("ARK_API_KEY") or os.getenv("DOUBAO_API_KEY", "2fa5010e-08bb-4dd5-872c-f61d9d67f708"),
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seed-1-6-251015",
        "enabled": True
    }
}

# 默认使用的LMM
DEFAULT_LMM = "qwen"

# 标注数据保存路径
ANNOTATION_JSON = NEWDATASET_DIR / "annotations.json"
OUTPUT_NPZ = NEWDATASET_DIR / "new_dataset.npz"
TEMP_CROPPED_DIR = NEWDATASET_DIR / "temp_cropped"
TEMP_CROPPED_DIR.mkdir(parents=True, exist_ok=True)

# 界面配置
IMAGE_SIZE = (224, 224)
ITEMS_PER_PAGE = 10

# LMMs Prompt模板
LMM_PROMPT = """Analyze the entire fashion image and output 3 English sentences (text only, no extra explanations):
Clothing description (≤25 words): Include style, sleeve type, material, pattern;
Background description 1 (≤15 words): Describe overall scene (e.g., street/cafe);
Background description 2 (≤15 words): Describe background details (e.g., coffee shops/trees);
Format: Clothing description | Background description 1 | Background description 2"""

