"""
模型调用工具函数（延迟导入torch，避免NumPy冲突）
"""
from pathlib import Path
import sys
import os
import base64
import requests
from typing import Tuple, Optional
import json

# 添加项目根目录到路径
if Path(__file__).parent.name == 'backend':
    PROJECT_ROOT = Path(__file__).parent.parent.parent
else:
    PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .config import SELF_MODEL_CHECKPOINT, SELF_MODEL_CONFIG, LMMS_CONFIG, DEFAULT_LMM, LMM_PROMPT
except ImportError:
    try:
        from backend.config import SELF_MODEL_CHECKPOINT, SELF_MODEL_CONFIG, LMMS_CONFIG, DEFAULT_LMM, LMM_PROMPT
    except ImportError:
        from config import SELF_MODEL_CHECKPOINT, SELF_MODEL_CONFIG, LMMS_CONFIG, DEFAULT_LMM, LMM_PROMPT


def self_model_infer(cropped_image_path: str, model_type: str = "model2") -> str:
    """
    使用自研模型生成服饰描述（延迟导入torch）
    默认使用 model2 (Model3/注意力RNN)
    """
    # 延迟导入torch，避免NumPy版本冲突
    import torch
    import torchvision.transforms as transforms
    
    try:
        project_root = PROJECT_ROOT
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from utils.feature_extractor import ResNetFeatureExtractor
        from utils.generate_self_model_captions import load_model
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 查找vocab文件
        vocab_file = project_root / "dataset" / "vocab.json"
        if not vocab_file.exists():
            vocab_file = project_root / "vocab.json"
            if not vocab_file.exists():
                raise FileNotFoundError(f"词汇表文件不存在: {vocab_file}")
        
        # 查找checkpoint文件
        checkpoint_path = SELF_MODEL_CHECKPOINT
        if not checkpoint_path.exists():
            if model_type == "model5":
                # 尝试不同的文件名（优先使用model5_checkpoint.pt）
                checkpoint_path = project_root / "models" / "checkpoints" / "model5_checkpoint.pt"
                if not checkpoint_path.exists():
                    checkpoint_path = project_root / "models" / "checkpoints" / "model5_transformer_best.pt"
            elif model_type == "model2":
                checkpoint_path = project_root / "models" / "checkpoints" / "model2_checkpoint.pt"
                if not checkpoint_path.exists():
                    checkpoint_path = project_root / "models" / "checkpoints" / "model2_lstm_best.pt"
            elif model_type == "model1b":
                checkpoint_path = project_root / "models" / "checkpoints" / "model1b_checkpoint.pt"
                if not checkpoint_path.exists():
                    checkpoint_path = project_root / "models" / "checkpoints" / "model1b_lstm_best.pt"
            else:
                checkpoint_path = project_root / "models" / "checkpoints" / "model1_checkpoint.pt"
                if not checkpoint_path.exists():
                    checkpoint_path = project_root / "models" / "checkpoints" / "model1_rnn_best.pt"
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"模型checkpoint不存在: {checkpoint_path}")
        
        # 加载模型
        model, idx2word, vocab_size = load_model(model_type, str(checkpoint_path), str(vocab_file), str(device))
        
        # 提取图像特征
        feature_extractor = ResNetFeatureExtractor(device=str(device))
        with torch.no_grad():
            # extract_features_batch返回3个值: (global_feats, local_feats, valid_indices)
            global_feat_np, local_feat_np, valid_indices = feature_extractor.extract_features_batch([cropped_image_path])
        
        if global_feat_np is None or local_feat_np is None:
            raise RuntimeError("无法提取图像特征，请检查图像路径是否正确")
        
        # 转换为torch tensor
        global_feat = torch.from_numpy(global_feat_np).float()
        local_feat = torch.from_numpy(local_feat_np).float()
        
        # 移动到设备
        global_feat = global_feat.to(device)
        local_feat = local_feat.to(device)
        
        # 生成描述
        max_len = 25
        with torch.no_grad():
            if model_type == "model5":
                generated_sequence = model.generate(local_feat, max_len=max_len, temperature=0.7)
            elif model_type == "model2":
                generated_sequence, _ = model.generate(local_feat, max_len=max_len, 
                                                       temperature=0.7, return_attn=False)
            elif model_type == "model1b":
                generated_sequence = model.generate(local_feat, max_len=max_len, temperature=0.7)
            else:
                generated_sequence = model.generate(local_feat, max_len=max_len)
        
        # 转换为文本
        gen_seq = generated_sequence[0].cpu()
        if hasattr(model, 'postprocess_caption'):
            gen_words = model.postprocess_caption(gen_seq, idx2word)
        else:
            gen_words = []
            for word_idx in gen_seq.tolist():
                if word_idx == 3:  # <END>
                    break
                if word_idx not in [0, 1, 2, 3]:
                    word = idx2word.get(word_idx, f"<{word_idx}>")
                    gen_words.append(word)
        
        description = ' '.join(gen_words) if gen_words else 'generation_failed'
        return description
        
    except Exception as e:
        raise RuntimeError(f"自研模型推理失败: {e}")


def encode_image_to_base64(image_path: str) -> str:
    """将图像编码为base64字符串"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')


def call_lmm_api(image_path: str, lmm_name: Optional[str] = None) -> Tuple[str, str, str]:
    """调用LMMs API生成描述"""
    if lmm_name is None:
        lmm_name = DEFAULT_LMM
    
    config = LMMS_CONFIG.get(lmm_name)
    if not config or not config.get("enabled", False):
        for name, cfg in LMMS_CONFIG.items():
            if cfg.get("enabled", False) and cfg.get("api_key"):
                lmm_name = name
                config = cfg
                break
        else:
            raise RuntimeError("没有可用的LMM API配置")
    
    api_key = config.get("api_key")
    if not api_key:
        raise RuntimeError(f"LMM {lmm_name} 的API密钥未配置")
    
    image_base64 = encode_image_to_base64(image_path)
    image_url = f"data:image/jpeg;base64,{image_base64}"
    
    try:
        if lmm_name == "gpt":
            return _call_openai_api(config, image_url)
        elif lmm_name == "qwen":
            return _call_qwen_api(config, image_url)
        elif lmm_name == "kimi":
            return _call_kimi_api(config, image_url)
        elif lmm_name == "doubao":
            return _call_doubao_api(config, image_url)
        else:
            raise ValueError(f"不支持的LMM: {lmm_name}")
    except Exception as e:
        raise RuntimeError(f"LMM API调用失败: {e}")


def _call_openai_api(config: dict, image_url: str) -> Tuple[str, str, str]:
    """调用OpenAI API"""
    import openai
    
    client = openai.OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"]
    )
    
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": LMM_PROMPT}
                ]
            }
        ],
        max_tokens=150,
        temperature=0.7
    )
    
    text = response.choices[0].message.content.strip()
    return _parse_lmm_response(text)


def _call_qwen_api(config: dict, image_url: str) -> Tuple[str, str, str]:
    """调用Qwen API"""
    import openai
    
    client = openai.OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"]
    )
    
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": LMM_PROMPT}
                ]
            }
        ],
        max_tokens=150,
        temperature=0.7
    )
    
    text = response.choices[0].message.content.strip()
    return _parse_lmm_response(text)


def _call_kimi_api(config: dict, image_url: str) -> Tuple[str, str, str]:
    """调用Kimi API"""
    import openai
    
    client = openai.OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"]
    )
    
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": "你是 Kimi。"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": LMM_PROMPT}
                ]
            }
        ],
        max_tokens=150,
        temperature=0.7
    )
    
    text = response.choices[0].message.content.strip()
    return _parse_lmm_response(text)


def _call_doubao_api(config: dict, image_url: str) -> Tuple[str, str, str]:
    """调用Doubao API"""
    url = f"{config['base_url']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}"
    }
    
    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": LMM_PROMPT}
                ]
            }
        ],
        "max_tokens": 150,
        "temperature": 0.7,
        "thinking": {"type": "disabled"}
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    
    result = response.json()
    text = result["choices"][0]["message"]["content"].strip()
    return _parse_lmm_response(text)


def _parse_lmm_response(text: str) -> Tuple[str, str, str]:
    """解析LMM返回的文本"""
    parts = [p.strip() for p in text.split("|")]
    
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return parts[0], parts[1], ""
    elif len(parts) == 1:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        elif len(lines) == 2:
            return lines[0], lines[1], ""
        else:
            return lines[0] if lines else "", "", ""
    else:
        return "", "", ""


def lmm_infer(full_image_path: str, lmm_name: Optional[str] = None) -> Tuple[str, str, str]:
    """调用LMMs API生成描述（对外接口）"""
    return call_lmm_api(full_image_path, lmm_name)

