"""
LMMs API调用客户端
支持GPT-4V、Qwen-VL、Kimi-VL、Doubao-VL四种大语言多模态模型
"""
import os
import time
import base64
from pathlib import Path
from typing import Optional, Dict, List
import requests
from PIL import Image
import io


class LMMAPIClient:
    """LMMs API调用基类"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化API客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
        """
        self.api_key = api_key or os.getenv(self._get_api_key_env_name())
        self.base_url = base_url
        self.max_retries = 3
        self.timeout = 30
        
    def _get_api_key_env_name(self) -> str:
        """返回API密钥环境变量名（子类需实现）"""
        raise NotImplementedError
    
    def _encode_image(self, image_path: str) -> str:
        """
        将图像编码为base64字符串
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            base64编码的字符串
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    
    def _prepare_image(self, image_path: str, target_size: tuple = (224, 224)) -> str:
        """
        准备图像：调整大小并保存为临时文件
        
        Args:
            image_path: 原始图像路径
            target_size: 目标尺寸，默认(224, 224)
            
        Returns:
            临时文件路径
        """
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # 保存为临时文件
        temp_path = f"/tmp/temp_{int(time.time() * 1000)}.jpg"
        img.save(temp_path, 'JPEG', quality=95)
        return temp_path
    
    def generate_caption(self, image_path: str, prompt: str) -> str:
        """
        生成图像描述
        
        Args:
            image_path: 图像文件路径
            prompt: 提示词
            
        Returns:
            生成的描述文本，失败时返回"generation_failed"
        """
        for attempt in range(self.max_retries):
            try:
                return self._call_api(image_path, prompt)
            except Exception as e:
                error_msg = str(e)
                # 检查是否是网络连接错误
                is_connection_error = any(keyword in error_msg.lower() for keyword in 
                                         ['connection', 'timeout', 'network', 'unreachable', 'dns'])
                
                if is_connection_error:
                    print(f"  尝试 {attempt + 1}/{self.max_retries} 失败: 网络连接错误 ({error_msg[:50]})")
                else:
                    print(f"  尝试 {attempt + 1}/{self.max_retries} 失败: {error_msg[:50]}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    return "generation_failed"
        return "generation_failed"
    
    def _call_api(self, image_path: str, prompt: str) -> str:
        """调用API（子类需实现）"""
        raise NotImplementedError


class GPT4VClient(LMMAPIClient):
    """GPT-4V (GPT-4o) API客户端"""
    
    def _get_api_key_env_name(self) -> str:
        return "OPENAI_API_KEY"
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if not self.api_key:
            raise ValueError("请设置OPENAI_API_KEY环境变量或传入api_key参数")
        self.base_url = "https://api.openai.com/v1"
    
    def _call_api(self, image_path: str, prompt: str) -> str:
        """调用OpenAI GPT-4V API"""
        import openai
        
        client = openai.OpenAI(api_key=self.api_key)
        
        # 准备图像
        temp_path = self._prepare_image(image_path)
        try:
            with open(temp_path, 'rb') as f:
                image_data = f.read()
            
            response = client.chat.completions.create(
                model="gpt-4o",  # 或 "gpt-4-vision-preview"
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64.b64encode(image_data).decode()}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=50,
                top_p=0.9
            )
            
            caption = response.choices[0].message.content.strip()
            return caption
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)


class QwenVLClient(LMMAPIClient):
    """通义千问-多模态 (Qwen-VL) API客户端"""
    
    def _get_api_key_env_name(self) -> str:
        # 支持两种环境变量名：DASHSCOPE_API_KEY 和 QWEN_API_KEY
        return "DASHSCOPE_API_KEY"
    
    def __init__(self, api_key: Optional[str] = None):
        # 优先使用传入的api_key，否则尝试从环境变量读取
        if api_key is None:
            # 先尝试 DASHSCOPE_API_KEY，再尝试 QWEN_API_KEY（向后兼容）
            api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        super().__init__(api_key)
        if not self.api_key:
            raise ValueError("请设置DASHSCOPE_API_KEY或QWEN_API_KEY环境变量或传入api_key参数")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    def _call_api(self, image_path: str, prompt: str) -> str:
        """调用阿里云Qwen-VL API（使用OpenAI兼容接口）"""
        import openai
        
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 准备图像
        temp_path = self._prepare_image(image_path)
        try:
            with open(temp_path, 'rb') as f:
                image_data = f.read()
            
            # 获取图片扩展名（用于base64格式）
            image_ext = os.path.splitext(temp_path)[1].lstrip('.') or 'jpeg'
            
            # 使用base64编码图片，格式：data:image/{ext};base64,{encoded_data}
            image_url = f"data:image/{image_ext};base64,{base64.b64encode(image_data).decode('utf-8')}"
            
            response = client.chat.completions.create(
                model="qwen-vl-plus",  # 使用 qwen-vl-plus 模型
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=50,
                top_p=0.9
            )
            
            caption = response.choices[0].message.content.strip()
            return caption
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class KimiVLClient(LMMAPIClient):
    """Kimi多模态 (Moonshot-VL) API客户端"""
    
    def _get_api_key_env_name(self) -> str:
        # 支持两种环境变量名：MOONSHOT_API_KEY 和 KIMI_API_KEY
        return "MOONSHOT_API_KEY"
    
    def __init__(self, api_key: Optional[str] = None):
        # 优先使用传入的api_key，否则尝试从环境变量读取
        if api_key is None:
            # 先尝试 MOONSHOT_API_KEY，再尝试 KIMI_API_KEY（向后兼容）
            api_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
        super().__init__(api_key)
        if not self.api_key:
            raise ValueError("请设置MOONSHOT_API_KEY或KIMI_API_KEY环境变量或传入api_key参数")
        self.base_url = "https://api.moonshot.cn/v1"
    
    def _call_api(self, image_path: str, prompt: str) -> str:
        """调用Kimi多模态API"""
        import openai
        
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 准备图像
        temp_path = self._prepare_image(image_path)
        try:
            with open(temp_path, 'rb') as f:
                image_data = f.read()
            
            # 获取图片扩展名（用于base64格式）
            image_ext = os.path.splitext(temp_path)[1].lstrip('.') or 'jpeg'
            
            # 使用base64编码图片，格式：data:image/{ext};base64,{encoded_data}
            image_url = f"data:image/{image_ext};base64,{base64.b64encode(image_data).decode('utf-8')}"
            
            response = client.chat.completions.create(
                model="moonshot-v1-32k-vision-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "你是 Kimi。"
                    },
                    {
                        "role": "user",
                        # content是一个list，包含image_url和text两个部分
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=50,
                top_p=0.9
            )
            
            caption = response.choices[0].message.content.strip()
            return caption
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class DoubaoVLClient(LMMAPIClient):
    """豆包-多模态 (Doubao-VL) API客户端"""
    
    def _get_api_key_env_name(self) -> str:
        # 支持两种环境变量名：ARK_API_KEY 和 DOUBAO_API_KEY
        return "ARK_API_KEY"
    
    def __init__(self, api_key: Optional[str] = None):
        # 优先使用传入的api_key，否则尝试从环境变量读取
        if api_key is None:
            # 先尝试 ARK_API_KEY，再尝试 DOUBAO_API_KEY（向后兼容）
            api_key = os.getenv("ARK_API_KEY") or os.getenv("DOUBAO_API_KEY")
        super().__init__(api_key)
        if not self.api_key:
            raise ValueError("请设置ARK_API_KEY或DOUBAO_API_KEY环境变量或传入api_key参数")
        # 豆包API URL
        self.base_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    
    def _call_api(self, image_path: str, prompt: str) -> str:
        """调用字节跳动Doubao-VL API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 准备图像
        temp_path = self._prepare_image(image_path)
        try:
            with open(temp_path, 'rb') as f:
                image_data = f.read()
            
            # 获取图片扩展名（用于base64格式）
            image_ext = os.path.splitext(temp_path)[1].lstrip('.') or 'jpeg'
            
            # 使用base64编码图片，格式：data:image/{ext};base64,{encoded_data}
            image_url = f"data:image/{image_ext};base64,{base64.b64encode(image_data).decode('utf-8')}"
            
            payload = {
                "model": "doubao-seed-1-6-251015",  # 使用最新的vision模型
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 50,
                "top_p": 0.9,
                # 明确禁用深度思考功能
                "thinking": {
                    "type": "disabled"  # 不使用深度思考能力
                }
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            # 提取响应内容
            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0].get('message', {})
                if isinstance(message.get('content'), str):
                    caption = message['content'].strip()
                elif isinstance(message.get('content'), list):
                    # 如果content是list，提取text部分
                    text_parts = [item.get('text', '') for item in message.get('content', []) if item.get('type') == 'text']
                    caption = ' '.join(text_parts).strip()
                else:
                    caption = str(message.get('content', '')).strip()
                return caption
            else:
                raise Exception(f"API响应格式错误: {result}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def create_lmm_client(lmm_type: str, api_key: Optional[str] = None) -> LMMAPIClient:
    """
    创建LMM客户端实例
    
    Args:
        lmm_type: LMM类型，可选: "gpt", "qwen", "kimi", "doubao"
        api_key: API密钥（可选，会从环境变量读取）
        
    Returns:
        LMMAPIClient实例
    """
    lmm_type = lmm_type.lower()
    if lmm_type == "gpt" or lmm_type == "gpt4v":
        return GPT4VClient(api_key)
    elif lmm_type == "qwen" or lmm_type == "qwen-vl":
        return QwenVLClient(api_key)
    elif lmm_type == "kimi" or lmm_type == "kimi-vl":
        return KimiVLClient(api_key)
    elif lmm_type == "doubao" or lmm_type == "doubao-vl":
        return DoubaoVLClient(api_key)
    else:
        raise ValueError(f"不支持的LMM类型: {lmm_type}，支持的类型: gpt, qwen, kimi, doubao")

