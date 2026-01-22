"""
GLM Judge实现
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional

try:
    from data_types import PairwiseExample, JudgeResponse, JudgeType
    from .judge import LocalJudge
except ImportError:
    from data_types import PairwiseExample, JudgeResponse, JudgeType
    from evaluation.judge import LocalJudge


class GLMJudge(LocalJudge):
    """GLM作为Judge的实现"""
    
    def __init__(self, model_path: str = "/share/disk/llm_cache/glm-4-9b-chat-hf", device: str = "cuda"):
        super().__init__(model_path, device)
    
    def _load_model(self):
        """加载GLM模型"""
        print(f"Loading GLM model from {self.model_path}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map={"": 0},  # 强制使用第一个GPU (CUDA_VISIBLE_DEVICES指定的卡)
                trust_remote_code=True
            )
            
            print("✅ GLM model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise


def create_glm_judge(model_path: str = "/share/disk/llm_cache/glm-4-9b-chat-hf") -> GLMJudge:
    """创建GLM Judge实例"""
    return GLMJudge(model_path)

