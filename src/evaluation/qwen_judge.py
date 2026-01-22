"""
Qwen3-8B Judge实现
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


class QwenJudge(LocalJudge):
    """Qwen3-8B作为Judge的实现"""
    
    def __init__(self, model_path: str = "/share/disk/llm_cache/Qwen3-8B", device: str = "cuda"):
        super().__init__(model_path, device)
    
    def _load_model(self):
        """加载Qwen3-8B模型"""
        print(f"Loading Qwen3-8B model from {self.model_path}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map={"": 0},  # 强制使用第一个GPU (CUDA_VISIBLE_DEVICES指定的卡)
                trust_remote_code=True
            )
            
            print("✅ Qwen3-8B model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

def create_qwen_judge(model_path: str = "/share/disk/llm_cache/Qwen3-8B") -> QwenJudge:
    """创建Qwen3-8B Judge实例"""
    return QwenJudge(model_path)
