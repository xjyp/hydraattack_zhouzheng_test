"""
Llama-3.1-8B-Instruct Judge实现
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


class LlamaJudge(LocalJudge):
    """Llama-3.1-8B-Instruct作为Judge的实现"""
    
    def __init__(self, model_path: str = "/share/disk/llm_cache/Llama-3.1-8B-Instruct", device: str = "cuda"):
        super().__init__(model_path, device)
    
    def _load_model(self):
        """加载Llama-3.1-8B-Instruct模型"""
        print(f"Loading Llama-3.1-8B-Instruct model from {self.model_path}...")
        
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
            
            print("✅ Llama-3.1-8B-Instruct model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise


def create_llama_judge(model_path: str = "/share/disk/llm_cache/Llama-3.1-8B-Instruct") -> LlamaJudge:
    """创建Llama-3.1-8B-Instruct Judge实例"""
    return LlamaJudge(model_path)
