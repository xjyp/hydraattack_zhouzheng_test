"""
Gemma-3-1B-IT Judge实现
"""

import torch
import random
from transformers import AutoTokenizer, Gemma3ForCausalLM
from typing import Dict, Any, Optional

try:
    from data_types import PairwiseExample, JudgeResponse, JudgeType
    from .judge import LocalJudge
except ImportError:
    from data_types import PairwiseExample, JudgeResponse, JudgeType
    from evaluation.judge import LocalJudge


class GemmaJudge(LocalJudge):
    """Gemma-3-1B-IT作为Judge的实现（纯文本模型）"""
    
    def __init__(self, model_path: str = "google/gemma-3-1b-it", device: str = "cuda"):
        super().__init__(model_path, device)
    
    def _load_model(self):
        """加载Gemma-3-1B-IT模型（纯文本版本）"""
        print(f"Loading Gemma-3-1B-IT model from {self.model_path}...")
        
        try:
            # Gemma-3-1B-IT is text-only, uses AutoTokenizer (not AutoProcessor)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # For compatibility with LocalJudge, assign tokenizer to processor
            self.processor = self.tokenizer
            
            # Resolve target device index (support strings like "cuda:1")
            device_index = 0
            if isinstance(self.device, str):
                if self.device.startswith("cuda") and ":" in self.device:
                    try:
                        device_index = int(self.device.split(":")[1])
                    except ValueError:
                        device_index = 0

            # Gemma-3-1B-IT uses Gemma3ForCausalLM (text-only model)
            self.model = Gemma3ForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": device_index},
                trust_remote_code=True
            ).eval()
            
            print("✅ Gemma-3-1B-IT model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def judge_pairwise(self, example: PairwiseExample, modified_instruction: str = None) -> JudgeResponse:
        """使用Gemma模型进行pairwise判断，重写以适配tokenizer的调用方式"""
        try:
            # If modified_instruction is provided, use it; otherwise use original
            instruction = modified_instruction if modified_instruction else example.instruction
            
            # Create temporary example for judging
            temp_example = PairwiseExample(
                question_id=example.question_id,
                instruction=instruction,
                response_a=example.response_a,
                response_b=example.response_b,
                model_a=example.model_a,
                model_b=example.model_b
            )
            
            prompt = self.get_judge_prompt(temp_example)
            
            # Format messages for Gemma-3-1B-IT chat template (nested list format)
            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
            ]
            
            # Use tokenizer.apply_chat_template with tokenize=True
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Move to device - input_ids should remain as long (int64), not converted to bfloat16
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate response
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
                generation = generation[0][input_len:]
            
            # Decode only the generated part
            raw_output = self.tokenizer.decode(generation, skip_special_tokens=True).strip()
            
            # Parse the response
            preference, confidence = self._parse_response(raw_output)
            
            return JudgeResponse(
                preference=preference,
                confidence=confidence,
                raw_response=raw_output
            )
            
        except Exception as e:
            print(f"Error in GemmaJudge: {e}")
            # Return random result as fallback
            return JudgeResponse(
                preference=random.randint(0, 1),
                confidence=0.1,
                raw_response=f"Error: {str(e)}"
            )


def create_gemma_judge(model_path: str = "google/gemma-3-1b-it") -> GemmaJudge:
    """创建Gemma-3-1B-IT Judge实例"""
    return GemmaJudge(model_path)

