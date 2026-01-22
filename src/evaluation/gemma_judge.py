"""
Gemma-3-4B-IT Judge实现
"""

import torch
import random
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from typing import Dict, Any, Optional

try:
    from data_types import PairwiseExample, JudgeResponse, JudgeType
    from .judge import LocalJudge
except ImportError:
    from data_types import PairwiseExample, JudgeResponse, JudgeType
    from evaluation.judge import LocalJudge


class GemmaJudge(LocalJudge):
    """Gemma-3-4B-IT作为Judge的实现"""
    
    def __init__(self, model_path: str = "google/gemma-3-4b-it", device: str = "cuda"):
        super().__init__(model_path, device)
    
    def _load_model(self):
        """加载Gemma-3-4B-IT模型"""
        print(f"Loading Gemma-3-4B-IT model from {self.model_path}...")
        
        try:
            # Gemma uses AutoProcessor instead of AutoTokenizer
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # For compatibility with LocalJudge, assign processor to tokenizer
            # This allows the base class methods to work with processor
            self.tokenizer = self.processor
            
            # Resolve target device index (support strings like "cuda:1")
            device_index = 0
            if isinstance(self.device, str):
                if self.device.startswith("cuda") and ":" in self.device:
                    try:
                        device_index = int(self.device.split(":")[1])
                    except ValueError:
                        device_index = 0

            # Gemma uses Gemma3ForConditionalGeneration instead of AutoModelForCausalLM
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": device_index},
                trust_remote_code=True
            ).eval()
            
            print("✅ Gemma-3-4B-IT model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def judge_pairwise(self, example: PairwiseExample, modified_instruction: str = None) -> JudgeResponse:
        """使用Gemma模型进行pairwise判断，重写以适配processor的调用方式"""
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
            
            # Format messages for Gemma's chat template
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            
            # Use processor.apply_chat_template with tokenize=True
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)
            
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
            raw_output = self.processor.decode(generation, skip_special_tokens=True).strip()
            
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


def create_gemma_judge(model_path: str = "google/gemma-3-4b-it") -> GemmaJudge:
    """创建Gemma-3-4B-IT Judge实例"""
    return GemmaJudge(model_path)

