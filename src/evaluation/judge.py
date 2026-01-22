"""
Judge接口实现
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import json
import time
import re
import random
import os
import requests
from requests import RequestException
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass

from data_types import PairwiseExample, JudgeResponse


@dataclass
class JudgeConfig:
    """Judge配置"""
    api_key: str
    base_url: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 30


class BaseJudge(ABC):
    """基础Judge类"""
    
    def __init__(self, config: JudgeConfig):
        self.config = config
    
    @abstractmethod
    def judge_pairwise(self, example: PairwiseExample) -> JudgeResponse:
        """进行pairwise判断"""
        pass
    
    @abstractmethod
    def get_judge_prompt(self, example: PairwiseExample) -> str:
        """获取judge prompt"""
        pass


class OnlineJudge(BaseJudge):
    """在线模型Judge基类，提供统一的模型加载和判断逻辑"""
    
    _DEFAULT_BASE_URL = "https://www.dmxapi.com/v1"
    
    def __init__(self, config: JudgeConfig):
        super().__init__(config)
        if not self.config.api_key:
            raise ValueError("API key is required for OnlineJudge")
        self.session = requests.Session()
        base_url = config.base_url or self._DEFAULT_BASE_URL
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
    
    def get_judge_prompt(self, example: PairwiseExample) -> str:
        return get_standard_judge_prompt(example)
    
    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Construct chat messages payload."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    
    def judge_pairwise(self, example: PairwiseExample, modified_instruction: str = None) -> JudgeResponse:
        """使用在线模型进行pairwise判断"""
        instruction = modified_instruction if modified_instruction else example.instruction
        temp_example = PairwiseExample(
            question_id=example.question_id,
            instruction=instruction,
            response_a=example.response_a,
            response_b=example.response_b,
            model_a=example.model_a,
            model_b=example.model_b,
            metadata=example.metadata,
        )
        
        prompt = self.get_judge_prompt(temp_example)
        messages = self._build_messages(prompt)
        
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        last_error = None
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                response_json = response.json()
                content = response_json["choices"][0]["message"]["content"].strip()
                preference, confidence = parse_standard_judge_response(content)
                return JudgeResponse(
                    preference=preference,
                    confidence=confidence,
                    reasoning=content,
                    raw_response=content,
                )
            except (RequestException, KeyError, ValueError) as e:
                last_error = e
                print(f"Error in OnlineJudge (attempt {attempt}/{max_attempts}): {e}")
                if attempt < max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
        
        return JudgeResponse(
            preference=random.randint(0, 1),
            confidence=0.1,
            raw_response=f"Error after {max_attempts} attempts: {last_error}",
        )


class LocalJudge(BaseJudge):
    """本地模型Judge基类，提供统一的模型加载和判断逻辑"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        # 创建一个虚拟的JudgeConfig来满足BaseJudge的要求
        config = JudgeConfig(api_key="dummy")  # LocalJudge不需要API key
        super().__init__(config)
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载本地模型 - 子类需要重写此方法"""
        raise NotImplementedError("Subclasses must implement _load_model")
    
    def get_judge_prompt(self, example: PairwiseExample) -> str:
        return get_standard_judge_prompt(example)
    
    def judge_pairwise(self, example: PairwiseExample, modified_instruction: str = None) -> JudgeResponse:
        """使用本地模型进行pairwise判断"""
        # 如果提供了修改后的instruction，使用它；否则使用原始的
        instruction = modified_instruction if modified_instruction else example.instruction
        
        # 创建临时的example用于judge
        temp_example = PairwiseExample(
            question_id=example.question_id,
            instruction=instruction,
            response_a=example.response_a,
            response_b=example.response_b,
            model_a=example.model_a,
            model_b=example.model_b
        )
        
        prompt = self.get_judge_prompt(temp_example)
        
        # 编码输入：如果tokenizer有chat_template，使用apply_chat_template格式化
        # 这对于chat模型（如GLM-4, Llama-3.1-Instruct, Qwen等）是必要的
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            # 使用chat template格式化输入
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        else:
            # 对于没有chat_template的模型，直接使用原始prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        input_length = inputs.input_ids.shape[1]  # 输入token的数量
        
        # 生成和解析部分，带重试机制
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # 生成回答
                # 第一次尝试使用确定性生成（do_sample=False）
                # 后续尝试使用带少量随机性的生成（do_sample=True, temperature=0.1）
                with torch.no_grad():
                    if attempt == 0:
                        # 第一次尝试：确定性生成
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=50,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    else:
                        # 后续尝试：带少量随机性的生成
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=50,
                            do_sample=True,
                            temperature=0.1,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                
                # 只解码新生成的部分（排除输入的prompt）
                generated_ids = outputs[0][input_length:]
                raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                # 提取判断结果（直接解析生成的内容）
                preference, confidence = self._parse_response(raw_output)
                
                # 如果解析成功（置信度 > 0.1），返回结果
                if confidence > 0.1:
                    return JudgeResponse(
                        preference=preference,
                        confidence=confidence,
                        raw_response=raw_output  # 只存储生成的内容，不包含prompt
                    )
                
                # 如果置信度太低，继续重试（除非是最后一次尝试）
                if attempt < max_attempts - 1:
                    continue
                else:
                    # 最后一次尝试，即使置信度低也返回结果
                    return JudgeResponse(
                        preference=preference,
                        confidence=confidence,
                        raw_response=raw_output
                    )
                    
            except Exception as e:
                # 如果生成或解析过程中出现异常，继续重试
                if attempt < max_attempts - 1:
                    print(f"Error in LocalJudge (attempt {attempt + 1}/{max_attempts}): {e}")
                    continue
                else:
                    # 最后一次尝试也失败，返回随机结果作为fallback
                    print(f"Error in LocalJudge after {max_attempts} attempts: {e}")
                    return JudgeResponse(
                        preference=random.randint(0, 1),
                        confidence=0.5,
                        raw_response=f"Error: {str(e)}"
                    )
    
    def _parse_response(self, generated_text: str) -> Tuple[int, float]:
        return parse_standard_judge_response(generated_text)

def get_standard_judge_prompt(example: PairwiseExample) -> str:
        """获取统一的judge prompt，来自ICLR25 paper judgebench的vanilla prompt template"""
        return f"""You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to select the best output for the given instruction.
Select the Output (a) or Output (b) that is better for the given instruction. The two outputs are generated by two different AI chatbots respectively.
Do NOT provide any explanation for your choice.
Do NOT say both / neither are good.
You should answer using ONLY "Output (a)" or "Output (b)". Do NOT output any other words.
# Instruction:
{example.instruction}
# Output (a):
{example.response_a}
# Output (b):
{example.response_b}
# Which is better, Output (a) or Output (b)? Your response should be either "Output (a)" or "Output (b)":"""


def parse_standard_judge_response(generated_text: str) -> Tuple[int, float]:
        """解析本地模型响应：根据judge prompt的期望输出格式进行简单匹配
        
        参数:
            generated_text: 模型生成的纯文本内容（不包含prompt）
        
        返回:
            (preference, confidence): preference为0表示选择Output (a)，1表示选择Output (b)
        
        匹配规则：
        1. 完美匹配"Output (a)"或"Output (b)" -> 置信度0.9
        2. 包含一个output选项（如"Output (a) is better"） -> 置信度0.5
        3. 包含两个output选项或无法找到 -> 随机选择，置信度0.1
        """
        gen_text = generated_text.strip()
        
        # 如果为空，直接返回随机结果
        if not gen_text:
            return random.randint(0, 1), 0.1
        
        # 转换为小写用于匹配
        gen_text_lower = gen_text.lower()
        
        # 检查是否包含 "Output (a)" 或 "Output (b)"
        contains_a = re.search(r'output\s*\(\s*a\s*\)', gen_text_lower)
        contains_b = re.search(r'output\s*\(\s*b\s*\)', gen_text_lower)
        
        # 如果只包含一个选项
        if contains_a and not contains_b:
            # 检查是否是完美匹配（整个文本就是 "Output (a)"，因为已经strip过了）
            perfect_pattern_a = r'^output\s*\(\s*a\s*\)$'
            if re.match(perfect_pattern_a, gen_text_lower):
                return 0, 0.9
            else:
                return 0, 0.5
        
        if contains_b and not contains_a:
            # 检查是否是完美匹配（整个文本就是 "Output (b)"，因为已经strip过了）
            perfect_pattern_b = r'^output\s*\(\s*b\s*\)$'
            if re.match(perfect_pattern_b, gen_text_lower):
                return 1, 0.9
            else:
                return 1, 0.5
        
        # 如果包含两个选项或无法找到任何选项，返回随机结果
        return random.randint(0, 1), 0.1


def create_online_judge(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: int = 30,
) -> OnlineJudge:
    """创建OnlineJudge实例，支持从环境变量Fallback."""
    resolved_api_key = api_key or os.environ.get("ONLINE_JUDGE_API_KEY") or os.environ.get("DMX_API_KEY")
    if not resolved_api_key:
        raise ValueError("API key is required to create an OnlineJudge")
    config = JudgeConfig(
        api_key=resolved_api_key,
        base_url=base_url or OnlineJudge._DEFAULT_BASE_URL,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return OnlineJudge(config)