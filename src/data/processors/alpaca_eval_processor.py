"""
AlpacaEval数据处理器
"""

import json
from pathlib import Path
from typing import List, Dict, Any

try:
    from ..base_processor import BaseProcessor
    from ...data_types import PairwiseExample, BenchmarkType
except ImportError:
    from data.base_processor import BaseProcessor
    from data_types import PairwiseExample, BenchmarkType


class AlpacaEvalProcessor(BaseProcessor):
    """AlpacaEval数据处理器"""
    
    def __init__(self, data_dir: str, output_dir: str):
        super().__init__(data_dir, output_dir)
        self.alpaca_dir = self.data_dir / "alpaca_eval"
    
    def load_data(self) -> List[Dict[str, Any]]:
        """加载AlpacaEval数据"""
        # 加载目标模型输出数据（排除参考模型）
        model_outputs = self._load_model_outputs()
        # 加载参考模型数据
        reference_outputs = self._load_reference_outputs()
        
        # 合并数据
        combined_data = []
        for model_output in model_outputs:
            # 跳过参考模型的输出，避免重复
            if model_output["generator"] == reference_outputs[0]["generator"]:
                continue
                
            # 找到对应的参考输出
            ref_output = None
            for ref in reference_outputs:
                if ref["instruction"] == model_output["instruction"]:
                    ref_output = ref
                    break
            
            if ref_output:
                combined_data.append({
                    "instruction": model_output["instruction"],
                    "model_output": model_output,
                    "reference_output": ref_output
                })
        
        return combined_data
    
    def _load_model_outputs(self) -> List[Dict[str, Any]]:
        """加载模型输出数据"""
        results_dir = self.alpaca_dir / "results"
        all_outputs = []
        
        for model_dir in results_dir.iterdir():
            if model_dir.is_dir():
                output_file = model_dir / "model_outputs.json"
                if output_file.exists():
                    with open(output_file, 'r', encoding='utf-8') as f:
                        outputs = json.load(f)
                        # 添加模型名称
                        for output in outputs:
                            output["model_name"] = model_dir.name
                        all_outputs.extend(outputs)
        
        return all_outputs
    
    def _load_reference_outputs(self) -> List[Dict[str, Any]]:
        """加载参考模型输出数据"""
        # 使用text-davinci-003作为参考模型
        ref_file = self.alpaca_dir / "results" / "text-davinci-003" / "model_outputs.json"
        
        if ref_file.exists():
            with open(ref_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 如果没有找到，尝试其他参考模型
        ref_models = ["gpt-4", "claude-3-5-sonnet", "airoboros-33b"]
        for model in ref_models:
            ref_file = self.alpaca_dir / "results" / model / "model_outputs.json"
            if ref_file.exists():
                with open(ref_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # 如果还是没有找到，使用第一个可用的模型
        results_dir = self.alpaca_dir / "results"
        for model_dir in results_dir.iterdir():
            if model_dir.is_dir():
                ref_file = model_dir / "model_outputs.json"
                if ref_file.exists():
                    with open(ref_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
        
        raise FileNotFoundError("No reference model outputs found")
    
    def convert_to_pairwise(self, data: List[Dict[str, Any]]) -> List[PairwiseExample]:
        """转换为pairwise格式"""
        examples = []
        
        for item in data:
            instruction = item["instruction"]
            model_output = item["model_output"]
            reference_output = item["reference_output"]
            
            # 随机决定哪个是A，哪个是B
            import random
            if random.random() < 0.5:
                response_a = model_output["output"]
                response_b = reference_output["output"]
                model_a = model_output["model_name"]
                model_b = reference_output["generator"]
            else:
                response_a = reference_output["output"]
                response_b = model_output["output"]
                model_a = reference_output["generator"]
                model_b = model_output["model_name"]
            
            example = PairwiseExample(
                question_id=f"alpaca_{len(examples)}",
                instruction=instruction,
                response_a=response_a,
                response_b=response_b,
                model_a=model_a,
                model_b=model_b,
                metadata={
                    "dataset": model_output.get("dataset", "helpful_base"),
                    "original_model_output": model_output,
                    "original_reference_output": reference_output
                }
            )
            examples.append(example)
        
        return examples
    
    def get_benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.ALPACA_EVAL
