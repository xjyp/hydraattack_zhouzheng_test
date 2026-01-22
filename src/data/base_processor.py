"""
基础数据处理器
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json
import os
from pathlib import Path

try:
    from ..data_types import PairwiseExample, BenchmarkType
except ImportError:
    from data_types import PairwiseExample, BenchmarkType


class BaseProcessor(ABC):
    """基础数据处理器抽象类"""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """加载原始数据"""
        pass
    
    @abstractmethod
    def convert_to_pairwise(self, data: List[Dict[str, Any]]) -> List[PairwiseExample]:
        """转换为pairwise格式"""
        pass
    
    def process(self) -> List[PairwiseExample]:
        """处理数据的主流程"""
        raw_data = self.load_data()
        pairwise_data = self.convert_to_pairwise(raw_data)
        self.save_processed_data(pairwise_data)
        return pairwise_data
    
    def save_processed_data(self, data: List[PairwiseExample]) -> None:
        """保存处理后的数据"""
        output_file = self.output_dir / f"{self.get_benchmark_type().value}_processed.json"
        
        # 转换为可序列化的格式
        serializable_data = []
        for example in data:
            serializable_data.append({
                "question_id": example.question_id,
                "instruction": example.instruction,
                "response_a": example.response_a,
                "response_b": example.response_b,
                "model_a": example.model_a,
                "model_b": example.model_b,
                "metadata": example.metadata or {}
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        print(f"Processed data saved to {output_file}")
        print(f"Total examples: {len(data)}")
    
    @abstractmethod
    def get_benchmark_type(self) -> BenchmarkType:
        """获取基准测试类型"""
        pass
    
    def load_processed_data(self) -> List[PairwiseExample]:
        """加载已处理的数据"""
        output_file = self.output_dir / f"{self.get_benchmark_type().value}_processed.json"
        
        if not output_file.exists():
            raise FileNotFoundError(f"Processed data not found: {output_file}")
        
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            examples.append(PairwiseExample(
                question_id=item["question_id"],
                instruction=item["instruction"],
                response_a=item["response_a"],
                response_b=item["response_b"],
                model_a=item["model_a"],
                model_b=item["model_b"],
                metadata=item.get("metadata", {})
            ))
        
        return examples
