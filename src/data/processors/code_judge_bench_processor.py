"""
CodeJudgeBench数据处理器
"""

import json
from typing import List, Dict, Any
from datasets import load_dataset

try:
    from ..base_processor import BaseProcessor
    from ...data_types import PairwiseExample, BenchmarkType
except ImportError:
    from data.base_processor import BaseProcessor
    from data_types import PairwiseExample, BenchmarkType


class CodeJudgeBenchProcessor(BaseProcessor):
    """CodeJudgeBench数据处理器"""
    
    def __init__(self, data_dir: str = "./raw_data", output_dir: str = "./data/processed"):
        super().__init__(data_dir, output_dir)
        self.benchmark_type = BenchmarkType.CODE_JUDGE_BENCH
    
    def load_data(self) -> List[Dict[str, Any]]:
        """加载CodeJudgeBench数据"""
        try:
            # 从本地parquet文件加载数据
            import pandas as pd
            import glob
            
            # 查找所有parquet文件
            parquet_files = glob.glob(f"{self.data_dir}/CodeJudgeBench/codejudgebench_data/codegen/*.parquet")
            
            if not parquet_files:
                print(f"❌ 在 {self.data_dir}/CodeJudgeBench/codejudgebench_data/codegen/ 中未找到parquet文件")
                return []
            
            all_data = []
            for file_path in parquet_files:
                print(f"📁 加载文件: {file_path}")
                df = pd.read_parquet(file_path)
                # 将DataFrame转换为字典列表
                data = df.to_dict('records')
                all_data.extend(data)
            
            print(f"✅ 成功加载 {len(all_data)} 条CodeJudgeBench数据")
            return all_data
            
        except Exception as e:
            print(f"❌ 无法加载CodeJudgeBench数据: {e}")
            # 返回空数据，避免程序崩溃
            return []
    
    def convert_to_pairwise(self, data: List[Dict[str, Any]]) -> List[PairwiseExample]:
        """将CodeJudgeBench数据转换为PairwiseExample格式"""
        examples = []
        
        for item in data:
            try:
                # CodeJudgeBench的数据结构
                question_content = item.get("question_content", "")
                pos_response = item.get("pos_response", "")
                neg_response = item.get("neg_response", "")
                question_id = item.get("question_id", "")
                question_title = item.get("question_title", "")
                
                # 构建完整的指令（包含标题和内容）
                full_instruction = f"Title: {question_title}\n\n{question_content}"
                
                # 创建PairwiseExample
                example = PairwiseExample(
                    question_id=question_id,
                    instruction=full_instruction,
                    response_a=pos_response,
                    response_b=neg_response,
                    model_a="positive_model",
                    model_b="negative_model",
                    metadata={
                        "platform": item.get("platform", ""),
                        "difficulty": item.get("difficulty", ""),
                        "contest_id": item.get("contest_id", ""),
                        "question_title": question_title,
                        "starter_code": item.get("starter_code", "")
                    }
                )
                examples.append(example)
                
            except Exception as e:
                print(f"❌ 处理CodeJudgeBench数据项时出错: {e}")
                continue
        
        return examples
    
    def get_benchmark_type(self) -> BenchmarkType:
        """获取基准测试类型"""
        return self.benchmark_type
