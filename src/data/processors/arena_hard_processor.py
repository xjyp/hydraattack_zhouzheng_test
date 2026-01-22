"""
Arena-Hard-Auto数据处理器
"""

import json
from typing import List, Dict, Any
from pathlib import Path

try:
    from ..base_processor import BaseProcessor
    from ...data_types import PairwiseExample, BenchmarkType
except ImportError:
    from data.base_processor import BaseProcessor
    from data_types import PairwiseExample, BenchmarkType


class ArenaHardProcessor(BaseProcessor):
    """Arena-Hard-Auto数据处理器"""
    
    def __init__(self, data_dir: str = "./raw_data", output_dir: str = "./data/processed"):
        super().__init__(data_dir, output_dir)
        self.benchmark_type = BenchmarkType.ARENA_HARD
    
    def load_data(self) -> List[Dict[str, Any]]:
        """加载Arena-Hard数据"""
        try:
            # 加载问题数据
            split = "arena-hard-v2.0"  # 默认使用这个split
            question_file = f"raw_data/arena-hard-auto/data/{split}/question.jsonl"
            questions = []
            
            with open(question_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
            
            # 加载模型回答数据
            model_answer_dir = f"raw_data/arena-hard-auto/data/{split}/model_answer"
            model_answers = {}
            
            if Path(model_answer_dir).exists():
                for answer_file in Path(model_answer_dir).glob("*.jsonl"):
                    model_name = answer_file.stem
                    model_answers[model_name] = []
                    
                    with open(answer_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                model_answers[model_name].append(json.loads(line))
            
            # 组合问题和回答
            combined_data = []
            for question in questions:
                uid = question["uid"]
                combined_item = {
                    "uid": uid,
                    "question": question,
                    "answers": {}
                }
                
                # 添加所有模型的回答
                for model_name, answers in model_answers.items():
                    for answer in answers:
                        if answer["uid"] == uid:
                            combined_item["answers"][model_name] = answer
                            break
                
                combined_data.append(combined_item)
            
            return combined_data
            
        except Exception as e:
            print(f"❌ 无法加载Arena-Hard数据: {e}")
            return []
    
    def convert_to_pairwise(self, data: List[Dict[str, Any]]) -> List[PairwiseExample]:
        """将Arena-Hard数据转换为PairwiseExample格式"""
        examples = []
        
        for item in data:
            try:
                uid = item["uid"]
                question = item["question"]
                answers = item["answers"]
                
                # 获取问题内容
                instruction = question.get("prompt", "")
                category = question.get("category", "general")
                subcategory = question.get("subcategory", "general")
                
                # 选择两个不同的模型回答进行比较
                model_names = list(answers.keys())
                if len(model_names) < 2:
                    continue
                
                # 选择前两个模型
                model_a_name = model_names[0]
                model_b_name = model_names[1]
                
                answer_a = answers[model_a_name]
                answer_b = answers[model_b_name]
                
                # 提取回答内容
                response_a = self._extract_response_content(answer_a)
                response_b = self._extract_response_content(answer_b)
                
                example = PairwiseExample(
                    question_id=f"arena_hard_{uid}",
                    instruction=instruction,
                    response_a=response_a,
                    response_b=response_b,
                    model_a=model_a_name,
                    model_b=model_b_name,
                    metadata={
                        "category": category,
                        "subcategory": subcategory,
                        "uid": uid
                    }
                )
                examples.append(example)
                
            except Exception as e:
                print(f"❌ 处理Arena-Hard数据项时出错: {e}")
                continue
        
        return examples
    
    def get_benchmark_type(self) -> BenchmarkType:
        """获取基准测试类型"""
        return self.benchmark_type
    
    def _extract_response_content(self, answer: Dict[str, Any]) -> str:
        """从回答中提取内容"""
        try:
            # 尝试从messages中提取内容
            if "messages" in answer:
                messages = answer["messages"]
                for message in reversed(messages):  # 从后往前找assistant的回答
                    if message.get("role") == "assistant":
                        content = message.get("content", "")
                        if isinstance(content, dict):
                            # 如果content是字典，尝试提取text或answer字段
                            return content.get("answer", content.get("text", str(content)))
                        elif isinstance(content, str):
                            return content
            
            # 如果没有找到messages，尝试其他字段
            if "answer" in answer:
                return answer["answer"]
            elif "content" in answer:
                return answer["content"]
            elif "text" in answer:
                return answer["text"]
            else:
                return str(answer)
                
        except Exception as e:
            print(f"❌ 提取回答内容时出错: {e}")
            return str(answer)
