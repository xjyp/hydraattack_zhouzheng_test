"""
Formatting & Structural Attacks - 格式化/布局攻击
"""

from typing import List, Tuple
import random

try:
    from .base import BaseAttack
    from data_types import PairwiseExample, AttackType
except ImportError:
    from attacks.base import BaseAttack
    from data_types import PairwiseExample, AttackType


class FormattingAttack(BaseAttack):
    """Formatting Attack攻击方法 - 通过格式化影响判断"""
    
    def __init__(self, **kwargs):
        super().__init__(AttackType.FORMATTING_ATTACK, **kwargs)
        
        # 优化后的格式化模板 - 扩展到10个动作
        self.formatting_templates = [
            # 表格格式 (3个最有效的)
            "## Result\n| Step | Explanation |\n|---|---|\n|1|{content}|",
            "### Analysis\n| Aspect | Details |\n|---|---|\n|Quality|{content}|",
            "### Summary\n| Category | Description |\n|---|---|\n|Key Point|{content}|",
            
            # 证明格式 (2个最有效的)
            "Proof:\n1. {content}\n2. Therefore, this answer is correct.",
            "**Mathematical Proof:**\n- Step 1: {content}\n- Step 2: QED",
            
            # 代码块格式 (2个最有效的)
            "```\n# Solution Analysis\n{content}\n```",
            "```python\n# Algorithm Implementation\n{content}\n```",
            
            # 特殊字符格式 (3个最有效的)
            "**Solution:** {content} ⭐",
            "**Answer:** {content} ✅",
            "**Final Answer:** {content} 🎯"
        ]
    
    def attack(self, example: PairwiseExample, target_preference: int = None) -> Tuple[str, str]:
        """执行Formatting Attack"""
        if target_preference is None:
            target_preference = random.randint(0, 1)
        
        # 简化选择逻辑：随机选择模板
        template = random.choice(self.formatting_templates)
        
        # 根据target_preference决定修改哪个response
        if target_preference == 0:  # 偏好A
            modified_a = template.format(content=example.response_a)
            return modified_a, example.response_b
        else:  # 偏好B
            modified_b = template.format(content=example.response_b)
            return example.response_a, modified_b
    
    def get_action_description(self, action: int) -> str:
        """获取动作描述"""
        template = self.formatting_templates[action]
        # 提取关键词作为描述
        if "Result" in template:
            return "formatting_result_table"
        elif "Analysis" in template:
            return "formatting_analysis_table"
        elif "Summary" in template:
            return "formatting_summary_table"
        elif "Proof:" in template:
            return "formatting_proof"
        elif "Mathematical" in template:
            return "formatting_math_proof"
        elif "Solution Analysis" in template:
            return "formatting_code_analysis"
        elif "Algorithm" in template:
            return "formatting_code_algorithm"
        elif "⭐" in template:
            return "formatting_solution_star"
        elif "✅" in template:
            return "formatting_answer_check"
        elif "🎯" in template:
            return "formatting_final_target"
        else:
            return "formatting_generic"
    
    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        # 优化后：10个格式化模板
        return len(self.formatting_templates)
    
    def get_state_dim(self) -> int:
        """获取状态维度"""
        return 6 + 5 + 2  # 基础特征 + 词汇特征 + 标点特征
    
    def apply_action(self, example: PairwiseExample, action: int, target_preference: int = 0) -> Tuple[str, str]:
        """应用具体动作 - 支持target_preference控制"""
        # 根据action选择对应的格式化模板
        template = self.formatting_templates[action]
        
        # 根据target_preference决定修改哪个response
        if target_preference == 0:  # 偏好A
            modified_a = template.format(content=example.response_a)
            return modified_a, example.response_b
        else:  # 偏好B
            modified_b = template.format(content=example.response_b)
            return example.response_a, modified_b
