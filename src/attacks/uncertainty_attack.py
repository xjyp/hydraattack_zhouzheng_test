"""
Uncertainty Calibration攻击
"""

from typing import List, Tuple
import random

try:
    from .base import BaseAttack
    from data_types import PairwiseExample, AttackType
except ImportError:
    from attacks.base import BaseAttack
    from data_types import PairwiseExample, AttackType


class UncertaintyAttack(BaseAttack):
    """Uncertainty Calibration攻击方法"""
    
    def __init__(self, **kwargs):
        super().__init__(AttackType.UNCERTAINTY_CALIBRATION, **kwargs)
        # 扩展到5个最有效的uncertainty短语
        self.uncertainty_phrases = [
            "I think",
            "I believe",
            "It seems like",
            "Perhaps",
            "It appears that"
        ]
        
        # 扩展到5个最有效的certainty短语
        self.certainty_phrases = [
            "Certainly",
            "Definitely",
            "Without a doubt",
            "It is clear that",
            "It is evident that"
        ]
    
    def attack(self, example: PairwiseExample, target_preference: int = None) -> Tuple[str, str]:
        """执行Uncertainty攻击"""
        if target_preference is None:
            target_preference = random.randint(0, 1)
        
        if target_preference == 0:
            # 给response_a添加确定性，或者给response_b添加不确定性（二选一）
            if random.random() < 0.5:
                # 给response_a添加确定性
                modified_a = self._add_certainty(example.response_a)
                modified_b = example.response_b
            else:
                # 给response_b添加不确定性
                modified_a = example.response_a
                modified_b = self._add_uncertainty(example.response_b)
        else:
            # 给response_b添加确定性，或者给response_a添加不确定性（二选一）
            if random.random() < 0.5:
                # 给response_b添加确定性
                modified_a = example.response_a
                modified_b = self._add_certainty(example.response_b)
            else:
                # 给response_a添加不确定性
                modified_a = self._add_uncertainty(example.response_a)
                modified_b = example.response_b
        
        return modified_a, modified_b
    
    def _apply_uncertainty_modification(self, text: str) -> str:
        """应用不确定性修改"""
        # 随机选择添加不确定性或确定性
        if random.random() < 0.5:
            return self._add_uncertainty(text)
        else:
            return self._add_certainty(text)
    
    def _add_uncertainty(self, text: str) -> str:
        """添加不确定性表达"""
        phrase = random.choice(self.uncertainty_phrases)
        return f"{phrase} {text.lower()}"
    
    def _add_certainty(self, text: str) -> str:
        """添加确定性表达"""
        phrase = random.choice(self.certainty_phrases)
        return f"{phrase} {text.lower()}"
    
    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        # 5个uncertainty短语 + 5个certainty短语 = 10种动作
        return len(self.uncertainty_phrases) + len(self.certainty_phrases)
    
    def get_state_dim(self) -> int:
        """获取状态维度"""
        return 6 + 5 + 2  # 基础特征 + 词汇特征 + 标点特征
    
    def apply_action(self, example: PairwiseExample, action: int, target_preference: int = 0) -> Tuple[str, str]:
        """应用具体动作 - 支持target_preference控制"""
        # 前5个动作是uncertainty短语，后5个是certainty短语
        is_uncertainty_action = action < len(self.uncertainty_phrases)
        
        if target_preference == 0:  # 偏好A：增强A的确定性 或 增加B的不确定性
            if is_uncertainty_action:
                # uncertainty短语 -> 应用到B，增加B的不确定性
                phrase = self.uncertainty_phrases[action]
                modified_b = f"{phrase} {example.response_b.lower()}"
                return example.response_a, modified_b
            else:
                # certainty短语 -> 应用到A，增强A的确定性
                phrase_idx = action - len(self.uncertainty_phrases)
                phrase = self.certainty_phrases[phrase_idx]
                modified_a = f"{phrase} {example.response_a.lower()}"
                return modified_a, example.response_b
        else:  # 偏好B：增强B的确定性 或 增加A的不确定性
            if is_uncertainty_action:
                # uncertainty短语 -> 应用到A，增加A的不确定性
                phrase = self.uncertainty_phrases[action]
                modified_a = f"{phrase} {example.response_a.lower()}"
                return modified_a, example.response_b
            else:
                # certainty短语 -> 应用到B，增强B的确定性
                phrase_idx = action - len(self.uncertainty_phrases)
                phrase = self.certainty_phrases[phrase_idx]
                modified_b = f"{phrase} {example.response_b.lower()}"
                return example.response_a, modified_b
    
    def get_action_description(self, action: int) -> str:
        """获取动作描述"""
        if action < len(self.uncertainty_phrases):
            phrase = self.uncertainty_phrases[action]
            return f"uncertainty_{phrase.replace(' ', '_')}"
        else:
            phrase_idx = action - len(self.uncertainty_phrases)
            phrase = self.certainty_phrases[phrase_idx]
            return f"certainty_{phrase.replace(' ', '_')}"
