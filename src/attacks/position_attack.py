"""
Position Attack - 通过翻转两个candidate的顺序进行攻击
"""

from typing import List, Tuple
import random

try:
    from .base import BaseAttack
    from data_types import PairwiseExample, AttackType
except ImportError:
    from attacks.base import BaseAttack
    from data_types import PairwiseExample, AttackType


class PositionAttack(BaseAttack):
    """Position Attack攻击方法 - 翻转候选回答的顺序"""
    
    def __init__(self, **kwargs):
        super().__init__(AttackType.POSITION_MANIPULATION, **kwargs)
    
    def attack(self, example: PairwiseExample, target_preference: int = None) -> Tuple[str, str]:
        """执行Position Attack"""
        # 直接翻转两个回答的顺序
        return example.response_b, example.response_a
    
    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        # 只有一种动作：翻转顺序
        return 1
    
    def get_state_dim(self) -> int:
        """获取状态维度"""
        return 6 + 5 + 2  # 基础特征 + 词汇特征 + 标点特征
    
    def apply_action(self, example: PairwiseExample, action: int, target_preference: int = 0) -> Tuple[str, str]:
        """应用具体动作 - 支持target_preference控制"""
        # 只有action 0：翻转顺序，其他action无效
        if action == 0:
            # PositionAttack总是翻转顺序，target_preference不影响结果
            return example.response_b, example.response_a
        else:
            # 对于无效的action，返回原始顺序（但这种情况不应该发生）
            return example.response_a, example.response_b
    
    def get_action_description(self, action: int) -> str:
        """获取动作描述"""
        if action == 0:
            return "flip_position"
        else:
            return "no_change"
    
    def is_attack_successful(self, original_preference: int, new_preference: int) -> bool:
        """
        判断Position Attack是否成功
        
        对于Position Attack，由于交换了response的位置：
        - 如果原始偏好是A（0），目标是选择B（原来的response_b）
        - 交换后，原来的response_b在位置A（0）
        - 如果judge现在偏好A（0），意味着选择了原来的response_b，这是成功的
        - 因此成功条件是：new_preference == original_preference
        
        Args:
            original_preference: 原始偏好 (0 for A, 1 for B)
            new_preference: 攻击后的偏好 (0 for A, 1 for B)
        
        Returns:
            True if attack is successful, False otherwise
        """
        # PositionAttack交换了位置，所以成功条件是偏好不变
        # 因为交换后，原来的response_b在位置A，原来的response_a在位置B
        # 如果原始偏好是A（0），目标是B，交换后如果judge偏好A（0），意味着选择了原来的B，成功
        return original_preference == new_preference