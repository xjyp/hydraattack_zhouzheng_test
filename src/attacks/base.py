"""
基础攻击类
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import random

try:
    from data_types import PairwiseExample, AttackType
except ImportError:
    from data_types import PairwiseExample, AttackType


class BaseAttack(ABC):
    """基础攻击类"""
    
    def __init__(self, attack_type: AttackType, **kwargs):
        self.attack_type = attack_type
        self.config = kwargs
    
    @abstractmethod
    def attack(self, example: PairwiseExample, target_preference: int = None) -> Tuple[str, str]:
        """
        执行攻击
        
        Args:
            example: 要攻击的pairwise样本
            target_preference: 目标偏好 (0 for A, 1 for B, None for random)
        
        Returns:
            (modified_response_a, modified_response_b): 修改后的回答
        """
        pass
    
    @abstractmethod
    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        pass
    
    @abstractmethod
    def get_state_dim(self) -> int:
        """获取状态维度"""
        pass
    
    def encode_state(self, example: PairwiseExample, query_count: int = 0) -> List[float]:
        """
        将状态编码为向量
        
        Args:
            example: pairwise样本
            query_count: 当前查询次数
        
        Returns:
            状态向量
        """
        # 基础特征：文本长度、查询次数等
        features = [
            len(example.instruction),
            len(example.response_a),
            len(example.response_b),
            query_count,
            float(len(example.response_a) > len(example.response_b)),
            float(len(example.response_a) == len(example.response_b))
        ]
        
        # 添加文本特征
        features.extend(self._extract_text_features(example))
        
        return features
    
    def _extract_text_features(self, example: PairwiseExample) -> List[float]:
        """提取文本特征"""
        # 简单的文本特征提取
        features = []
        
        # 词汇特征
        words_a = example.response_a.split()
        words_b = example.response_b.split()
        
        features.extend([
            len(words_a),
            len(words_b),
            len(set(words_a)),
            len(set(words_b)),
            len(set(words_a) & set(words_b)) / max(len(set(words_a) | set(words_b)), 1)
        ])
        
        # 标点符号特征
        punct_a = sum(1 for c in example.response_a if c in '.,!?;:')
        punct_b = sum(1 for c in example.response_b if c in '.,!?;:')
        
        features.extend([punct_a, punct_b])
        
        return features
    
    def get_available_actions(self, example: PairwiseExample) -> List[int]:
        """获取可用的动作列表"""
        return list(range(self.get_action_space_size()))
    
    def get_action_description(self, action: int) -> str:
        """获取动作描述"""
        return f"action_{action}"
    
    def apply_action(self, example: PairwiseExample, action: int, target_preference: int = 0) -> Tuple[str, str]:
        """
        应用动作到样本 - 支持target_preference控制
        
        Args:
            example: 原始样本
            action: 动作ID
            target_preference: 目标偏好 (0 for A, 1 for B)
        
        Returns:
            (modified_response_a, modified_response_b): 修改后的回答
        """
        return self.attack(example, target_preference=target_preference)
    
    def calculate_reward(self, 
                        original_preference: int, 
                        new_preference: int, 
                        query_count: int,
                        success_bonus: float = 10.0,
                        query_penalty: float = 0.1) -> float:
        """
        计算奖励
        
        Args:
            original_preference: 原始偏好
            new_preference: 新偏好
            query_count: 查询次数
            success_bonus: 成功奖励
            query_penalty: 查询惩罚
        
        Returns:
            奖励值
        """
        if original_preference != new_preference:
            # 成功翻转偏好
            return success_bonus - query_count * query_penalty
        else:
            # 未成功翻转
            return -query_count * query_penalty
    
    def is_attack_successful(self, original_preference: int, new_preference: int) -> bool:
        """
        判断攻击是否成功
        
        对于大多数攻击方法，成功条件是偏好改变（new_preference != original_preference）
        但对于 PositionAttack，由于交换了位置，成功条件是偏好不变（new_preference == original_preference）
        
        Args:
            original_preference: 原始偏好 (0 for A, 1 for B)
            new_preference: 攻击后的偏好 (0 for A, 1 for B)
        
        Returns:
            True if attack is successful, False otherwise
        """
        # 默认行为：成功意味着偏好改变
        return original_preference != new_preference

