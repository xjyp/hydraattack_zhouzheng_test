"""
增强版强化学习环境
包含更多创新特性和优化
"""

import gymnasium as gym
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import random
from collections import deque
import torch
import torch.nn as nn

from data_types import PairwiseExample, AttackResult, JudgeResponse
from attacks.base import BaseAttack
from evaluation.judge import BaseJudge


class EnhancedHydraAttackEnv(gym.Env):
    """增强版Hydra Attack强化学习环境"""
    
    def __init__(self, 
                 examples: List[PairwiseExample],
                 attacks: List[BaseAttack],
                 judge: BaseJudge,
                 max_queries: int = 10,
                 success_reward: float = 10.0,
                 query_penalty: float = 0.1,
                 diversity_bonus: float = 0.5,
                 efficiency_bonus: float = 1.0,
                 confidence_threshold: float = 0.7):
        super().__init__()
        
        self.examples = examples
        self.attacks = attacks
        self.judge = judge
        self.max_queries = max_queries
        self.success_reward = success_reward
        self.query_penalty = query_penalty
        self.diversity_bonus = diversity_bonus
        self.efficiency_bonus = efficiency_bonus
        self.confidence_threshold = confidence_threshold
        
        # 当前状态
        self.current_example_idx = 0
        self.current_example = None
        self.query_count = 0
        self.original_preference = None
        self.original_confidence = None
        self.attack_history = []
        self.used_attacks = set()
        self.confidence_history = []
        
        # 动作空间：每个攻击方法的动作空间（不再乘以2）
        self.action_space = gym.spaces.Discrete(
            sum(attack.get_action_space_size() for attack in self.attacks)
        )
        
        # 增强的状态空间：包含更多信息
        state_dim = self._calculate_state_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # 动作到攻击方法的映射
        self._build_action_mapping()
        
        # 攻击效果统计
        self.attack_effectiveness = {i: {'success': 0, 'total': 0} for i in range(len(attacks))}
        
        # 样本难度评估
        self.sample_difficulty = {}
        
        # 失败action记录（用于action mask）
        self.failed_actions_in_episode = set()  # 当前episode中失败的action（全局action索引）
        self.failed_action_ids_by_attack = {}   # {attack_idx: set(failed_action_ids)} 每个攻击方法失败的action_id集合
        
    def _calculate_state_dim(self) -> int:
        """计算状态维度state_dim"""
        # 基础状态：样本特征 + 查询历史 + 攻击统计
        base_dim = 20  # 样本特征
        history_dim = 1 + len(self.attacks)  # 查询历史（查询比例 + 攻击one-hot，长度随攻击数量确定）
        attack_stats_dim = len(self.attacks) * 4  # 每个攻击方法的统计信息（增加了失败action比例）
        confidence_dim = 5  # 置信度历史
        planning_dim = 6  # 预算与回合(3) + 多样性/重复率(2) + 可行动作比例(1)
        return base_dim + history_dim + attack_stats_dim + confidence_dim + planning_dim
    
    def _build_action_mapping(self):
        """构建动作到攻击方法的映射"""
        self.action_to_attack = {}
        self.action_to_action_id = {}
        
        action_idx = 0
        for attack_idx, attack in enumerate(self.attacks): # 遍历所有攻击方法
            for action_id in range(attack.get_action_space_size()): # 遍历每个攻击的动作
                self.action_to_attack[action_idx] = attack_idx
                self.action_to_action_id[action_idx] = action_id
                action_idx += 1
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境
        
        Args:
            seed: 随机种子
            options: 可选的配置参数
                - 'original_preference': 预先获取的原始偏好（用于评估模式）
                - 'original_confidence': 预先获取的原始置信度（用于评估模式）
                - 'use_specific_sample': 是否使用特定样本（True=评估模式，False=训练模式）
                - 'sample_idx': 指定使用的样本索引（仅在use_specific_sample=True时有效）
        """
        super().reset(seed=seed)
        
        # 根据options决定样本选择策略
        if options is not None and options.get('use_specific_sample', False):
            # 评估模式：使用指定的样本
            sample_idx = options.get('sample_idx', 0)
            if 0 <= sample_idx < len(self.examples):
                self.current_example_idx = sample_idx
                self.current_example = self.examples[sample_idx]
            else:
                # 如果索引无效，使用第一个样本
                self.current_example_idx = 0
                self.current_example = self.examples[0]
        else:
            # 训练模式：随机选择样本
            self.current_example_idx = random.randint(0, len(self.examples) - 1)
            self.current_example = self.examples[self.current_example_idx]
        
        self.query_count = 0
        
        # 从options中获取original_preference和original_confidence，如果没有则设为None
        if options is not None:
            self.original_preference = options.get('original_preference', None)
            self.original_confidence = options.get('original_confidence', None)
        else:
            self.original_preference = None
            self.original_confidence = None
            
        self.attack_history = []
        self.used_attacks = set()
        self.confidence_history = []
        
        # 重置失败action记录
        self.failed_actions_in_episode = set()
        self.failed_action_ids_by_attack = {}
        
        # 获取初始状态
        state = self._get_enhanced_state()
        
        return state, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        if self.current_example is None:
            raise ValueError("Environment not reset")
        
        # 获取对应的攻击方法
        attack_idx = self.action_to_attack[action]
        action_id = self.action_to_action_id[action]
        attack = self.attacks[attack_idx]
        
        # 确定目标偏好：如果原始偏好是0，目标偏好是1；如果原始偏好是1，目标偏好是0
        if self.original_preference is not None:
            target_pref = 1 - self.original_preference
        else:
            # 如果还没有原始偏好，先获取一次原始偏好
            try:
                judge_response = self.judge.judge_pairwise(self.current_example)
                self.original_preference = judge_response.preference
                self.original_confidence = judge_response.confidence
                target_pref = 1 - self.original_preference
            except Exception as e:
                # Judge调用失败，使用随机偏好
                raise Exception(f"Judge调用失败: {e}")
        
        # 执行攻击（传入target_preference）
        modified_a, modified_b = attack.apply_action(self.current_example, action_id, target_pref)
        
        # 检查是否是FlipAttack或PromptInjectionAttack，如果是则使用修改后的instruction
        if hasattr(attack, 'has_modified_instruction') and attack.has_modified_instruction():
            modified_instruction = attack.get_last_modified_instruction()
        else:
            modified_instruction = self.current_example.instruction
        
        # 创建修改后的样本
        modified_example = PairwiseExample(
            question_id=self.current_example.question_id,
            instruction=modified_instruction,
            response_a=modified_a,
            response_b=modified_b,
            model_a=self.current_example.model_a,
            model_b=self.current_example.model_b,
            metadata=self.current_example.metadata
        )
        
        # 获取judge的偏好
        try:
            judge_response = self.judge.judge_pairwise(modified_example)
            preference = judge_response.preference
            confidence = judge_response.confidence
        except Exception as e:
            # Judge调用失败
            raise Exception(f"Judge调用失败: {e}")
        
        # 检查是否被PPL防御过滤（通过检查raw_response）
        # PPL防御过滤时会返回 "Filtered by PPL defense (PPL: ...)" 格式的 raw_response
        # 使用 startswith 确保精确匹配，避免误判普通 judge 输出中包含类似字符串的情况
        is_filtered = False
        if judge_response.raw_response and judge_response.raw_response.startswith("Filtered by PPL defense"):
            is_filtered = True
        
        # 记录攻击历史
        self.attack_history.append({
            'attack_idx': attack_idx,
            'action_id': action_id,
            'preference': preference,
            'confidence': confidence,
            'query_count': self.query_count,
            'is_filtered': is_filtered
        })
        
        # 更新攻击效果统计
        self.attack_effectiveness[attack_idx]['total'] += 1
        # 使用攻击方法的成功判断逻辑
        # 对于PositionAttack，需要特殊处理（因为交换了位置）
        # 但如果被PPL防御过滤，应该认为是失败的（无论攻击方法如何）
        if is_filtered:
            # 被过滤的查询应该被认为是失败的
            attack_successful = False
        else:
            attack_successful = attack.is_attack_successful(self.original_preference, preference)
        if attack_successful:
            self.attack_effectiveness[attack_idx]['success'] += 1
        
        # 记录失败的action（用于action mask）
        # 如果攻击失败，记录该action，禁止后续重复使用
        if not attack_successful:
            self.failed_actions_in_episode.add(action)
            # 记录该攻击方法下失败的action_id
            if attack_idx not in self.failed_action_ids_by_attack:
                self.failed_action_ids_by_attack[attack_idx] = set()
            self.failed_action_ids_by_attack[attack_idx].add(action_id)
        
        # 记录使用的攻击方法
        self.used_attacks.add(attack_idx)
        self.confidence_history.append(confidence)
        
        self.query_count += 1
        
        # 计算增强奖励
        reward = self._calculate_enhanced_reward(preference, confidence, attack_idx)
        
        # 检查是否结束 - 使用攻击方法的成功判断逻辑
        # 对于PositionAttack，需要特殊处理（因为交换了位置）
        success = (self.original_preference is not None and attack_successful)
        done = (self.query_count >= self.max_queries or success)
        
        # 获取新状态
        next_state = self._get_enhanced_state()
        
        info = {
            'query_count': self.query_count,
            'original_preference': self.original_preference,
            'current_preference': preference,
            'confidence': confidence,
            'success': success,
            'is_filtered': is_filtered,  # Whether this query was filtered by PPL defense
            'attack_history': self.attack_history,
            'used_attacks': list(self.used_attacks),
            'attack_effectiveness': self.attack_effectiveness,
            'diversity_score': len(self.used_attacks) / len(self.attacks),
            'efficiency_score': 1.0 / self.query_count if success else 0.0
        }
        
        return next_state, reward, done, False, info
    
    def _get_enhanced_state(self) -> np.ndarray:
        """获取增强状态"""
        if self.current_example is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        state = []
        
        # 1. 样本特征编码
        sample_features = self._encode_sample_features()
        state.extend(sample_features)
        
        # 2. 查询历史编码
        history_features = self._encode_query_history()
        state.extend(history_features)
        
        # 3. 攻击方法统计
        attack_stats = self._encode_attack_statistics()
        state.extend(attack_stats)
        
        # 4. 置信度历史
        confidence_features = self._encode_confidence_history()
        state.extend(confidence_features)

        # 5. 规划/预算与多样性等
        planning_features = self._encode_planning_features()
        state.extend(planning_features)
        
        return np.array(state, dtype=np.float32)

    def _encode_planning_features(self) -> List[float]:
        """预算/回合进度 + 动作多样性/重复率 + 可行动作比例"""
        features: List[float] = []
        # 预算与回合（3维）
        used_ratio = float(self.query_count) / float(max(1, self.max_queries))
        remaining = max(0, self.max_queries - self.query_count)
        remaining_ratio = float(remaining) / float(max(1, self.max_queries))
        is_last_attempt = 1.0 if (self.max_queries - self.query_count == 1) else 0.0
        features.extend([used_ratio, remaining_ratio, is_last_attempt])

        # 动作多样性/重复率（最近k步，2维）
        k = 5
        if self.attack_history:
            recent = self.attack_history[-k:]
            seq = [h['attack_idx'] for h in recent]
            total = len(seq)
            uniq = len(set(seq)) if total > 0 else 0
            diversity_rate = float(uniq) / float(total) if total > 0 else 0.0
            repeat_rate = 1.0 - diversity_rate if total > 0 else 0.0
        else:
            diversity_rate = 0.0
            repeat_rate = 0.0
        features.extend([diversity_rate, repeat_rate])

        # 可行动作比例（1维）
        feasible_ratio = self._compute_feasible_actions_ratio()
        features.append(feasible_ratio)

        return features

    def _compute_feasible_actions_ratio(self) -> float:
        """计算当前可行动作比例（默认全可行动作=1.0，预留掩码机制扩展）"""
        total_actions = self.action_space.n if hasattr(self.action_space, 'n') else 0
        if total_actions <= 0:
            return 1.0
        # 如未来加入动作掩码，在此返回 feasible_count / total_actions
        return 1.0
    
    def _encode_sample_features(self) -> List[float]:
        """编码样本特征"""
        features = []
        
        # 指令长度
        features.append(len(self.current_example.instruction) / 1000.0)
        
        # 回答A长度
        features.append(len(self.current_example.response_a) / 1000.0)
        
        # 回答B长度
        features.append(len(self.current_example.response_b) / 1000.0)
        
        # 长度差异
        features.append(abs(len(self.current_example.response_a) - len(self.current_example.response_b)) / 1000.0)
        
        # 词汇多样性比较（替代模型信息）
        words_a = set(self.current_example.response_a.lower().split())
        words_b = set(self.current_example.response_b.lower().split())
        
        # 词汇重叠度
        if len(words_a) > 0 and len(words_b) > 0:
            overlap_ratio = len(words_a & words_b) / len(words_a | words_b)
        else:
            overlap_ratio = 0.0
        features.append(overlap_ratio)
        
        # 词汇数量差异（归一化）
        vocab_diff = abs(len(words_a) - len(words_b)) / max(len(words_a) + len(words_b), 1)
        features.append(vocab_diff)
        
        # 填充到固定长度
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def _encode_query_history(self) -> List[float]:
        """编码查询历史"""
        features = []
        
        # 查询次数归一化
        features.append(self.query_count / self.max_queries)
        
        # 最近使用的攻击方法（one-hot，长度= len(self.attacks)）
        if self.attack_history:
            recent_attacks = [h['attack_idx'] for h in self.attack_history[-5:]]
            for i in range(len(self.attacks)):
                features.append(1.0 if i in recent_attacks else 0.0)
        else:
            features.extend([0.0] * len(self.attacks))

        # 直接返回完整长度：1 + len(self.attacks)
        return features
    
    def _encode_attack_statistics(self) -> List[float]:
        """编码攻击方法统计"""
        features = []
        
        for attack_idx in range(len(self.attacks)):
            stats = self.attack_effectiveness[attack_idx]
            # 成功率
            success_rate = stats['success'] / max(stats['total'], 1)
            features.append(success_rate)
            
            # 使用次数
            usage_count = stats['total']
            denom = max(1, self.max_queries)
            features.append(usage_count / float(denom))  # 归一化到[0,1]，基于本episode预算
            
            # 是否在当前episode中使用过
            features.append(1.0 if attack_idx in self.used_attacks else 0.0)
            
            # 该攻击方法在当前episode中失败的action_id比例
            failed_count = len(self.failed_action_ids_by_attack.get(attack_idx, set()))
            total_actions = self.attacks[attack_idx].get_action_space_size()
            failed_ratio = failed_count / max(total_actions, 1)
            features.append(failed_ratio)
        
        return features
    
    def _encode_confidence_history(self) -> List[float]:
        """编码置信度历史"""
        features = []
        
        if self.confidence_history:
            conf = self.confidence_history
            n = len(conf)
            # 1) 平均置信度（全程均值保持不变）
            features.append(np.mean(conf))
            
            # 2) 最近k步EMA增量（替代首末差）
            k = 5
            alpha = 2.0 / (k + 1.0)
            def ema(values: List[float]) -> float:
                if not values:
                    return 0.0
                m = values[0]
                for v in values[1:]:
                    m = alpha * v + (1 - alpha) * m
                return float(m)
            if n >= k + 1:
                window_curr = conf[-k:]
                window_prev = conf[-(k+1):-1]
                ema_inc = ema(window_curr) - ema(window_prev)
            elif n >= 2:
                ema_inc = conf[-1] - conf[0]
            else:
                ema_inc = 0.0
            features.append(float(ema_inc))
            
            # 3) 最近k步滑动方差（替代全程方差）
            if n >= 2:
                tail = conf[-k:] if n >= k else conf
                features.append(float(np.var(tail)))
            else:
                features.append(0.0)
            
            # 4) 最近置信度（保持）
            features.append(float(conf[-1]))
            
            # 5) 置信度是否下降（保持定义，但基于最近k步首末对比）
            if n >= 2:
                base = conf[0] if n < k else conf[-k]
                is_decrease = 1.0 if conf[-1] < base else 0.0
            else:
                is_decrease = 0.0
            features.append(is_decrease)
        else:
            features.extend([0.0] * 5)
        
        return features
    
    def _calculate_enhanced_reward(self, preference: int, confidence: float, attack_idx: int) -> float:
        """计算增强奖励"""
        base_reward = 0.0
        
        # 1. 基础成功奖励
        # 使用当前攻击方法的成功判断逻辑
        # 使用传入的attack_idx获取攻击方法（更可靠）
        attack_successful = False
        if self.original_preference is not None and 0 <= attack_idx < len(self.attacks):
            current_attack = self.attacks[attack_idx]
            attack_successful = current_attack.is_attack_successful(self.original_preference, preference)
        elif self.original_preference is not None:
            # 如果没有攻击方法信息，使用默认逻辑（向后兼容）
            # 注意：对于PositionAttack，这个逻辑是错误的，但这种情况理论上不应该发生
            attack_successful = (preference != self.original_preference)
        
        if attack_successful:
            base_reward += self.success_reward
            
            # 效率奖励：越早成功奖励越高
            efficiency_reward = self.efficiency_bonus * (self.max_queries - self.query_count + 1) / self.max_queries
            base_reward += efficiency_reward
            
            # 置信度奖励：成功时置信度越高奖励越高
            if confidence > self.confidence_threshold:
                base_reward += confidence * 2.0
        
        # 2. 查询惩罚
        query_penalty = self.query_penalty * self.query_count
        base_reward -= query_penalty
        
        # 3. 多样性奖励：鼓励使用不同的攻击方法
        if attack_idx not in self.used_attacks:
            base_reward += self.diversity_bonus
        
        # 4. 探索奖励：鼓励尝试效果不佳的攻击方法
        attack_stats = self.attack_effectiveness[attack_idx]
        if attack_stats['total'] > 0:
            success_rate = attack_stats['success'] / attack_stats['total']
            if success_rate < 0.3:  # 效果不佳的攻击方法
                base_reward += 0.1
        
        # 5. 置信度变化奖励：如果置信度下降，说明攻击有效
        if len(self.confidence_history) > 0:
            confidence_change = confidence - self.confidence_history[0]
            if confidence_change < -0.1:  # 置信度显著下降
                base_reward += 0.5
        
        return base_reward
    
    def get_action_mask(self) -> np.ndarray:
        """
        生成动作掩码：1表示可用，0表示禁用
        
        规则：
        - 如果某个action在当前episode中失败过，则禁用
        - 允许同一攻击方法的不同action_id（只要该action_id没失败过）
        
        Returns:
            action_mask: shape=(action_space.n,), dtype=np.float32
        """
        mask = np.ones(self.action_space.n, dtype=np.float32)
        
        # 禁用所有在当前episode中失败过的action
        for failed_action in self.failed_actions_in_episode:
            mask[failed_action] = 0.0
        
        return mask
    
    def get_attack_recommendations(self) -> List[int]:
        """获取攻击方法推荐"""
        # 基于历史效果推荐攻击方法
        recommendations = []
        
        for attack_idx in range(len(self.attacks)):
            stats = self.attack_effectiveness[attack_idx]
            if stats['total'] > 0:
                success_rate = stats['success'] / stats['total']
                # 推荐成功率高的攻击方法
                if success_rate > 0.5:
                    recommendations.append(attack_idx)
        
        # 如果没有高成功率的，推荐使用次数少的
        if not recommendations:
            usage_counts = [(i, stats['total']) for i, stats in self.attack_effectiveness.items()]
            usage_counts.sort(key=lambda x: x[1])
            recommendations = [i for i, _ in usage_counts[:3]]
        
        return recommendations
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """渲染环境状态"""
        if mode == 'human':
            print(f"Example: {self.current_example.question_id}")
            print(f"Query count: {self.query_count}")
            print(f"Original preference: {self.original_preference}")
            print(f"Used attacks: {self.used_attacks}")
            print(f"Diversity score: {len(self.used_attacks) / len(self.attacks):.3f}")
            if self.attack_history:
                last_attack = self.attack_history[-1]
                print(f"Last attack: {last_attack}")
    
    def close(self):
        """关闭环境"""
        pass


