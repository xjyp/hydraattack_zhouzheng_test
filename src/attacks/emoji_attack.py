"""
Emoji Attack - 通过插入积极emoji表情来引导judge LLM发生偏转 (优化版)
策略性在关键位置（开头、结尾、句子结尾等）插入emoji表情，提高可见性和影响力
"""

from typing import List, Tuple
import random
import re

try:
    from .base import BaseAttack
    from data_types import PairwiseExample, AttackType
except ImportError:
    from attacks.base import BaseAttack
    from data_types import PairwiseExample, AttackType


class EmojiAttack(BaseAttack):
    """Emoji Attack攻击方法 - 通过策略性插入积极emoji表情诱导判断 (优化版)"""
    
    def __init__(self, **kwargs):
        super().__init__(AttackType.EMOJI_ATTACK, **kwargs)
        
        # 10个候选emoji表情（积极表情）
        self.emoji_candidates = [
            "😊",  # 笑脸
            "👍",  # 点赞
            "⭐",  # 星星
            "✨",  # 闪光
            "💯",  # 一百分
            "🎉",  # 庆祝
            "👏",  # 鼓掌
            "❤️",  # 爱心
            "🌟",  # 闪亮星星
            "🔥"   # 火焰
        ]
        
        # 优化：关键位置插入策略
        self.use_strategic_positions = True  # 使用策略性位置
        self.ensure_start_end = True  # 确保在开头和结尾插入
        self.max_insertion_positions = 8  # 减少随机位置，增加策略位置
    
    def _find_insertion_positions(self, text: str, num_positions: int) -> List[int]:
        """
        在文本中找到策略性的插入位置（优化版）
        优先选择：开头、结尾、句子结尾、段落结尾等关键位置
        
        Args:
            text: 目标文本
            num_positions: 需要插入的位置数量
        
        Returns:
            插入位置的索引列表（按升序排列）
        """
        if len(text) == 0:
            return []
        
        positions = []
        
        # 策略1: 确保在开头插入（最显眼的位置）
        if self.ensure_start_end and len(text) > 0:
            positions.append(0)
        
        # 策略2: 找到所有句子结尾位置（高优先级）
        sentence_end_positions = []
        for match in re.finditer(r'[.!?]\s+', text):
            sentence_end_positions.append(match.end())
        # 如果文本以句号结尾，也添加
        if text.rstrip().endswith(('.', '!', '?')):
            sentence_end_positions.append(len(text.rstrip()))
        
        # 策略3: 找到段落结尾（换行后的位置）
        paragraph_end_positions = []
        for match in re.finditer(r'\n\s*\n', text):
            paragraph_end_positions.append(match.end())
        
        # 策略4: 找到重要关键词后的位置（基于实际数据分析的高频词）
        # 这些关键词在实际数据中出现频率高，表示"更好"或"更正确"的回答
        important_keywords = [
            r'\bright\b', r'\btrue\b', r'\bvalid\b',  # 最高频（220, 207, 142次）
            r'\beffective\b', r'\bbest\b', r'\bcomplete\b', r'\befficient\b',  # 高频（70, 63, 47, 47次）
            r'\bbetter\b', r'\bcorrect\b', r'\bclear\b', r'\boptimal\b',  # 中高频（43, 41, 39, 34次）
            r'\bchosen\b', r'\bappropriate\b', r'\bproper\b', r'\bsuitable\b',  # 中频（30, 22, 22, 20次）
            r'\bdetailed\b', r'\bprofessional\b', r'\bsufficient\b', r'\bperfect\b', r'\brecommended\b'  # 补充（19, 18, 15, 13, 13次）
        ]
        keyword_positions = []
        for pattern in important_keywords:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # 在关键词后找到下一个空格或标点
                next_space = text.find(' ', match.end())
                if next_space != -1:
                    keyword_positions.append(next_space + 1)
                else:
                    keyword_positions.append(match.end())
        
        # 策略5: 找到标点符号后的位置（作为备选）
        punctuation_positions = []
        for match in re.finditer(r'[.,!?;:]\s*', text):
            punctuation_positions.append(match.end())
        
        # 合并所有策略位置，按优先级排序
        all_strategic_positions = []
        all_strategic_positions.extend(sentence_end_positions)  # 最高优先级
        all_strategic_positions.extend(paragraph_end_positions)
        all_strategic_positions.extend(keyword_positions)
        all_strategic_positions.extend(punctuation_positions)
        
        # 去重并排序
        all_strategic_positions = sorted(set(all_strategic_positions))
        
        # 选择策略位置
        remaining_positions = num_positions - len(positions)
        if remaining_positions > 0:
            if len(all_strategic_positions) >= remaining_positions:
                # 优先选择句子结尾，然后均匀分布其他位置
                selected = []
                # 先选句子结尾
                sentence_selected = min(len(sentence_end_positions), remaining_positions // 2)
                selected.extend(random.sample(sentence_end_positions, sentence_selected) if sentence_selected > 0 else [])
                
                # 再从其他位置选择
                remaining = remaining_positions - len(selected)
                other_positions = [p for p in all_strategic_positions if p not in selected]
                if remaining > 0 and len(other_positions) > 0:
                    selected.extend(random.sample(other_positions, min(remaining, len(other_positions))))
                
                positions.extend(selected)
            else:
                positions.extend(all_strategic_positions)
        
        # 策略6: 确保在结尾插入（最显眼的位置之一）
        if self.ensure_start_end and len(text) > 0:
            if len(text) not in positions:
                positions.append(len(text))
        
        # 如果位置还不够，在文本中均匀分布补充
        if len(positions) < num_positions:
            remaining = num_positions - len(positions)
            step = max(1, len(text) // (remaining + 1))
            additional = [step * (i + 1) for i in range(remaining)]
            additional = [min(p, len(text)) for p in additional if p not in positions]
            positions.extend(additional)
        
        # 去重、排序并限制数量
        positions = sorted(set(positions))
        if len(positions) > num_positions:
            # 保持开头和结尾，然后均匀选择其他位置
            if 0 in positions:
                positions.remove(0)
            if len(text) in positions and len(positions) > num_positions:
                positions.remove(len(text))
            
            if len(positions) > num_positions:
                # 均匀采样
                step = len(positions) // num_positions
                positions = [positions[i * step] for i in range(num_positions)]
            
            # 重新添加开头和结尾
            if self.ensure_start_end:
                if 0 not in positions:
                    positions.insert(0, 0)
                if len(text) not in positions:
                    positions.append(len(text))
        
        return sorted(positions)
    
    def _insert_emoji_at_positions(self, text: str, emoji: str, positions: List[int]) -> str:
        """
        在指定位置插入emoji
        
        Args:
            text: 原始文本
            emoji: 要插入的emoji
            positions: 插入位置的索引列表（按升序排列）
        
        Returns:
            修改后的文本
        """
        if not positions:
            return text
        
        # 从后往前插入，避免位置索引变化
        result = text
        for pos in reversed(positions):
            if 0 <= pos <= len(result):
                result = result[:pos] + emoji + result[pos:]
        
        return result
    
    def attack(self, example: PairwiseExample, target_preference: int = None) -> Tuple[str, str]:
        """执行Emoji Attack (优化版)"""
        if target_preference is None:
            target_preference = random.randint(0, 1)
        
        # 随机选择一个emoji
        emoji = random.choice(self.emoji_candidates)
        
        # 优化：根据文本长度调整插入位置数量
        if target_preference == 0:
            target_text = example.response_a
        else:
            target_text = example.response_b
        
        # 根据文本长度动态调整位置数量
        text_length = len(target_text)
        if text_length < 100:
            num_positions = random.randint(2, 4)
        elif text_length < 500:
            num_positions = random.randint(3, 6)
        else:
            num_positions = random.randint(4, self.max_insertion_positions)
        
        if target_preference == 0:
            # 在回答A中插入emoji
            positions = self._find_insertion_positions(target_text, num_positions)
            modified_a = self._insert_emoji_at_positions(target_text, emoji, positions)
            return modified_a, example.response_b
        else:
            # 在回答B中插入emoji
            positions = self._find_insertion_positions(target_text, num_positions)
            modified_b = self._insert_emoji_at_positions(target_text, emoji, positions)
            return example.response_a, modified_b
    
    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        # 10个不同的emoji候选
        return len(self.emoji_candidates)
    
    def get_state_dim(self) -> int:
        """获取状态维度"""
        return 6 + 5 + 2  # 基础特征 + 词汇特征 + 标点特征
    
    def apply_action(self, example: PairwiseExample, action: int, target_preference: int = 0) -> Tuple[str, str]:
        """应用具体动作 - 支持target_preference控制 (优化版)"""
        # 根据action选择对应的emoji
        if 0 <= action < len(self.emoji_candidates):
            emoji = self.emoji_candidates[action]
        else:
            emoji = random.choice(self.emoji_candidates)
        
        # 根据target_preference决定修改哪个response
        if target_preference == 0:  # 偏好A
            target_text = example.response_a
        else:  # 偏好B
            target_text = example.response_b
        
        # 根据文本长度动态调整位置数量
        text_length = len(target_text)
        if text_length < 100:
            num_positions = random.randint(2, 4)
        elif text_length < 500:
            num_positions = random.randint(3, 6)
        else:
            num_positions = random.randint(4, self.max_insertion_positions)
        
        positions = self._find_insertion_positions(target_text, num_positions)
        
        if target_preference == 0:  # 偏好A
            modified_a = self._insert_emoji_at_positions(target_text, emoji, positions)
            return modified_a, example.response_b
        else:  # 偏好B
            modified_b = self._insert_emoji_at_positions(target_text, emoji, positions)
            return example.response_a, modified_b
    
    def get_action_description(self, action: int) -> str:
        """获取动作描述"""
        if 0 <= action < len(self.emoji_candidates):
            emoji = self.emoji_candidates[action]
            emoji_names = {
                "😊": "smile",
                "👍": "thumbs_up",
                "⭐": "star",
                "✨": "sparkles",
                "💯": "hundred",
                "🎉": "celebration",
                "👏": "clap",
                "❤️": "heart",
                "🌟": "glowing_star",
                "🔥": "fire"
            }
            return f"emoji_{emoji_names.get(emoji, 'unknown')}"
        else:
            return "emoji_unknown"

