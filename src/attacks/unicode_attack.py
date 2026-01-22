"""
Unicode Attack - Unicode字符攻击 (优化版)
动作数量从66减少到10，提高平衡性
"""

from typing import List, Tuple
import random

try:
    from .base import BaseAttack
    from data_types import PairwiseExample, AttackType
except ImportError:
    from attacks.base import BaseAttack
    from data_types import PairwiseExample, AttackType


class UnicodeAttack(BaseAttack):
    """Unicode Attack攻击方法 - 通过Unicode字符影响判断 (优化版)"""
    
    def __init__(self, **kwargs):
        super().__init__(AttackType.UNICODE_ATTACK, **kwargs)
        
        # 优化后的Unicode攻击 - 扩展到10个动作
        self.attack_strategies = [
            "zero_width_injection",
            "confusable_replacement", 
            "unicode_emphasis",
            "mixed_script",
            "invisible_separators",
            "homograph_attack",
            "bidirectional_override",
            "unicode_normalization",
            "unicode_whitespace",
            "unicode_punctuation"
        ]
        
        # 零宽字符 (2个最有效的)
        self.zero_width_chars = ["\u200B", "\u200C"]  # ZERO WIDTH SPACE, ZERO WIDTH NON-JOINER
        
        # 混淆字符对 (2个最有效的)
        self.confusable_chars = [
            ("a", "а"),  # Latin 'a' vs Cyrillic 'а'
            ("o", "о")   # Latin 'o' vs Cyrillic 'о'
        ]
        
        # 特殊Unicode字符 (2个最有效的)
        self.special_unicode = ["\u202E", "\u202D"]  # RIGHT-TO-LEFT OVERRIDE, LEFT-TO-RIGHT OVERRIDE
        
        # 攻击策略 (2个最有效的)
        self.strategy_templates = [
            "**{content}**",  # 加粗
            "_{content}_"     # 斜体
        ]
    
    def attack(self, example: PairwiseExample, target_preference: int = None) -> Tuple[str, str]:
        """执行Unicode Attack"""
        if target_preference is None:
            target_preference = random.randint(0, 1)
        
        # 随机选择攻击策略
        strategy = random.choice(self.attack_strategies)
        
        # 随机选择修改哪个response
        modify_a = random.random() < 0.5
        
        if strategy == "zero_width_injection":
            char = random.choice(self.zero_width_chars)
            if modify_a:
                modified_a = self._inject_zero_width(example.response_a, char)
                return modified_a, example.response_b
            else:
                modified_b = self._inject_zero_width(example.response_b, char)
                return example.response_a, modified_b
            
        elif strategy == "confusable_replacement":
            char_pair = random.choice(self.confusable_chars)
            if modify_a:
                modified_a = self._replace_confusables(example.response_a, char_pair)
                return modified_a, example.response_b
            else:
                modified_b = self._replace_confusables(example.response_b, char_pair)
                return example.response_a, modified_b
            
        elif strategy == "unicode_emphasis":
            char = random.choice(self.special_unicode)
            if modify_a:
                modified_a = self._add_unicode_emphasis(example.response_a, char)
                return modified_a, example.response_b
            else:
                modified_b = self._add_unicode_emphasis(example.response_b, char)
                return example.response_a, modified_b
            
        elif strategy == "mixed_script":
            if modify_a:
                modified_a = self._mixed_script_attack(example.response_a)
                return modified_a, example.response_b
            else:
                modified_b = self._mixed_script_attack(example.response_b)
                return example.response_a, modified_b
            
        elif strategy == "invisible_separators":
            if modify_a:
                modified_a = self._invisible_separators(example.response_a)
                return modified_a, example.response_b
            else:
                modified_b = self._invisible_separators(example.response_b)
                return example.response_a, modified_b
            
        elif strategy == "homograph_attack":
            if modify_a:
                modified_a = self._homograph_attack(example.response_a)
                return modified_a, example.response_b
            else:
                modified_b = self._homograph_attack(example.response_b)
                return example.response_a, modified_b
            
        elif strategy == "bidirectional_override":
            if modify_a:
                modified_a = self._bidirectional_override(example.response_a)
                return modified_a, example.response_b
            else:
                modified_b = self._bidirectional_override(example.response_b)
                return example.response_a, modified_b
            
        elif strategy == "unicode_whitespace":
            if modify_a:
                modified_a = self._unicode_whitespace(example.response_a)
                return modified_a, example.response_b
            else:
                modified_b = self._unicode_whitespace(example.response_b)
                return example.response_a, modified_b
            
        elif strategy == "unicode_punctuation":
            if modify_a:
                modified_a = self._unicode_punctuation(example.response_a)
                return modified_a, example.response_b
            else:
                modified_b = self._unicode_punctuation(example.response_b)
                return example.response_a, modified_b
            
        else:  # unicode_normalization
            if modify_a:
                modified_a = self._unicode_normalization(example.response_a)
                return modified_a, example.response_b
            else:
                modified_b = self._unicode_normalization(example.response_b)
                return example.response_a, modified_b
    
    def _inject_zero_width(self, text: str, char: str) -> str:
        """注入零宽字符"""
        if len(text) > 0:
            return text[0] + char + text[1:]
        return text
    
    def _replace_confusables(self, text: str, char_pair: Tuple[str, str]) -> str:
        """替换混淆字符"""
        old_char, new_char = char_pair
        return text.replace(old_char, new_char)
    
    def _add_unicode_emphasis(self, text: str, char: str) -> str:
        """添加Unicode强调"""
        return f"{char}{text}{char}"
    
    def _mixed_script_attack(self, text: str) -> str:
        """混合脚本攻击 - 使用不同脚本的字符"""
        # 将拉丁字符替换为西里尔字符
        mixed_text = text.replace('a', 'а').replace('o', 'о').replace('e', 'е')
        return f"**{mixed_text}**"
    
    def _invisible_separators(self, text: str) -> str:
        """不可见分隔符 - 在字符间插入零宽字符"""
        if len(text) > 1:
            return text[0] + '\u200B' + text[1:-1] + '\u200B' + text[-1]
        return text
    
    def _homograph_attack(self, text: str) -> str:
        """同形异义攻击 - 使用视觉相似的字符"""
        # 替换一些容易混淆的字符
        homograph_text = text.replace('l', '|').replace('I', 'l').replace('0', 'O')
        return f"_{homograph_text}_"
    
    def _bidirectional_override(self, text: str) -> str:
        """双向覆盖攻击 - 改变文本显示方向"""
        return f"\u202E{text}\u202D"
    
    def _unicode_normalization(self, text: str) -> str:
        """Unicode规范化攻击 - 使用特殊Unicode字符包装"""
        return f"\u2060{text}\u2060"  # 使用WORD JOINER字符
    
    def _unicode_whitespace(self, text: str) -> str:
        """Unicode空白字符攻击 - 使用特殊空白字符"""
        # 使用EM SPACE和EN SPACE替换普通空格
        return text.replace(' ', '\u2003').replace('\t', '\u2009')
    
    def _unicode_punctuation(self, text: str) -> str:
        """Unicode标点攻击 - 使用特殊标点字符"""
        # 使用全角标点替换半角标点
        return text.replace('.', '。').replace(',', '，').replace('!', '！').replace('?', '？')
    
    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        # 优化后：攻击策略数量 = 10
        return len(self.attack_strategies)
    
    def get_state_dim(self) -> int:
        """获取状态维度"""
        return 6 + 5 + 2  # 基础特征 + 词汇特征 + 标点特征
    
    def apply_action(self, example: PairwiseExample, action: int, target_preference: int = 0) -> Tuple[str, str]:
        """应用具体动作 - 支持target_preference控制"""
        strategy = self.attack_strategies[action]
        
        if strategy == "zero_width_injection":
            # 确定性选择第一个零宽字符
            char = self.zero_width_chars[0]
            modified_text = self._inject_zero_width(example.response_a, char)
            if target_preference == 0:  # 偏好A
                return modified_text, example.response_b
            else:  # 偏好B
                modified_text = self._inject_zero_width(example.response_b, char)
                return example.response_a, modified_text
            
        elif strategy == "confusable_replacement":
            # 确定性选择第一个混淆字符对
            char_pair = self.confusable_chars[0]
            modified_text = self._replace_confusables(example.response_a, char_pair)
            if target_preference == 0:  # 偏好A
                return modified_text, example.response_b
            else:  # 偏好B
                modified_text = self._replace_confusables(example.response_b, char_pair)
                return example.response_a, modified_text
            
        elif strategy == "unicode_emphasis":
            # 确定性选择第一个特殊Unicode字符
            char = self.special_unicode[0]
            modified_text = self._add_unicode_emphasis(example.response_a, char)
            if target_preference == 0:  # 偏好A
                return modified_text, example.response_b
            else:  # 偏好B
                modified_text = self._add_unicode_emphasis(example.response_b, char)
                return example.response_a, modified_text
            
        elif strategy == "mixed_script":
            modified_text = self._mixed_script_attack(example.response_a)
            if target_preference == 0:  # 偏好A
                return modified_text, example.response_b
            else:  # 偏好B
                modified_text = self._mixed_script_attack(example.response_b)
                return example.response_a, modified_text
            
        elif strategy == "invisible_separators":
            modified_text = self._invisible_separators(example.response_a)
            if target_preference == 0:  # 偏好A
                return modified_text, example.response_b
            else:  # 偏好B
                modified_text = self._invisible_separators(example.response_b)
                return example.response_a, modified_text
            
        elif strategy == "homograph_attack":
            modified_text = self._homograph_attack(example.response_a)
            if target_preference == 0:  # 偏好A
                return modified_text, example.response_b
            else:  # 偏好B
                modified_text = self._homograph_attack(example.response_b)
                return example.response_a, modified_text
            
        elif strategy == "bidirectional_override":
            modified_text = self._bidirectional_override(example.response_a)
            if target_preference == 0:  # 偏好A
                return modified_text, example.response_b
            else:  # 偏好B
                modified_text = self._bidirectional_override(example.response_b)
                return example.response_a, modified_text
            
        elif strategy == "unicode_whitespace":
            modified_text = self._unicode_whitespace(example.response_a)
            if target_preference == 0:  # 偏好A
                return modified_text, example.response_b
            else:  # 偏好B
                modified_text = self._unicode_whitespace(example.response_b)
                return example.response_a, modified_text
            
        elif strategy == "unicode_punctuation":
            modified_text = self._unicode_punctuation(example.response_a)
            if target_preference == 0:  # 偏好A
                return modified_text, example.response_b
            else:  # 偏好B
                modified_text = self._unicode_punctuation(example.response_b)
                return example.response_a, modified_text
            
        else:  # unicode_normalization
            modified_text = self._unicode_normalization(example.response_a)
            if target_preference == 0:  # 偏好A
                return modified_text, example.response_b
            else:  # 偏好B
                modified_text = self._unicode_normalization(example.response_b)
                return example.response_a, modified_text
    
    def get_action_description(self, action: int) -> str:
        """获取动作描述"""
        return self.attack_strategies[action]
