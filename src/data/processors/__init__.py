"""
数据处理器模块
"""

from .alpaca_eval_processor import AlpacaEvalProcessor
from .code_judge_bench_processor import CodeJudgeBenchProcessor
from .arena_hard_processor import ArenaHardProcessor

__all__ = [
    "AlpacaEvalProcessor",
    "CodeJudgeBenchProcessor",
    "ArenaHardProcessor"
]
