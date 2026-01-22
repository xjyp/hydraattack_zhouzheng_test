import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class AttackSample:
    """单个攻击样本的详细信息"""
    sample_id: str
    question_id: str
    instruction: str
    response_a: str
    response_b: str
    model_a: str
    model_b: str
    original_preference: str
    original_confidence: float
    attack_method: str
    attack_action: str
    modified_response_a: str
    modified_response_b: str
    new_preference: str
    new_confidence: float
    success: bool
    timestamp: float
    modified_instruction: str = None  # 修改后的指令（针对Prompt的攻击）
    queries_used: int = 1  # 使用的查询次数
    efficiency_score: float = 0.0  # 效率分数
    attack_time: float = 0.0  # 攻击耗时（秒）
    metadata: Dict[str, Any] = None


@dataclass
class AttackResults:
    """攻击结果汇总"""
    attack_method: str
    total_samples: int
    successful_attacks: int
    success_rate: float
    action_space_size: int
    avg_confidence_change: float
    avg_queries_used: float = 0.0  # AQA: 平均查询次数（所有攻击）
    avg_queries_successful: float = 0.0  # AQSA: 平均查询次数（成功攻击）
    avg_efficiency_score: float = 0.0  # 平均效率分数
    avg_attack_time: float = 0.0  # 平均攻击时间
    total_queries_saved: int = 0  # 总节省查询次数
    samples: List[AttackSample] = None


@dataclass
class BenchmarkResults:
    """Benchmark结果汇总"""
    benchmark_name: str
    total_samples: int
    baseline_results: Dict[str, AttackResults]
    rl_results: Optional[AttackResults] = None
    overall_success_rate: float = 0.0
    best_attack_method: str = ""
    worst_attack_method: str = ""


@dataclass
class ExperimentConfig:
    """实验配置"""
    gpu_config: Dict[str, Any]
    data_config: Dict[str, Any]
    attack_config: Dict[str, Any]
    judge_config: Dict[str, Any]
    rl_config: Dict[str, Any]
    output_config: Dict[str, Any]
    timestamp: float
    random_seed: int


class HydraLogger:
    """Hydra-Attack 日志管理器"""
    
    def __init__(self, log_dir: str = "./logs", results_dir: str = "./results"):
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        
        # 创建目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"hydra_attack_{self.timestamp}"
        
        # 设置日志文件
        self.log_file = self.log_dir / f"{self.session_id}.log"
        self.config_file = self.log_dir / f"{self.session_id}_config.json"
        self.results_file = self.results_dir / f"{self.session_id}_results.json"
        self.samples_file = self.log_dir / f"{self.session_id}_samples.json"
        
        # 配置日志
        self._setup_logging()
        
        # 存储数据
        self.config: Optional[ExperimentConfig] = None
        self.benchmark_results: List[BenchmarkResults] = []
        self.all_samples: List[AttackSample] = []
        
    def _setup_logging(self):
        """设置日志配置"""
        # 创建logger
        self.logger = logging.getLogger('hydra_attack')
        self.logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def log_config(self, config: ExperimentConfig):
        """记录实验配置"""
        self.config = config
        self.logger.info("=" * 80)
        self.logger.info("🚀 Hydra-Attack 实验配置")
        self.logger.info("=" * 80)
        
        # 记录GPU配置
        self.logger.info(f"🖥️  GPU配置: {config.gpu_config}")
        
        # 记录数据配置
        self.logger.info(f"📊 数据配置: {config.data_config}")
        
        # 记录攻击配置
        self.logger.info(f"⚔️  攻击配置: {config.attack_config}")
        
        # 记录Judge配置
        self.logger.info(f"⚖️  Judge配置: {config.judge_config}")
        
        # 记录RL配置
        if config.rl_config:
            self.logger.info(f"🧠 RL配置: {config.rl_config}")
        
        # 记录输出配置
        self.logger.info(f"📁 输出配置: {config.output_config}")
        
        # 保存配置到文件
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 配置已保存到: {self.config_file}")
        
    def log_benchmark_start(self, benchmark_name: str, sample_count: int):
        """记录benchmark开始"""
        self.logger.info("=" * 80)
        self.logger.info(f"📊 开始测试Benchmark: {benchmark_name}")
        self.logger.info(f"📈 样本数量: {sample_count}")
        self.logger.info("=" * 80)
        
    def log_attack_start(self, attack_method: str, action_space_size: int):
        """记录攻击方法开始"""
        self.logger.info(f"⚔️  开始测试攻击方法: {attack_method}")
        self.logger.info(f"🎯 动作空间大小: {action_space_size}")
        
    def log_attack_sample(self, sample: AttackSample):
        """记录单个攻击样本"""
        self.all_samples.append(sample)
        
        status = "✅ 成功" if sample.success else "❌ 失败"
        self.logger.info(f"  样本 {sample.sample_id}: {status}")
        self.logger.info(f"    原始偏好: {sample.original_preference} (置信度: {sample.original_confidence:.3f})")
        self.logger.info(f"    新偏好: {sample.new_preference} (置信度: {sample.new_confidence:.3f})")
        self.logger.info(f"    攻击方法: {sample.attack_method}")
        self.logger.info(f"    攻击动作: {sample.attack_action}")
        
        # 记录效率指标
        self.logger.info(f"    ⚡ 查询次数: {sample.queries_used}")
        self.logger.info(f"    🎯 效率分数: {sample.efficiency_score:.3f}")
        self.logger.info(f"    ⏱️  攻击耗时: {sample.attack_time:.3f}s")
        
        # 记录LLM输入输出
        self.logger.info(f"    📝 原始指令: {sample.instruction[:100]}...")
        
        # 如果有修改后的指令，显示它（针对Prompt的攻击）
        if sample.modified_instruction:
            self.logger.info(f"    📝 修改后指令: {sample.modified_instruction[:100]}...")
        
        self.logger.info(f"    📝 原始回答A: {sample.response_a[:100]}...")
        self.logger.info(f"    📝 原始回答B: {sample.response_b[:100]}...")
        self.logger.info(f"    📝 修改后回答A: {sample.modified_response_a[:100]}...")
        self.logger.info(f"    📝 修改后回答B: {sample.modified_response_b[:100]}...")
        
    def log_attack_results(self, attack_results: AttackResults):
        """记录攻击方法结果"""
        self.logger.info(f"📊 {attack_results.attack_method} 结果:")
        self.logger.info(f"  总样本数: {attack_results.total_samples}")
        self.logger.info(f"  成功攻击数: {attack_results.successful_attacks}")
        self.logger.info(f"  成功率: {attack_results.success_rate:.2%}")
        self.logger.info(f"  动作空间大小: {attack_results.action_space_size}")
        self.logger.info(f"  平均置信度变化: {attack_results.avg_confidence_change:.3f}")
        
        # 记录效率指标
        self.logger.info(f"  ⚡ 平均查询次数: {attack_results.avg_queries_used:.2f}")
        self.logger.info(f"  🎯 平均效率分数: {attack_results.avg_efficiency_score:.3f}")
        self.logger.info(f"  ⏱️  平均攻击时间: {attack_results.avg_attack_time:.3f}s")
        self.logger.info(f"  💰 总节省查询次数: {attack_results.total_queries_saved}")
        
    def log_rl_results(self, rl_results: AttackResults):
        """记录RL结果"""
        self.logger.info("=" * 80)
        self.logger.info("🧠 强化学习组合攻击结果")
        self.logger.info("=" * 80)
        self.logger.info(f"总样本数: {rl_results.total_samples}")
        self.logger.info(f"成功攻击数: {rl_results.successful_attacks}")
        self.logger.info(f"成功率: {rl_results.success_rate:.2%}")
        self.logger.info(f"总动作空间大小: {rl_results.action_space_size}")
        self.logger.info(f"平均置信度变化: {rl_results.avg_confidence_change:.3f}")
        
    def log_benchmark_results(self, benchmark_results: BenchmarkResults):
        """记录benchmark结果"""
        self.benchmark_results.append(benchmark_results)
        
        self.logger.info("=" * 80)
        self.logger.info(f"📊 {benchmark_results.benchmark_name} 测试结果总结")
        self.logger.info("=" * 80)
        
        # 记录baseline结果
        self.logger.info("Baseline攻击结果 (按成功率排序):")
        sorted_results = sorted(
            benchmark_results.baseline_results.items(),
            key=lambda x: x[1].success_rate,
            reverse=True
        )
        
        for attack_name, result in sorted_results:
            # 显示query次数和效率指标
            query_info = f" [查询: {result.avg_queries_used:.1f}]" if hasattr(result, 'avg_queries_used') else ""
            efficiency_info = f" [效率: {result.avg_efficiency_score:.3f}]" if hasattr(result, 'avg_efficiency_score') else ""
            self.logger.info(f"  {attack_name}: {result.successful_attacks}/{result.total_samples} ({result.success_rate:.2%}) [动作空间: {result.action_space_size}]{query_info}{efficiency_info}")
        
        # 记录RL结果
        if benchmark_results.rl_results:
            self.logger.info(f"\nRL组合攻击结果:")
            rl = benchmark_results.rl_results
            query_info = f" [查询: {rl.avg_queries_used:.1f}]" if hasattr(rl, 'avg_queries_used') else ""
            efficiency_info = f" [效率: {rl.avg_efficiency_score:.3f}]" if hasattr(rl, 'avg_efficiency_score') else ""
            self.logger.info(f"  RL组合攻击: {rl.successful_attacks}/{rl.total_samples} ({rl.success_rate:.2%}) [总动作空间: {rl.action_space_size}]{query_info}{efficiency_info}")
        
        # 记录最佳和最差攻击方法
        if benchmark_results.best_attack_method:
            self.logger.info(f"\n🏆 最佳攻击方法: {benchmark_results.best_attack_method}")
        if benchmark_results.worst_attack_method:
            self.logger.info(f"📉 最差攻击方法: {benchmark_results.worst_attack_method}")
            
    def log_experiment_summary(self):
        """记录实验总结"""
        self.logger.info("=" * 100)
        self.logger.info("🎉 Hydra-Attack 实验总结")
        self.logger.info("=" * 100)
        
        # 计算总体统计
        total_samples = sum(br.total_samples for br in self.benchmark_results)
        total_successful = sum(
            sum(ar.successful_attacks for ar in br.baseline_results.values())
            for br in self.benchmark_results
        )
        overall_success_rate = total_successful / total_samples if total_samples > 0 else 0
        
        self.logger.info(f"📊 总体统计:")
        self.logger.info(f"  测试的Benchmark数量: {len(self.benchmark_results)}")
        self.logger.info(f"  总样本数: {total_samples}")
        self.logger.info(f"  总成功攻击数: {total_successful}")
        self.logger.info(f"  总体成功率: {overall_success_rate:.2%}")
        
        # 记录每个benchmark的结果
        for br in self.benchmark_results:
            self.logger.info(f"\n📈 {br.benchmark_name}:")
            self.logger.info(f"  样本数: {br.total_samples}")
            self.logger.info(f"  成功率: {br.overall_success_rate:.2%}")
            self.logger.info(f"  最佳方法: {br.best_attack_method}")
        
        # 保存结果到文件
        self._save_results()
        
    def _save_results(self):
        """保存结果到文件"""
        # 保存详细结果
        results_data = {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "config": asdict(self.config) if self.config else None,
            "benchmark_results": [asdict(br) for br in self.benchmark_results],
            "overall_stats": {
                "total_benchmarks": len(self.benchmark_results),
                "total_samples": sum(br.total_samples for br in self.benchmark_results),
                "total_successful": sum(
                    sum(ar.successful_attacks for ar in br.baseline_results.values())
                    for br in self.benchmark_results
                ),
                "overall_success_rate": sum(
                    sum(ar.successful_attacks for ar in br.baseline_results.values())
                    for br in self.benchmark_results
                ) / sum(br.total_samples for br in self.benchmark_results) if self.benchmark_results else 0
            }
        }
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # 保存样本详情
        samples_data = {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "samples": [asdict(sample) for sample in self.all_samples]
        }
        
        with open(self.samples_file, 'w', encoding='utf-8') as f:
            json.dump(samples_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 结果已保存到: {self.results_file}")
        self.logger.info(f"💾 样本详情已保存到: {self.samples_file}")
        
    def get_log_file_path(self) -> str:
        """获取日志文件路径"""
        return str(self.log_file)
        
    def get_results_file_path(self) -> str:
        """获取结果文件路径"""
        return str(self.results_file)
        
    def get_samples_file_path(self) -> str:
        """获取样本文件路径"""
        return str(self.samples_file)
