#!/usr/bin/env python3
"""
Hydra-Attack 主测试脚本
统一运行所有攻击测试
"""

import argparse
import sys
import json
import time
import os
import random
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# 添加src到路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_types import PairwiseExample, AttackResult, BenchmarkType, AttackType
from attacks import (
    FlipAttackFWO, FlipAttackFCW, FlipAttackFCS, UncertaintyAttack, PositionAttack, DistractorAttack,
    PromptInjectionAttack, MarkerInjectionAttack, FormattingAttack,
    AuthorityAttack, UnicodeAttack, CoTPoisoningAttack, EmojiAttack
)
from evaluation.qwen_judge import create_qwen_judge
from evaluation.llama_judge import create_llama_judge
from evaluation.glm_judge import create_glm_judge
from evaluation.mistral_judge import create_mistral_judge
from evaluation.gemma_judge import create_gemma_judge
from evaluation.gemma_judge1b import create_gemma_judge as create_gemma_judge_1b
from utils.logger import HydraLogger, AttackSample, AttackResults, BenchmarkResults, ExperimentConfig


def load_benchmark_data(benchmark: str, max_samples = 10, random_seed: int = 42) -> List[PairwiseExample]:
    """加载测试数据（确保与其他方法使用相同数据）"""
    # 优先使用分割后的测试数据
    test_file = f"data/split/{benchmark}_test.json"
    
    if os.path.exists(test_file):
        # 使用预分割的测试数据
        with open(test_file, 'r') as f:
            data = json.load(f)
        print(f"✅ 使用预分割的测试数据: {len(data)} 样本")
    else:
        raise ValueError(f"❌ {benchmark} 测试数据不存在")
    
    examples = []
    for sample in data:
        example = PairwiseExample(
            question_id=sample["question_id"],
            instruction=sample["instruction"],
            response_a=sample["response_a"],
            response_b=sample["response_b"],
            model_a=sample["model_a"],
            model_b=sample["model_b"]
        )
        examples.append(example)
    
    return examples


def get_attack_methods(attack_methods: List[str]) -> Dict[str, Any]:
    """获取攻击方法实例"""
    all_attacks = {
        "flip_attack_fwo": FlipAttackFWO(),
        "flip_attack_fcw": FlipAttackFCW(),
        "flip_attack_fcs": FlipAttackFCS(),
        "uncertainty_attack": UncertaintyAttack(),
        "position_attack": PositionAttack(),
        "distractor_attack": DistractorAttack(),
        "prompt_injection_attack": PromptInjectionAttack(),
        "marker_injection_attack": MarkerInjectionAttack(),
        "formatting_attack": FormattingAttack(),
        "authority_attack": AuthorityAttack(),
        "unicode_attack": UnicodeAttack(),
        "cot_poisoning_attack": CoTPoisoningAttack(),
        "emoji_attack": EmojiAttack(),
    }
    
    if "all" in attack_methods:
        return all_attacks
    
    selected_attacks = {}
    for method in attack_methods:
        if method in all_attacks:
            selected_attacks[method] = all_attacks[method]
        else:
            print(f"⚠️  未知的攻击方法: {method}")
    
    return selected_attacks


def test_attacks(examples: List[PairwiseExample], judge, attacks: Dict[str, Any], 
                max_samples = 10, logger: HydraLogger = None, max_queries: int = None, 
                results_dir: str = None, benchmark: str = None,
                experiment_config: Dict[str, Any] = None) -> Dict[str, AttackResults]:
    """测试攻击方法"""
    # if logger:
    #     logger.log_benchmark_start("攻击方法测试", min(len(examples), max_samples))
    
    results = {}
    
    # 添加攻击方法进度条
    attack_progress = tqdm(
        attacks.items(), 
        desc="测试攻击方法",
        unit="攻击",
        ncols=100,
        position=1,
        leave=False
    )
    
    for attack_name, attack in attack_progress:
        # 处理所有攻击方法（现在都是独立的）
        attack_progress.set_description(f"测试 {attack_name}")
        if logger:
            logger.log_attack_start(attack_name, attack.get_action_space_size())
        
        attack_results = test_single_attack(
            examples, judge, attack, max_samples, attack_name, logger, max_queries
        )
        results[attack_name] = attack_results
        
        if logger:
            logger.log_attack_results(attack_results)
        
        # 立即保存单个攻击方法的结果文件
        if results_dir and benchmark:
            save_single_attack_result(attack_results, results_dir, benchmark, experiment_config)
    
    return results


def test_single_attack(examples: List[PairwiseExample], judge, attack, max_samples, 
                      attack_name: str, logger: HydraLogger = None, max_queries: int = None) -> AttackResults:
    """测试单个攻击方法"""
    successful_attacks = 0
    total_samples = 0
    samples = []
    confidence_changes = []
    total_queries = 0  # All attacks (success + failure)
    total_queries_successful = 0  # Only successful attacks
    total_time = 0.0
    efficiency_scores = []
    
    # 添加进度条
    if max_samples == "full" or max_samples is None:
        sample_limit = len(examples)
    else:
        sample_limit = min(max_samples, len(examples))
    
    progress_bar = tqdm(
        enumerate(examples[:sample_limit]), 
        total=sample_limit,
        desc=f"测试 {attack_name}",
        unit="样本",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for i, example in progress_bar:
        sample_id = f"{attack_name}_sample_{i+1}"
        
        try:
            # 获取原始偏好
            original_prompt = judge.get_judge_prompt(example)
            original_response = judge.judge_pairwise(example)
            original_preference = "A" if original_response.preference == 0 else "B"
            original_confidence = original_response.confidence
            original_raw_response = original_response.raw_response or "N/A"

            # 记录原始judge输入输出
            if logger:
                logger.logger.info(f"    🔍 [原始Judge] Input Prompt:")
                # 按行输出prompt，便于阅读
                for line in original_prompt.split('\n'):
                    logger.logger.info(f"        {line}")
                logger.logger.info(f"    🔍 [原始Judge] Raw Output:")
                # 按行输出response，便于阅读
                for line in original_raw_response.split('\n'):
                    logger.logger.info(f"        {line}")
                logger.logger.info(f"    🔍 [原始Judge] Parsed: {original_preference} (confidence: {original_confidence:.3f})")

            # 确定目标偏好
            target_preference = 1 - original_response.preference
            
            # 尝试所有可能的动作，选择最成功的
            best_success = False
            best_modified_a = example.response_a
            best_modified_b = example.response_b
            best_modified_instruction = None  # 添加修改后指令的跟踪
            best_action = "no_action"
            best_new_preference = original_preference
            best_new_confidence = original_confidence
            queries_used = 0
            attack_start_time = time.time()
            
            # 确定最大查询次数
            if max_queries is not None:
                max_actions = min(attack.get_action_space_size(), max_queries)
            else:
                max_actions = attack.get_action_space_size()
            
            for action in range(max_actions):
                try: 
                    modified_a, modified_b = attack.apply_action(example, action, target_preference)
                    queries_used += 1  # 每次尝试都算一次查询
                    
                    # 检查是否是FlipAttack、promptinjection（修改instruction的attack）
                    if hasattr(attack, 'get_modified_instruction'):
                        # FlipAttack: 使用修改后的instruction
                        modified_instruction = attack.get_modified_instruction(example)
                        # 创建临时example用于获取prompt
                        temp_example = PairwiseExample(
                            question_id=example.question_id,
                            instruction=modified_instruction,
                            response_a=example.response_a,
                            response_b=example.response_b,
                            model_a=example.model_a,
                            model_b=example.model_b,
                            metadata=example.metadata
                        )
                        new_prompt = judge.get_judge_prompt(temp_example)
                        new_response = judge.judge_pairwise(example, modified_instruction)
                    else:
                        # 其他attack: 创建修改后的样本
                        modified_instruction = None  # 其他攻击不修改instruction
                        modified_example = PairwiseExample(
                            question_id=example.question_id,
                            instruction=example.instruction,
                            response_a=modified_a,
                            response_b=modified_b,
                            model_a=example.model_a,
                            model_b=example.model_b,
                            metadata=example.metadata
                        )
                        new_prompt = judge.get_judge_prompt(modified_example)
                        new_response = judge.judge_pairwise(modified_example)
                    
                    new_preference = "A" if new_response.preference == 0 else "B"
                    new_confidence = new_response.confidence
                    new_raw_response = new_response.raw_response or "N/A"
                    
                    # 记录攻击后judge输入输出
                    if logger:
                        logger.logger.info(f"    🔍 [攻击Judge查询#{queries_used}] Action: {attack.get_action_description(action)}")
                        logger.logger.info(f"    🔍 [攻击Judge查询#{queries_used}] Input Prompt:")
                        # 按行输出prompt，便于阅读
                        for line in new_prompt.split('\n'):
                            logger.logger.info(f"        {line}")
                        logger.logger.info(f"    🔍 [攻击Judge查询#{queries_used}] Raw Output:")
                        # 按行输出response，便于阅读
                        for line in new_raw_response.split('\n'):
                            logger.logger.info(f"        {line}")
                        logger.logger.info(f"    🔍 [攻击Judge查询#{queries_used}] Parsed: {new_preference} (confidence: {new_confidence:.3f})")
                    
                    # 检查是否成功 - 使用攻击方法的成功判断逻辑
                    # 对于PositionAttack，需要特殊处理（因为交换了位置）
                    original_pref_int = 0 if original_preference == "A" else 1
                    new_pref_int = 0 if new_preference == "A" else 1
                    success = attack.is_attack_successful(original_pref_int, new_pref_int)
                    if success:
                        best_success = True
                        best_modified_a = modified_a
                        best_modified_b = modified_b
                        best_modified_instruction = modified_instruction  # 保存修改后的指令
                        best_action = attack.get_action_description(action)
                        best_new_preference = new_preference
                        best_new_confidence = new_confidence
                        break  # 找到成功的攻击就停止
                        
                except Exception as e:
                    queries_used += 1  # 即使失败也算一次查询
                    if logger:
                        logger.logger.warning(f"    ⚠️  [攻击Judge查询#{queries_used}] 查询失败: {e}")
                    continue
            
            if best_success:
                successful_attacks += 1
                total_queries_successful += queries_used  # Only count queries for successful attacks
            
            total_samples += 1
            confidence_change = best_new_confidence - original_confidence
            confidence_changes.append(confidence_change)
            
            # 计算效率指标
            attack_time = time.time() - attack_start_time
            efficiency_score = 1.0 / queries_used if best_success else 0.0
            
            # 更新统计
            total_queries += queries_used  # Count queries for all attacks
            total_time += attack_time
            efficiency_scores.append(efficiency_score)
            
            # 创建攻击样本记录
            sample = AttackSample(
                sample_id=sample_id,
                question_id=example.question_id,
                instruction=example.instruction,
                response_a=example.response_a,
                response_b=example.response_b,
                model_a=example.model_a,
                model_b=example.model_b,
                original_preference=original_preference,
                original_confidence=original_confidence,
                attack_method=attack_name,
                attack_action=best_action,
                modified_instruction=best_modified_instruction,  # 添加修改后的指令
                modified_response_a=best_modified_a,
                modified_response_b=best_modified_b,
                new_preference=best_new_preference,
                new_confidence=best_new_confidence,
                success=best_success,
                timestamp=time.time(),
                queries_used=queries_used,
                efficiency_score=efficiency_score,
                attack_time=attack_time,
                metadata=example.metadata
            )
            samples.append(sample)
            
            # 记录到日志
            if logger:
                logger.log_attack_sample(sample)
            
        except Exception as e:
            if logger:
                logger.logger.error(f"样本 {sample_id} 处理失败: {e}")
            continue
    
    # 计算平均置信度变化
    avg_confidence_change = sum(confidence_changes) / len(confidence_changes) if confidence_changes else 0.0
    
    # 计算效率统计
    avg_queries_used = total_queries / total_samples if total_samples > 0 else 0.0  # AQA: Average Queries per Attack (all attempts)
    avg_queries_successful = total_queries_successful / successful_attacks if successful_attacks > 0 else 0.0  # AQSA: Average Queries per Successful Attack
    avg_efficiency_score = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.0
    avg_attack_time = total_time / total_samples if total_samples > 0 else 0.0
    total_queries_saved = total_samples * attack.get_action_space_size() - total_queries
    
    # 创建攻击结果
    success_rate = successful_attacks / total_samples if total_samples > 0 else 0.0
    success_rate = round(success_rate, 4)  # 保留四位小数精度
    attack_results = AttackResults(
        attack_method=attack_name,
        total_samples=total_samples,
        successful_attacks=successful_attacks,
        success_rate=success_rate,
        action_space_size=attack.get_action_space_size(),
        avg_confidence_change=avg_confidence_change,
        avg_queries_used=avg_queries_used,  # AQA: Average Queries per Attack (all attempts)
        avg_queries_successful=avg_queries_successful,  # AQSA: Average Queries per Successful Attack
        avg_efficiency_score=avg_efficiency_score,
        avg_attack_time=avg_attack_time,
        total_queries_saved=total_queries_saved,
        samples=samples
    )
    
    return attack_results


def save_single_attack_result(attack_results: AttackResults, results_dir: str, benchmark: str, 
                               experiment_config: Dict[str, Any] = None):
    """立即保存单个攻击方法的结果文件"""
    baseline_file = os.path.join(results_dir, f"baseline_{attack_results.attack_method}_results.json")
    
    # 基础结果数据
    # 确保success_rate保留四位小数精度
    success_rate = round(attack_results.success_rate, 4)
    baseline_data = {
        "benchmark": benchmark,
        "attack_method": attack_results.attack_method,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "total_samples": attack_results.total_samples,
        "successful_attacks": attack_results.successful_attacks,
        "success_rate": success_rate,
        "action_space_size": attack_results.action_space_size,
        "avg_confidence_change": attack_results.avg_confidence_change,
        "avg_queries_used": attack_results.avg_queries_used,  # AQA: Average Queries per Attack (all attempts)
        "avg_queries_successful": attack_results.avg_queries_successful,  # AQSA: Average Queries per Successful Attack
        "avg_efficiency_score": attack_results.avg_efficiency_score,
        "avg_attack_time": attack_results.avg_attack_time,
        "total_queries_saved": attack_results.total_queries_saved
    }
    
    # 添加实验配置信息（超参数）
    if experiment_config:
        baseline_data["experiment_config"] = experiment_config
    else:
        # 如果没有传入配置，尝试从环境变量和默认值获取
        baseline_data["experiment_config"] = {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
            "max_samples": "unknown",
            "judge_model_path": "unknown",
            "judge_type": "unknown",
            "random_seed": "unknown",
            "baseline_max_queries": "unknown"
        }
    
    with open(baseline_file, 'w') as f:
        json.dump(baseline_data, f, indent=2)
    
    print(f"💾 {attack_results.attack_method} 结果已立即保存到: {baseline_file}")
    # 同时记录到日志中
    import logging
    logger = logging.getLogger('hydra_attack')
    logger.info(f"💾 {attack_results.attack_method} 结果已立即保存到: {baseline_file}")


"""
def test_rl_combined_attacks(examples: List[PairwiseExample], judge, attacks: Dict[str, Any], 
                            max_samples = 10, logger: HydraLogger = None, max_queries: int = None) -> AttackResults:
    '''测试强化学习组合攻击'''
    # 处理max_samples参数，支持"full"字符串
    if isinstance(max_samples, str) and max_samples.lower() == "full":
        actual_max_samples = len(examples)
    else:
        actual_max_samples = min(len(examples), int(max_samples))
    
    if logger:
        logger.log_benchmark_start("强化学习组合攻击", actual_max_samples)
    
    # 收集所有攻击方法实例
    all_attack_instances = []
    for attack_name, attack in attacks.items():
        if isinstance(attack, dict):
            for mode, attack_instance in attack.items():
                all_attack_instances.append(attack_instance)
        else:
            all_attack_instances.append(attack)
    
    successful_attacks = 0
    total_samples = 0
    samples = []
    confidence_changes = []
    total_action_space = sum(attack.get_action_space_size() for attack in all_attack_instances)
    total_queries = 0
    total_time = 0.0
    efficiency_scores = []
    
    # 添加进度条
    sample_limit = actual_max_samples
    
    progress_bar = tqdm(
        enumerate(examples[:sample_limit]), 
        total=sample_limit,
        desc="测试 RL组合攻击",
        unit="样本",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for i, example in progress_bar:
        sample_id = f"rl_combined_sample_{i+1}"
        
        try:
            # 获取原始偏好
            original_response = judge.judge_pairwise(example)
            original_preference = "A" if original_response.preference == 0 else "B"
            original_confidence = original_response.confidence
            
            # 尝试所有攻击方法，选择最成功的
            best_success = False
            best_attack_name = "no_attack"
            best_modified_a = example.response_a
            best_modified_b = example.response_b
            best_modified_instruction = None  # 添加修改后指令的跟踪
            best_action = "no_action"
            best_new_preference = original_preference
            best_new_confidence = original_confidence
            queries_used = 0
            attack_start_time = time.time()
            
            for attack in all_attack_instances:
                try:
                    # 检查是否已达到查询次数限制
                    if max_queries is not None and queries_used >= max_queries:
                        break
                    
                    # 尝试所有可能的动作，但限制查询次数
                    if max_queries is not None:
                        max_actions = min(attack.get_action_space_size(), max_queries - queries_used)
                    else:
                        max_actions = attack.get_action_space_size()
                    
                    for action in range(max_actions):
                        modified_a, modified_b = attack.apply_action(example, action)
                        queries_used += 1  # 每次尝试都算一次查询
                        
                        # 检查是否是FlipAttack（修改instruction的attack）
                        if hasattr(attack, 'get_modified_instruction'):
                            # FlipAttack: 使用修改后的instruction
                            modified_instruction = attack.get_modified_instruction(example)
                            new_response = judge.judge_pairwise(example, modified_instruction)
                        else:
                            # 其他attack: 创建修改后的样本
                            modified_instruction = None  # 其他攻击不修改instruction
                            modified_example = PairwiseExample(
                                question_id=example.question_id,
                                instruction=example.instruction,
                                response_a=modified_a,
                                response_b=modified_b,
                                model_a=example.model_a,
                                model_b=example.model_b,
                                metadata=example.metadata
                            )
                            new_response = judge.judge_pairwise(modified_example)
                        
                        new_preference = "A" if new_response.preference == 0 else "B"
                        new_confidence = new_response.confidence
                        
                        # 检查是否成功 - 使用攻击方法的成功判断逻辑
                        # 对于PositionAttack，需要特殊处理（因为交换了位置）
                        original_pref_int = 0 if original_preference == "A" else 1
                        new_pref_int = 0 if new_preference == "A" else 1
                        if attack.is_attack_successful(original_pref_int, new_pref_int):
                            best_success = True
                            best_attack_name = f"{attack.attack_type.value}_{action}"
                            best_modified_a = modified_a
                            best_modified_b = modified_b
                            best_modified_instruction = modified_instruction  # 保存修改后的指令
                            best_action = attack.get_action_description(action)
                            best_new_preference = new_preference
                            best_new_confidence = new_confidence
                            break
                    
                    if best_success:
                        break
                        
                except Exception as e:
                    continue
            
            if best_success:
                successful_attacks += 1
            
            total_samples += 1
            confidence_change = best_new_confidence - original_confidence
            confidence_changes.append(confidence_change)
            
            # 计算效率指标
            attack_time = time.time() - attack_start_time
            efficiency_score = 1.0 / queries_used if best_success else 0.0
            
            # 更新统计
            total_queries += queries_used
            total_time += attack_time
            efficiency_scores.append(efficiency_score)
            
            # 创建攻击样本记录
            sample = AttackSample(
                sample_id=sample_id,
                question_id=example.question_id,
                instruction=example.instruction,
                response_a=example.response_a,
                response_b=example.response_b,
                model_a=example.model_a,
                model_b=example.model_b,
                original_preference=original_preference,
                original_confidence=original_confidence,
                attack_method=best_attack_name,
                attack_action=best_action,
                modified_instruction=best_modified_instruction,  # 添加修改后的指令
                modified_response_a=best_modified_a,
                modified_response_b=best_modified_b,
                new_preference=best_new_preference,
                new_confidence=best_new_confidence,
                success=best_success,
                timestamp=time.time(),
                queries_used=queries_used,
                efficiency_score=efficiency_score,
                attack_time=attack_time,
                metadata=example.metadata
            )
            samples.append(sample)
            
            # 记录到日志
            if logger:
                logger.log_attack_sample(sample)
            
            # 添加延迟
            time.sleep(0.2)
            
        except Exception as e:
            if logger:
                logger.logger.error(f"样本 {sample_id} 处理失败: {e}")
            continue
    
    # 计算平均置信度变化
    avg_confidence_change = sum(confidence_changes) / len(confidence_changes) if confidence_changes else 0.0
    
    # 计算效率统计
    avg_queries_used = total_queries / total_samples if total_samples > 0 else 0.0
    avg_efficiency_score = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.0
    avg_attack_time = total_time / total_samples if total_samples > 0 else 0.0
    total_queries_saved = total_samples * total_action_space - total_queries
    
    # 创建攻击结果
    rl_results = AttackResults(
        attack_method="RL_Combined",
        total_samples=total_samples,
        successful_attacks=successful_attacks,
        success_rate=successful_attacks / total_samples if total_samples > 0 else 0.0,
        action_space_size=total_action_space,
        avg_confidence_change=avg_confidence_change,
        avg_queries_used=avg_queries_used,
        avg_efficiency_score=avg_efficiency_score,
        avg_attack_time=avg_attack_time,
        total_queries_saved=total_queries_saved,
        samples=samples
    )
    
    if logger:
        logger.log_rl_results(rl_results)
    
    return rl_results
"""

def main():
    """主函数"""
    # 记录开始时间
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Hydra-Attack 主测试脚本")
    parser.add_argument("--benchmarks", nargs="+", required=True, help="要测试的benchmark列表")
    parser.add_argument("--max_samples", default=80, help="每个benchmark的测试样本数，使用 'full' 表示使用全量数据")
    parser.add_argument("--attack_methods", nargs="+", default=["all"], help="攻击方法列表")
    parser.add_argument("--judge_model_path", type=str, required=True, help="Judge模型路径")
    parser.add_argument("--judge_type", type=str, default="qwen", help="Judge类型")
    parser.add_argument("--results_dir", type=str, default="./results", help="结果输出目录")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志输出目录")
    parser.add_argument("--save_detailed_results", action="store_true", help="保存详细结果")
    parser.add_argument("--max_retries", type=int, default=3, help="最大重试次数")
    parser.add_argument("--timeout", type=int, default=300, help="超时时间（秒）")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    parser.add_argument("--baseline_max_queries", type=int, default=5, help="baseline方法的最大查询次数")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.random_seed)
    
    # 初始化日志系统
    logger = HydraLogger(log_dir=args.log_dir, results_dir=args.results_dir)
    
    # 创建实验配置
    config = ExperimentConfig(
        gpu_config={
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "device_count": len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
        },
        data_config={
            "benchmarks": args.benchmarks,
            "max_samples": args.max_samples,
            "random_seed": args.random_seed
        },
        attack_config={
            "attack_methods": args.attack_methods,
        },
        judge_config={
            "judge_type": args.judge_type,
            "model_path": args.judge_model_path
        },
        rl_config={
            "enabled": False,
            "max_queries": args.baseline_max_queries,
            "learning_rate": 0.01,
            "exploration_rate": 0.1
        },
        output_config={
            "results_dir": args.results_dir,
            "log_dir": args.log_dir,
            "save_detailed_results": args.save_detailed_results
        },
        timestamp=time.time(),
        random_seed=args.random_seed
    )
    
    # 记录配置
    logger.log_config(config)
    
    # 初始化Judge
    logger.logger.info("初始化Judge...")
    try:
        if args.judge_type == "qwen":
            judge = create_qwen_judge(args.judge_model_path)
        elif args.judge_type == "llama":
            judge = create_llama_judge(args.judge_model_path)
        elif args.judge_type == "glm":
            judge = create_glm_judge(args.judge_model_path)   
        elif args.judge_type == "mistral":
            judge = create_mistral_judge(args.judge_model_path)
        elif args.judge_type == "gemma":
            # Check if it's 1B model (text-only)
            model_path_lower = args.judge_model_path.lower()
            if "1b" in model_path_lower or "gemma-3-1b" in model_path_lower:
                judge = create_gemma_judge_1b(args.judge_model_path)
            else:
                judge = create_gemma_judge(args.judge_model_path)
        else:
            raise ValueError(f"不支持的Judge类型: {args.judge_type}")
        logger.logger.info("✅ Judge初始化成功")
    except Exception as e:
        logger.logger.error(f"❌ Judge初始化失败: {e}")
        return
    
    # 获取攻击方法
    attacks = get_attack_methods(args.attack_methods)
    logger.logger.info(f"✅ 加载了 {len(attacks)} 种攻击方法")
    
    # 估算总时间
    total_benchmarks = len(args.benchmarks)
    total_attacks = sum(len(attack) if isinstance(attack, dict) else 1 for attack in attacks.values())
    
    if args.max_samples == "full":
        # 对于全量数据，使用平均样本数估算
        avg_samples_per_benchmark = 1000  # 估算平均值
        total_samples = avg_samples_per_benchmark * total_benchmarks * total_attacks
    else:
        total_samples = args.max_samples * total_benchmarks * total_attacks
    
    estimated_time_per_sample = 1  # 假设每个样本需要1秒
    estimated_total_time = total_samples * estimated_time_per_sample
    
    hours = int(estimated_total_time // 3600)
    minutes = int((estimated_total_time % 3600) // 60)
    
    logger.logger.info(f"📊 预估统计:")
    logger.logger.info(f"   - Benchmark数量: {total_benchmarks}")
    logger.logger.info(f"   - 攻击方法数量: {total_attacks}")
    logger.logger.info(f"   - 总样本数: {total_samples}")
    logger.logger.info(f"   - 预估总时间: {hours:02d}:{minutes:02d}:00")
    logger.logger.info("=" * 60)
    
    # 测试每个benchmark
    # 添加总体进度条
    benchmark_progress = tqdm(
        args.benchmarks, 
        desc="处理Benchmark",
        unit="benchmark",
        ncols=100,
        position=0
    )
    
    for benchmark in benchmark_progress:
        benchmark_progress.set_description(f"处理 {benchmark}")
        logger.log_benchmark_start(benchmark, args.max_samples)
        
        # 加载数据
        examples = load_benchmark_data(benchmark, args.max_samples, args.random_seed)
        if not examples:
            logger.logger.error(f"❌ 无法加载 {benchmark} 数据")
            continue
        
        logger.logger.info(f"✅ 加载了 {len(examples)} 个样本")
        
        # 尝试从数据分割信息文件中读取train_ratio和test_ratio
        train_ratio = None
        test_ratio = None
        split_info_file = f"data/split/{benchmark}_split_info.json"
        if os.path.exists(split_info_file):
            try:
                with open(split_info_file, 'r') as f:
                    split_info = json.load(f)
                    train_ratio = split_info.get("train_ratio")
                    test_ratio = split_info.get("test_ratio")
            except Exception as e:
                logger.logger.warning(f"⚠️  无法读取数据分割信息: {e}")
        
        # 构建实验配置信息
        experiment_config = {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
            "max_samples": args.max_samples,
            "train_ratio": train_ratio,
            "test_ratio": test_ratio,
            "judge_model_path": args.judge_model_path,
            "judge_type": args.judge_type,
            "random_seed": args.random_seed,
            "baseline_max_queries": args.baseline_max_queries,
            "attack_methods": args.attack_methods,
            "benchmark": benchmark,
            "total_samples": len(examples)
        }
        
        # 测试攻击方法
        baseline_results = test_attacks(examples, judge, attacks, args.max_samples, logger, 
                                       args.baseline_max_queries, args.results_dir, benchmark,
                                       experiment_config)
        
        # 计算benchmark结果
        total_successful_attacks = sum(ar.successful_attacks for ar in baseline_results.values()) if baseline_results else 0
        total_attack_attempts = sum(ar.total_samples for ar in baseline_results.values()) if baseline_results else 0
        overall_success_rate = total_successful_attacks / total_attack_attempts if total_attack_attempts > 0 else 0.0
        overall_success_rate = round(overall_success_rate, 4)  # 保留四位小数精度
        
        # 找到最佳和最差攻击方法
        best_attack_method = ""
        worst_attack_method = ""
        if baseline_results:
            sorted_results = sorted(baseline_results.items(), key=lambda x: x[1].success_rate, reverse=True)
            best_attack_method = sorted_results[0][0] if sorted_results else ""
            worst_attack_method = sorted_results[-1][0] if sorted_results else ""
        
        # 创建benchmark结果
        benchmark_results = BenchmarkResults(
            benchmark_name=benchmark,
            total_samples=len(examples),
            baseline_results=baseline_results,
            rl_results=None,  # 明确设置为None
            overall_success_rate=overall_success_rate,
            best_attack_method=best_attack_method,
            worst_attack_method=worst_attack_method
        )
        
        # 记录benchmark结果
        logger.log_benchmark_results(benchmark_results)
        
        # 注意：每个攻击方法的结果文件已经在测试完成后立即保存，无需重复保存
        
        # 保存实验配置
        config_file = os.path.join(args.results_dir, "experiment_config.json")
        config_data = {
            "benchmark": benchmark,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "max_samples": args.max_samples,
            "max_queries": args.baseline_max_queries,
            "judge_model_path": args.judge_model_path,
            "judge_type": args.judge_type,
            "random_seed": args.random_seed,
            "attack_methods": list(baseline_results.keys()),
            "total_baseline_methods": len(baseline_results)
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.logger.info(f"💾 实验配置已保存到: {config_file}")
    
    # 记录实验总结
    logger.log_experiment_summary()
    
    # 计算总时间
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    logger.logger.info("🎉 所有测试完成！")
    logger.logger.info(f"⏱️  总耗时: {hours:02d}:{minutes:02d}:{seconds:02d}")
    logger.logger.info(f"📁 日志文件: {logger.get_log_file_path()}")
    logger.logger.info(f"📊 结果文件: {logger.get_results_file_path()}")
    logger.logger.info(f"📝 样本详情: {logger.get_samples_file_path()}")


if __name__ == "__main__":
    main()