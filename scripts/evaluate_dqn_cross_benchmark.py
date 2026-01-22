#!/usr/bin/env python3
"""
DQN跨数据集泛化评估脚本
加载在一个benchmark上训练的DQN模型，在另外两个benchmark的测试集上进行测试
"""

import argparse
import sys
import json
import time
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# 添加src到路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_types import PairwiseExample
from attacks import (
    FlipAttackFWO, FlipAttackFCW, FlipAttackFCS, UncertaintyAttack, PositionAttack, DistractorAttack,
    PromptInjectionAttack, MarkerInjectionAttack, FormattingAttack,
    AuthorityAttack, UnicodeAttack, CoTPoisoningAttack, EmojiAttack
)
from evaluation.qwen_judge import create_qwen_judge
from evaluation.llama_judge import create_llama_judge
from rl.enhanced_environment import EnhancedHydraAttackEnv
from rl.agent import DQNAgent
from utils.logger import HydraLogger


def convert_numpy_types(obj):
    """转换numpy类型为Python原生类型，用于JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def load_benchmark_data(benchmark: str, max_samples = 100, random_seed: int = 42) -> List[PairwiseExample]:
    """加载benchmark测试数据"""
    test_file = f"data/split/{benchmark}_test.json"
    
    if not os.path.exists(test_file):
        print(f"❌ 测试数据不存在: {test_file}")
        raise FileNotFoundError(f"测试数据不存在: {test_file}")
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    print(f"✅ 加载测试数据: {len(data)} 样本")
    
    # 限制样本数量
    if max_samples != "full" and max_samples is not None:
        try:
            max_samples_int = int(max_samples)
            if len(data) > max_samples_int:
                data = data[:max_samples_int]
                print(f"✅ 限制样本数量为: {max_samples_int}")
        except (ValueError, TypeError):
            # 如果max_samples无法转换为整数，忽略限制
            pass
    
    examples = []
    for sample in data:
        example = PairwiseExample(
            question_id=sample["question_id"],
            instruction=sample["instruction"],
            response_a=sample["response_a"],
            response_b=sample["response_b"],
            model_a=sample["model_a"],
            model_b=sample["model_b"],
            metadata=sample.get("metadata", {})
        )
        examples.append(example)
    
    return examples


def get_attack_methods() -> List[Any]:
    """获取所有攻击方法实例"""
    attacks = [
        FlipAttackFCS(),
        FlipAttackFWO(),
        FlipAttackFCW(),
        UncertaintyAttack(),
        PositionAttack(),
        DistractorAttack(),
        PromptInjectionAttack(),
        MarkerInjectionAttack(),
        FormattingAttack(),
        AuthorityAttack(),
        UnicodeAttack(),
        CoTPoisoningAttack(),
        EmojiAttack(),
    ]
    return attacks


def load_dqn_agent(model_path: str, config_path: str, logger: HydraLogger = None) -> DQNAgent:
    """加载训练好的DQN智能体"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 获取DQN参数
    dqn_params = config.get('dqn_params', {})
    env_params = config.get('env_params', {})
    
    # 创建攻击方法以获取动作空间大小
    attacks = get_attack_methods()
    total_actions = sum(attack.get_action_space_size() for attack in attacks)
    
    # 从保存的模型中推断状态维度
    import torch
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 从Q网络的第一个线性层的权重形状推断状态维度
    if 'q_network_state_dict' in checkpoint:
        state_dict = checkpoint['q_network_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 找到第一个线性层的权重（通常是'0.weight'）
    first_layer_weight = None
    for key, value in state_dict.items():
        if 'weight' in key and len(value.shape) == 2 and key.startswith('0.'):
            first_layer_weight = value
            break
    
    if first_layer_weight is not None:
        state_dim = first_layer_weight.shape[1]  # 输入维度
    else:
        state_dim = 512  # 默认值
    
    if logger:
        logger.logger.info(f"   推断的状态维度: {state_dim}")
    
    # 创建DQN智能体（使用推断的状态维度）
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=total_actions,
        hidden_dim=dqn_params.get('hidden_dim', 512),
        learning_rate=dqn_params.get('learning_rate', 0.0001),
        gamma=dqn_params.get('gamma', 0.99),
        epsilon=0.0,  # 评估时使用贪婪策略
        epsilon_min=0.0,
        epsilon_decay=1.0,
        batch_size=dqn_params.get('batch_size', 32),
        memory_size=dqn_params.get('memory_size', 50000),
        target_update_freq=dqn_params.get('target_update_freq', 500)
    )
    
    # 加载模型权重
    agent.load_model(model_path)
    
    if logger:
        logger.logger.info(f"✅ 成功加载DQN模型: {model_path}")
        logger.logger.info(f"   状态维度: {state_dim}")
        logger.logger.info(f"   动作空间大小: {total_actions}")
        logger.logger.info(f"   隐藏层维度: {dqn_params.get('hidden_dim', 512)}")
        logger.logger.info(f"   学习率: {dqn_params.get('learning_rate', 0.0001)}")
    
    return agent


def evaluate_cross_benchmark_performance(agent, test_examples: List[PairwiseExample], 
                                       attacks: List[Any], judge, max_queries: int = 10,
                                       success_reward: float = 20.0, query_penalty: float = 0.5,
                                       diversity_bonus: float = 1.0, efficiency_bonus: float = 2.0,
                                       confidence_threshold: float = 0.7,
                                       logger: HydraLogger = None) -> Dict[str, float]:
    """评估跨数据集泛化性能"""
    
    if logger:
        logger.logger.info("🔍 开始跨数据集泛化评估...")
        logger.logger.info(f"   目标Judge: {judge.__class__.__name__}")
        logger.logger.info(f"   测试样本数: {len(test_examples)}")
    
    success_count = 0
    total_queries = 0
    total_reward = 0.0
    detailed_results = []
    
    for i, example in enumerate(test_examples):
        if logger and i % 10 == 0:
            logger.logger.info(f"  评估进度: {i+1}/{len(test_examples)}")
        
        # 1. 先获取原始偏好（用于评估）
        try:
            original_response = judge.judge_pairwise(example)
            original_preference = original_response.preference
            original_confidence = original_response.confidence
        except Exception as e:
            if logger:
                logger.logger.warning(f"获取原始偏好失败: {e}")
            continue  # 跳过失败的样本
        
        # 2. 创建环境
        env = EnhancedHydraAttackEnv(
            examples=[example],
            attacks=attacks,
            judge=judge,
            max_queries=max_queries,
            success_reward=success_reward,
            query_penalty=query_penalty,
            diversity_bonus=diversity_bonus,
            efficiency_bonus=efficiency_bonus,
            confidence_threshold=confidence_threshold
        )
        
        # 3. 重置环境（评估模式：使用特定样本和预先获取的偏好）
        state, _ = env.reset(options={
            'use_specific_sample': True,
            'sample_idx': 0,
            'original_preference': original_preference,
            'original_confidence': original_confidence
        })
        episode_reward = 0
        queries_used = 0
        attack_sequence = []
        
        # 4. 执行攻击序列
        for step in range(max_queries):
            # 获取action mask（禁止重复使用失败的action）
            action_mask = env.get_action_mask()
            
            # 使用训练好的智能体选择动作（贪婪策略）
            action = agent.select_action(state, training=False, action_mask=action_mask)
            next_state, reward, done, truncated, info = env.step(action)
            
            # 记录攻击信息
            attack_info = {
                'step': step + 1,
                'action': int(action),
                'reward': float(reward),
                'done': done,
                'truncated': truncated
            }
            attack_sequence.append(attack_info)
            
            episode_reward += reward
            queries_used += 1
            
            if done or truncated:
                break
            
            state = next_state
        
        # 5. 检查攻击结果
        original_preference = env.original_preference
        final_preference = info.get('current_preference', original_preference)
        # 使用环境返回的success信息，它已经考虑了PositionAttack的特殊逻辑
        success = info.get('success', False)
        
        if success:
            success_count += 1
        
        total_queries += queries_used
        total_reward += episode_reward
        
        # 记录详细结果
        sample_result = {
            'question_id': example.question_id,
            'success': success,
            'queries_used': queries_used,
            'episode_reward': episode_reward,
            'original_preference': original_preference,
            'final_preference': final_preference,
            'original_confidence': original_confidence,
            'final_confidence': info.get('current_confidence', original_confidence),
            'attack_sequence': attack_sequence
        }
        detailed_results.append(sample_result)
    
    # 计算评估指标
    success_rate = success_count / len(test_examples) if test_examples else 0.0
    avg_queries = total_queries / len(test_examples) if test_examples else 0.0
    avg_reward = total_reward / len(test_examples) if test_examples else 0.0
    
    results = {
        'success_rate': success_rate,
        'avg_queries': avg_queries,
        'avg_reward': avg_reward,
        'total_samples': len(test_examples),
        'successful_attacks': success_count,
        'detailed_results': detailed_results
    }
    
    if logger:
        logger.logger.info(f"📊 跨数据集泛化评估结果:")
        logger.logger.info(f"  成功率: {success_rate:.3f}")
        logger.logger.info(f"  平均查询次数: {avg_queries:.3f}")
        logger.logger.info(f"  平均奖励: {avg_reward:.3f}")
        logger.logger.info(f"  成功攻击数: {success_count}/{len(test_examples)}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DQN跨数据集泛化评估")
    parser.add_argument("--source_benchmark", type=str, required=True,
                       choices=["alpaca_eval", "arena_hard", "code_judge_bench"],
                       help="训练DQN模型的源benchmark")
    parser.add_argument("--target_benchmark", type=str, required=True,
                       choices=["alpaca_eval", "arena_hard", "code_judge_bench"],
                       help="测试的目标benchmark")
    parser.add_argument("--max_samples", default=100, help="测试样本数，使用 'full' 表示使用全量数据")
    parser.add_argument("--judge_model_path", type=str, required=True, help="Judge模型路径")
    parser.add_argument("--judge_type", type=str, default="qwen", help="Judge类型")
    parser.add_argument("--dqn_model_path", type=str, required=True, help="DQN模型路径")
    parser.add_argument("--dqn_config_path", type=str, required=True, help="DQN配置文件路径")
    parser.add_argument("--max_queries", type=int, default=10, help="最大查询次数")
    parser.add_argument("--output_dir", type=str, default="./cross_benchmark_results", help="结果输出目录")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    
    # 环境奖励参数
    parser.add_argument("--success_reward", type=float, default=20.0, help="成功攻击奖励")
    parser.add_argument("--query_penalty", type=float, default=0.5, help="查询惩罚")
    parser.add_argument("--diversity_bonus", type=float, default=1.0, help="多样性奖励")
    parser.add_argument("--efficiency_bonus", type=float, default=2.0, help="效率奖励")
    parser.add_argument("--confidence_threshold", type=float, default=0.7, help="置信度阈值")
    
    args = parser.parse_args()
    
    # 检查源benchmark和目标benchmark不能相同
    if args.source_benchmark == args.target_benchmark:
        print("❌ 源benchmark和目标benchmark不能相同")
        return
    
    # 设置随机种子
    import random
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化日志系统
    logger = HydraLogger(log_dir=args.output_dir, results_dir=args.output_dir)
    
    logger.logger.info("🚀 开始DQN跨数据集泛化评估")
    logger.logger.info(f"源benchmark: {args.source_benchmark}")
    logger.logger.info(f"目标benchmark: {args.target_benchmark}")
    logger.logger.info(f"Judge模型: {args.judge_model_path}")
    logger.logger.info(f"Judge类型: {args.judge_type}")
    logger.logger.info(f"DQN模型: {args.dqn_model_path}")
    logger.logger.info(f"测试样本数: {args.max_samples}")
    logger.logger.info(f"最大查询次数: {args.max_queries}")
    
    # 加载测试数据
    test_examples = load_benchmark_data(args.target_benchmark, args.max_samples, args.random_seed)
    if not test_examples:
        logger.logger.error("❌ 无法加载测试数据")
        return
    
    logger.logger.info(f"✅ 加载了 {len(test_examples)} 个测试样本")
    
    # 创建攻击方法
    attacks = get_attack_methods()
    logger.logger.info(f"✅ 加载了 {len(attacks)} 种攻击方法")
    
    # 创建Judge（目标模型）
    try:
        if args.judge_type == "qwen":
            judge = create_qwen_judge(args.judge_model_path)
        elif args.judge_type == "llama":
            judge = create_llama_judge(args.judge_model_path)
        else:
            raise ValueError(f"不支持的Judge类型: {args.judge_type}")
        logger.logger.info("✅ Judge初始化成功")
    except Exception as e:
        logger.logger.error(f"❌ Judge初始化失败: {e}")
        return
    
    # 加载DQN智能体
    try:
        agent = load_dqn_agent(args.dqn_model_path, args.dqn_config_path, logger)
        logger.logger.info("✅ DQN智能体加载成功")
    except Exception as e:
        logger.logger.error(f"❌ DQN智能体加载失败: {e}")
        return
    
    # 开始评估
    logger.logger.info("🎯 开始跨数据集泛化评估...")
    start_time = time.time()
    
    results = evaluate_cross_benchmark_performance(
        agent, test_examples, attacks, judge, args.max_queries,
        args.success_reward, args.query_penalty, args.diversity_bonus,
        args.efficiency_bonus, args.confidence_threshold, logger
    )
    
    evaluation_time = time.time() - start_time
    logger.logger.info(f"🎉 评估完成！")
    logger.logger.info(f"⏱️  评估耗时: {evaluation_time:.2f}秒")
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"cross_benchmark_evaluation_{args.source_benchmark}_to_{args.target_benchmark}_{timestamp}.json")
    
    # 准备保存的结果（不包含详细结果以节省空间）
    save_results = {
        'source_benchmark': args.source_benchmark,
        'target_benchmark': args.target_benchmark,
        'judge_model_path': args.judge_model_path,
        'judge_type': args.judge_type,
        'dqn_model_path': args.dqn_model_path,
        'dqn_config_path': args.dqn_config_path,
        'max_samples': args.max_samples,
        'max_queries': args.max_queries,
        'evaluation_time': evaluation_time,
        'success_rate': results['success_rate'],
        'avg_queries': results['avg_queries'],
        'avg_reward': results['avg_reward'],
        'total_samples': results['total_samples'],
        'successful_attacks': results['successful_attacks'],
        'env_params': {
            'success_reward': args.success_reward,
            'query_penalty': args.query_penalty,
            'diversity_bonus': args.diversity_bonus,
            'efficiency_bonus': args.efficiency_bonus,
            'confidence_threshold': args.confidence_threshold
        }
    }
    
    # 转换numpy类型
    save_results = convert_numpy_types(save_results)
    
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    logger.logger.info(f"💾 结果已保存到: {results_file}")
    
    # 输出最终结果
    logger.logger.info("=" * 80)
    logger.logger.info("📊 跨数据集泛化评估结果")
    logger.logger.info("=" * 80)
    logger.logger.info(f"源benchmark: {args.source_benchmark}")
    logger.logger.info(f"目标benchmark: {args.target_benchmark}")
    logger.logger.info(f"Judge模型: {args.judge_type}")
    logger.logger.info(f"成功率: {results['success_rate']:.3f}")
    logger.logger.info(f"平均查询次数: {results['avg_queries']:.3f}")
    logger.logger.info(f"平均奖励: {results['avg_reward']:.3f}")
    logger.logger.info(f"成功攻击数: {results['successful_attacks']}/{results['total_samples']}")
    logger.logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
