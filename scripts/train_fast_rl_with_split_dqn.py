#!/usr/bin/env python3
"""
改进的快速RL训练脚本 - 支持数据分割和未见数据评估
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

from data_types import PairwiseExample, RLConfig
from attacks import (
    FlipAttackFWO, FlipAttackFCW, FlipAttackFCS, UncertaintyAttack, PositionAttack, DistractorAttack,
    PromptInjectionAttack, MarkerInjectionAttack, FormattingAttack,
    AuthorityAttack, UnicodeAttack, CoTPoisoningAttack, EmojiAttack
)
from evaluation.qwen_judge import create_qwen_judge
from evaluation.llama_judge import create_llama_judge
from rl.enhanced_environment import EnhancedHydraAttackEnv
from rl.agent import DQNAgent
from rl.trainer import RLTrainer, evaluate_on_unseen_data
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


def load_benchmark_data(benchmark: str, max_samples = 100, random_seed: int = 42, validation_ratio: float = 0.1) -> tuple[List[PairwiseExample], List[PairwiseExample]]:
    """加载benchmark数据并分出验证集（必须使用预分割数据）"""
    # 必须使用预分割的训练数据
    train_file = f"data/split/{benchmark}_train.json"
    
    if not os.path.exists(train_file):
        print(f"❌ 预分割的训练数据不存在: {train_file}")
        print("请先运行 prepare_data.py 脚本准备数据分割")
        raise FileNotFoundError(f"预分割的训练数据不存在: {train_file}")
    
    # 使用预分割的训练数据
    with open(train_file, 'r') as f:
        data = json.load(f)
    
    print(f"✅ 使用预分割的训练数据: {len(data)} 样本")
    
    # 限制样本数量
    if max_samples != "full" and max_samples is not None:
        try:
            max_samples_int = int(max_samples)
            if len(data) > max_samples_int:
                data = data[:max_samples_int]
        except (ValueError, TypeError):
            # 如果max_samples无法转换为整数，忽略限制
            pass
    
    # 转换为PairwiseExample对象
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
    
    # 从训练集中分出验证集
    if validation_ratio > 0 and len(examples) > 1:
        np.random.seed(random_seed)
        n_total = len(examples)
        n_validation = int(n_total * validation_ratio)
        
        # 随机打乱数据
        indices = np.random.permutation(n_total)
        
        # 分出验证集和训练集
        validation_indices = indices[:n_validation]
        train_indices = indices[n_validation:]
        
        train_examples = [examples[i] for i in train_indices]
        validation_examples = [examples[i] for i in validation_indices]
        
        print(f"✅ 数据分割: 训练集 {len(train_examples)} 样本, 验证集 {len(validation_examples)} 样本")
        
        return train_examples, validation_examples
    else:
        print(f"✅ 使用全部数据作为训练集: {len(examples)} 样本")
        return examples, []




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



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练快速RL攻击智能体（支持数据分割）")
    parser.add_argument("--benchmark", type=str, default="alpaca_eval", 
                       choices=["alpaca_eval", "arena_hard", "code_judge_bench"],
                       help="使用的benchmark")
    parser.add_argument("--max_samples", default=100, help="总样本数，使用 'full' 表示使用全量数据")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--validation_ratio", type=float, default=0.1, help="验证集比例（从训练集中分出）")
    parser.add_argument("--judge_model_path", type=str, required=True, help="Judge模型路径")
    parser.add_argument("--judge_type", type=str, default="qwen", help="Judge类型")
    parser.add_argument("--episodes", type=int, default=1000, help="训练轮数")
    parser.add_argument("--max_queries", type=int, default=10, help="最大查询次数")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--model_dir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    parser.add_argument("--eval_freq", type=int, default=500, help="验证频率（每多少个episode评估一次）")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001, help="Early stopping最小改善阈值")
    
    # DQN算法特有参数
    parser.add_argument("--hidden_dim", type=int, default=512, help="神经网络隐藏层维度")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--epsilon", type=float, default=1.0, help="初始探索率")
    parser.add_argument("--epsilon_min", type=float, default=0.01, help="最小探索率")
    parser.add_argument("--epsilon_decay", type=float, default=0.9995, help="探索率衰减因子")
    parser.add_argument("--memory_size", type=int, default=50000, help="经验回放缓冲区大小")
    parser.add_argument("--target_update_freq", type=int, default=500, help="目标网络更新频率")
    
    # 环境奖励参数
    parser.add_argument("--success_reward", type=float, default=20.0, help="成功攻击奖励")
    parser.add_argument("--query_penalty", type=float, default=0.5, help="查询惩罚")
    parser.add_argument("--diversity_bonus", type=float, default=1.0, help="多样性奖励")
    parser.add_argument("--efficiency_bonus", type=float, default=2.0, help="效率奖励")
    parser.add_argument("--confidence_threshold", type=float, default=0.7, help="置信度阈值")
    
    args = parser.parse_args()
    
    # 设置随机种子
    import random
    import numpy as np
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # 创建模型目录和日志目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 初始化日志系统
    logger = HydraLogger(log_dir=args.log_dir, results_dir=args.model_dir)
    
    logger.logger.info("🚀 开始训练快速RL攻击智能体（支持数据分割）")
    logger.logger.info(f"Benchmark: {args.benchmark}")
    logger.logger.info(f"实验目录: {args.model_dir}")
    logger.logger.info(f"日志目录: {args.log_dir}")
    logger.logger.info(f"总样本数: {args.max_samples}")
    logger.logger.info(f"数据分割: 训练{args.train_ratio:.1%} / 测试{args.test_ratio:.1%} / 验证{args.validation_ratio:.1%}")
    logger.logger.info(f"训练轮数: {args.episodes}")
    logger.logger.info(f"最大查询次数: {args.max_queries}")
    
    # 加载训练数据（优先使用预分割数据）
    train_examples, validation_examples = load_benchmark_data(args.benchmark, args.max_samples, args.random_seed, args.validation_ratio)
    if not train_examples:
        logger.logger.error("❌ 无法加载训练数据")
        return
    
    logger.logger.info(f"✅ 加载了 {len(train_examples)} 个训练样本")
    if validation_examples:
        logger.logger.info(f"✅ 加载了 {len(validation_examples)} 个验证样本")
    
    # 加载测试数据（必须使用预分割的测试数据）
    test_file = f"data/split/{args.benchmark}_test.json"
    test_examples = []
    
    if not os.path.exists(test_file):
        logger.logger.error(f"❌ 预分割的测试数据不存在: {test_file}")
        logger.logger.error("请先运行 prepare_data.py 脚本准备数据分割")
        raise FileNotFoundError(f"预分割的测试数据不存在: {test_file}")
    
    # 使用预分割的测试数据
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    for sample in test_data:
        example = PairwiseExample(
            question_id=sample["question_id"],
            instruction=sample["instruction"],
            response_a=sample["response_a"],
            response_b=sample["response_b"],
            model_a=sample["model_a"],
            model_b=sample["model_b"],
            metadata=sample.get("metadata", {})
        )
        test_examples.append(example)
    
    logger.logger.info(f"✅ 使用预分割的测试数据: {len(test_examples)} 个样本")
    
    # 创建攻击方法
    attacks = get_attack_methods()
    logger.logger.info(f"✅ 加载了 {len(attacks)} 种攻击方法")
    
    # 创建Judge
    try:
        if args.judge_type == "qwen":
            judge = create_qwen_judge()
        elif args.judge_type == "llama":
            judge = create_llama_judge()
        else:
            raise ValueError(f"不支持的Judge类型: {args.judge_type}")
        logger.logger.info("✅ Judge初始化成功")
    except Exception as e:
        logger.logger.error(f"❌ Judge初始化失败: {e}")
        return
    
    # 创建训练环境
    train_env = EnhancedHydraAttackEnv(
        examples=train_examples,
        attacks=attacks,
        judge=judge,
        max_queries=args.max_queries,
        success_reward=args.success_reward,
        query_penalty=args.query_penalty,
        diversity_bonus=args.diversity_bonus,
        efficiency_bonus=args.efficiency_bonus,
        confidence_threshold=args.confidence_threshold
    )
    
    # 创建DQN智能体
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        target_update_freq=args.target_update_freq
    )
    
    logger.logger.info(f"✅ 创建DQN智能体: 状态维度={state_dim}, 动作维度={action_dim}")
    logger.logger.info(f"   DQN参数: 学习率={args.learning_rate}, 批次大小={args.batch_size}, 隐藏层维度={args.hidden_dim}")
    logger.logger.info(f"   探索参数: ε={args.epsilon}, ε_min={args.epsilon_min}, ε_decay={args.epsilon_decay}")
    logger.logger.info(f"   网络参数: γ={args.gamma}, 内存大小={args.memory_size}, 目标更新频率={args.target_update_freq}")
    logger.logger.info(f"   环境参数: 成功奖励={args.success_reward}, 查询惩罚={args.query_penalty}, 置信度阈值={args.confidence_threshold}")
    
    # 创建测试环境用于训练过程中的评估
    test_env = EnhancedHydraAttackEnv(
        examples=test_examples,
        attacks=attacks,
        judge=judge,
        max_queries=args.max_queries,
        success_reward=args.success_reward,
        query_penalty=args.query_penalty,
        diversity_bonus=args.diversity_bonus,
        efficiency_bonus=args.efficiency_bonus,
        confidence_threshold=args.confidence_threshold
    )
    
    # 创建训练器
    config = RLConfig(
        algorithm="dqn",
        total_timesteps=args.episodes,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
        epsilon_start=args.epsilon,
        epsilon_end=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update_freq,
        save_freq=1000000,  # 设置为很大的值，实际上不会触发保存
        eval_freq=args.eval_freq,
        max_queries=args.max_queries,
        success_reward=args.success_reward,
        query_penalty=args.query_penalty,
        diversity_bonus=args.diversity_bonus,
        efficiency_bonus=args.efficiency_bonus,
        confidence_threshold=args.confidence_threshold,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    
    trainer = RLTrainer(
        env=train_env,
        agent=agent,
        config=config,
        save_dir=args.model_dir,
        test_env=test_env,  # 传入测试环境用于评估
        test_examples=test_examples,  # 传入测试样本用于evaluate_on_unseen_data
        validation_examples=validation_examples,  # 传入验证样本用于周期性验证和early stopping
        attacks=attacks,  # 传入攻击方法
        judge=judge,  # 传入judge
        logger=logger  # 传入logger
    )
    
    # 开始训练
    logger.logger.info("🎯 开始训练...")
    start_time = time.time()
    
    training_results = trainer.train(args.episodes)
    
    training_time = time.time() - start_time
    logger.logger.info(f"🎉 训练完成！")
    logger.logger.info(f"⏱️  训练耗时: {training_time:.2f}秒")
    
    # 在测试集上评估（真正的未见数据）
    logger.logger.info("🔍 在测试集上评估模型（未见数据）...")
    test_results = evaluate_on_unseen_data(agent, test_examples, attacks, judge, args.max_queries, 
                                         args.success_reward, args.query_penalty, args.diversity_bonus, 
                                         args.efficiency_bonus, args.confidence_threshold, logger)
    
    # 保存模型 - 文件名包含数据集信息，新模型会覆盖旧模型
    model_path = os.path.join(args.model_dir, f"fast_rl_attacker_{args.benchmark}.pth")
    agent.save_model(model_path)
    logger.logger.info(f"💾 模型已保存到: {model_path}")
    
    # 保存动作映射 - 文件名包含数据集信息，新模型会覆盖旧模型
    action_mapping = {}
    for i, attack in enumerate(attacks):
        for j in range(attack.get_action_space_size()):
            action_mapping[i * 100 + j] = {
                'attack_method': attack.__class__.__name__,
                'action_id': j,
                'action_description': attack.get_action_description(j)
            }
    
    mapping_path = os.path.join(args.model_dir, f"action_mapping_{args.benchmark}.json")
    with open(mapping_path, 'w') as f:
        json.dump(action_mapping, f, indent=2)
    logger.logger.info(f"💾 动作映射已保存到: {mapping_path}")
    
    # 保存训练配置和结果 - 文件名包含数据集信息，新模型会覆盖旧模型
    config_dict = {
        'algorithm': 'DQN',
        'benchmark': args.benchmark,
        'max_samples': args.max_samples,
        'train_ratio': args.train_ratio,
        'test_ratio': args.test_ratio,
        'episodes': args.episodes,
        'max_queries': args.max_queries,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'random_seed': args.random_seed,
        'training_time': training_time,
        'train_samples': len(train_examples),
        'test_samples': len(test_examples),
        'validation_samples': len(validation_examples),
        'test_results': test_results,
        'dqn_params': {
            'hidden_dim': args.hidden_dim,
            'gamma': args.gamma,
            'epsilon': args.epsilon,
            'epsilon_min': args.epsilon_min,
            'epsilon_decay': args.epsilon_decay,
            'memory_size': args.memory_size,
            'target_update_freq': args.target_update_freq
        },
        'env_params': {
            'success_reward': args.success_reward,
            'query_penalty': args.query_penalty,
            'diversity_bonus': args.diversity_bonus,
            'efficiency_bonus': args.efficiency_bonus,
            'confidence_threshold': args.confidence_threshold
        },
        'early_stopping_params': {
            'patience': args.early_stopping_patience,
            'min_delta': args.early_stopping_min_delta
        }
    }
    
    config_path = os.path.join(args.model_dir, f"training_config_{args.benchmark}.json")
    # 转换numpy类型为Python原生类型
    config_dict = convert_numpy_types(config_dict)
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.logger.info(f"💾 训练配置已保存到: {config_path}")
    
    # 输出最终结果
    logger.logger.info("=" * 80)
    logger.logger.info("📊 最终训练结果")
    logger.logger.info("=" * 80)
    logger.logger.info(f"训练集表现: {training_results['final_eval']}")
    logger.logger.info(f"测试集表现（未见数据）: {test_results}")
    logger.logger.info("=" * 80)
    
    return {
        'training_results': training_results,
        'test_results': test_results,
        'config': config_dict
    }


if __name__ == "__main__":
    main()
