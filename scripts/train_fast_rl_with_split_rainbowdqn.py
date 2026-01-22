#!/usr/bin/env python3
"""
改进的快速RL训练脚本 - Rainbow DQN版本（支持数据分割和未见数据评估）
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
from evaluation.glm_judge import create_glm_judge
from evaluation.mistral_judge import create_mistral_judge
from evaluation.gemma_judge import create_gemma_judge
from evaluation.gemma_judge1b import create_gemma_judge as create_gemma_judge_1b
from rl.enhanced_environment import EnhancedHydraAttackEnv
from rl.agent import RainbowDQNAgent
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
    parser = argparse.ArgumentParser(description="训练快速RL攻击智能体 - Rainbow DQN（支持数据分割）")
    parser.add_argument("--benchmark", type=str, default="alpaca_eval", 
                       choices=["alpaca_eval", "arena_hard", "code_judge_bench", "mtbench", "llmbar"],
                       help="使用的benchmark")
    parser.add_argument("--max_samples", default=100, help="总样本数，使用 'full' 表示使用全量数据")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--validation_ratio", type=float, default=0.1, help="验证集比例（从训练集中分出）")
    parser.add_argument("--judge_model_path", type=str, required=True, help="Judge模型路径")
    parser.add_argument("--judge_type", type=str, default="qwen", help="Judge类型")
    parser.add_argument("--episodes", type=int, default=1000, help="训练轮数")
    parser.add_argument("--max_queries", type=int, default=10, help="训练时的最大查询次数")
    parser.add_argument("--max_queries_test", type=int, default=None, help="测试时的最大查询次数（默认与训练时相同）")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--model_dir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    parser.add_argument("--eval_freq", type=int, default=500, help="验证频率（每多少个episode评估一次）")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001, help="Early stopping最小改善阈值")
    
    # Rainbow DQN算法特有参数
    parser.add_argument("--hidden_dim", type=int, default=512, help="神经网络隐藏层维度")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--epsilon", type=float, default=1.0, help="初始探索率")
    parser.add_argument("--epsilon_min", type=float, default=0.01, help="最小探索率")
    parser.add_argument("--epsilon_decay", type=float, default=0.9995, help="探索率衰减因子")
    parser.add_argument("--memory_size", type=int, default=50000, help="经验回放缓冲区大小")
    parser.add_argument("--target_update_freq", type=int, default=500, help="目标网络更新频率")
    
    # Rainbow DQN特有参数
    parser.add_argument("--prioritized_replay", type=str, default="true", help="是否使用优先经验回放 (true/false)")
    parser.add_argument("--prioritized_replay_alpha", type=float, default=0.6, help="优先经验回放alpha参数")
    parser.add_argument("--prioritized_replay_beta", type=float, default=0.4, help="优先经验回放beta参数")
    parser.add_argument("--prioritized_replay_beta_increment", type=float, default=0.001, help="优先经验回放beta增量")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值（防止梯度爆炸）")
    
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
    
    # 在时间戳目录下创建新的目录结构
    config_dir = os.path.join(args.model_dir, "config")
    summary_dir = os.path.join(args.model_dir, "summary")
    training_dir = os.path.join(args.model_dir, "training")
    evaluation_dir = os.path.join(args.model_dir, "evaluation")
    judge_logs_dir = os.path.join(args.model_dir, "judge_logs")
    logs_dir = os.path.join(args.model_dir, "logs")
    
    # 创建所有子目录
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)
    os.makedirs(judge_logs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 创建评估子目录
    test_samples_dir = os.path.join(evaluation_dir, "test_samples")
    test_samples_successful_dir = os.path.join(test_samples_dir, "successful")
    test_samples_failed_dir = os.path.join(test_samples_dir, "failed")
    os.makedirs(test_samples_successful_dir, exist_ok=True)
    os.makedirs(test_samples_failed_dir, exist_ok=True)
    
    # 创建训练子目录
    checkpoints_dir = os.path.join(training_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 创建judge_logs子目录
    judge_logs_training_dir = os.path.join(judge_logs_dir, "training")
    judge_logs_test_dir = os.path.join(judge_logs_dir, "test")
    os.makedirs(judge_logs_training_dir, exist_ok=True)
    os.makedirs(judge_logs_test_dir, exist_ok=True)
    
    # 初始化日志系统（使用logs子目录）
    logger = HydraLogger(log_dir=logs_dir, results_dir=args.model_dir)
    
    logger.logger.info("🚀 开始训练快速RL攻击智能体 - Rainbow DQN（支持数据分割）")
    logger.logger.info(f"Benchmark: {args.benchmark}")
    logger.logger.info(f"实验目录: {args.model_dir}")
    logger.logger.info(f"日志目录: {logs_dir}")
    logger.logger.info("📁 已创建目录结构:")
    logger.logger.info(f"   - config/: {config_dir}")
    logger.logger.info(f"   - summary/: {summary_dir}")
    logger.logger.info(f"   - training/: {training_dir}")
    logger.logger.info(f"     - checkpoints/: {checkpoints_dir}")
    logger.logger.info(f"   - evaluation/: {evaluation_dir}")
    logger.logger.info(f"   - judge_logs/: {judge_logs_dir}")
    logger.logger.info(f"   - logs/: {logs_dir}")
    logger.logger.info(f"总样本数: {args.max_samples}")
    logger.logger.info(f"数据分割: 训练{args.train_ratio:.1%} / 测试{args.test_ratio:.1%} / 验证{args.validation_ratio:.1%}")
    logger.logger.info(f"训练轮数: {args.episodes}")
    # 设置测试时的最大查询次数（如果未指定，则使用训练时的值）
    max_queries_test = args.max_queries_test if args.max_queries_test is not None else args.max_queries
    logger.logger.info(f"训练时最大查询次数: {args.max_queries}")
    logger.logger.info(f"测试时最大查询次数: {max_queries_test}")
    
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
    
    # 创建Rainbow DQN智能体
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    
    agent = RainbowDQNAgent(
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
        target_update_freq=args.target_update_freq,
        prioritized_replay=(args.prioritized_replay.lower() in ['true', '1', 'yes']),
        prioritized_replay_alpha=args.prioritized_replay_alpha,
        prioritized_replay_beta=args.prioritized_replay_beta,
        prioritized_replay_beta_increment=args.prioritized_replay_beta_increment,
        max_grad_norm=args.max_grad_norm
    )
    
    logger.logger.info(f"✅ 创建Rainbow DQN智能体: 状态维度={state_dim}, 动作维度={action_dim}")
    logger.logger.info(f"   Rainbow DQN参数: 学习率={args.learning_rate}, 批次大小={args.batch_size}, 隐藏层维度={args.hidden_dim}")
    logger.logger.info(f"   探索参数: ε={args.epsilon}, ε_min={args.epsilon_min}, ε_decay={args.epsilon_decay}")
    logger.logger.info(f"   网络参数: γ={args.gamma}, 内存大小={args.memory_size}, 目标更新频率={args.target_update_freq}")
    prioritized_replay_enabled = (args.prioritized_replay.lower() in ['true', '1', 'yes'])
    logger.logger.info(f"   优先经验回放: {prioritized_replay_enabled}, α={args.prioritized_replay_alpha}, β={args.prioritized_replay_beta}")
    logger.logger.info(f"   环境参数: 成功奖励={args.success_reward}, 查询惩罚={args.query_penalty}, 置信度阈值={args.confidence_threshold}")
    
    # 创建测试环境用于训练过程中的评估（使用测试时的最大查询次数）
    test_env = EnhancedHydraAttackEnv(
        examples=test_examples,
        attacks=attacks,
        judge=judge,
        max_queries=max_queries_test,
        success_reward=args.success_reward,
        query_penalty=args.query_penalty,
        diversity_bonus=args.diversity_bonus,
        efficiency_bonus=args.efficiency_bonus,
        confidence_threshold=args.confidence_threshold
    )
    
    # 创建训练器
    config = RLConfig(
        algorithm="rainbowdqn",
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
        max_queries=args.max_queries,  # 训练时的最大查询次数
        max_queries_test=max_queries_test,  # 测试时的最大查询次数（用于验证集和测试集评估）
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
    # 启用详细样本记录保存（使用测试时的最大查询次数）
    test_results = evaluate_on_unseen_data(agent, test_examples, attacks, judge, max_queries_test, 
                                         args.success_reward, args.query_penalty, args.diversity_bonus, 
                                         args.efficiency_bonus, args.confidence_threshold, logger,
                                         save_detailed_samples=True, save_dir=args.model_dir)
    
    # 保存模型 - 保存到checkpoints目录，同时在根目录保留一份（兼容性）
    model_path_checkpoint = os.path.join(checkpoints_dir, f"fast_rl_attacker_{args.benchmark}.pth")
    agent.save_model(model_path_checkpoint)
    logger.logger.info(f"💾 模型已保存到: {model_path_checkpoint}")
    
    # 同时在根目录保留一份（兼容性）
    model_path_root = os.path.join(args.model_dir, f"fast_rl_attacker_{args.benchmark}.pth")
    import shutil
    shutil.copy2(model_path_checkpoint, model_path_root)
    logger.logger.info(f"💾 模型已复制到根目录: {model_path_root}")
    
    # 移动best_model和final_model到checkpoints目录（如果存在）
    best_model_src = os.path.join(args.model_dir, "best_model.pth")
    final_model_src = os.path.join(args.model_dir, "final_model.pth")
    
    if os.path.exists(best_model_src):
        best_model_dst = os.path.join(checkpoints_dir, "best_model.pth")
        shutil.move(best_model_src, best_model_dst)
        logger.logger.info(f"💾 最佳模型已移动到: {best_model_dst}")
    
    if os.path.exists(final_model_src):
        final_model_dst = os.path.join(checkpoints_dir, "final_model.pth")
        shutil.move(final_model_src, final_model_dst)
        logger.logger.info(f"💾 最终模型已移动到: {final_model_dst}")
    
    # 保存动作映射 - 保存到config目录，同时在根目录保留一份（兼容性）
    action_mapping = {}
    for i, attack in enumerate(attacks):
        for j in range(attack.get_action_space_size()):
            action_mapping[i * 100 + j] = {
                'attack_method': attack.__class__.__name__,
                'action_id': j,
                'action_description': attack.get_action_description(j)
            }
    
    # 保存到config目录
    mapping_path = os.path.join(config_dir, f"action_mapping_{args.benchmark}.json")
    with open(mapping_path, 'w') as f:
        json.dump(action_mapping, f, indent=2)
    logger.logger.info(f"💾 动作映射已保存到: {mapping_path}")
    
    # 同时在根目录保留一份（兼容性）
    mapping_path_root = os.path.join(args.model_dir, f"action_mapping_{args.benchmark}.json")
    with open(mapping_path_root, 'w') as f:
        json.dump(action_mapping, f, indent=2)
    
    # 保存训练配置和结果 - 文件名包含数据集信息，新模型会覆盖旧模型
    config_dict = {
        'algorithm': 'RainbowDQN',
        'benchmark': args.benchmark,
        'max_samples': args.max_samples,
        'train_ratio': args.train_ratio,
        'test_ratio': args.test_ratio,
        'validation_ratio': args.validation_ratio,
        'episodes': args.episodes,
        'max_queries': args.max_queries,
        'max_queries_test': max_queries_test,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'random_seed': args.random_seed,
        'training_time': training_time,
        'train_samples': len(train_examples),
        'test_samples': len(test_examples),
        'validation_samples': len(validation_examples),
        'test_results': test_results,
        'judge_config': {
            'judge_type': args.judge_type,
            'judge_model_path': args.judge_model_path
        },
        'rainbowdqn_params': {
            'hidden_dim': args.hidden_dim,
            'gamma': args.gamma,
            'epsilon': args.epsilon,
            'epsilon_min': args.epsilon_min,
            'epsilon_decay': args.epsilon_decay,
            'memory_size': args.memory_size,
            'target_update_freq': args.target_update_freq,
            'prioritized_replay': (args.prioritized_replay.lower() in ['true', '1', 'yes']),
            'prioritized_replay_alpha': args.prioritized_replay_alpha,
            'prioritized_replay_beta': args.prioritized_replay_beta,
            'prioritized_replay_beta_increment': args.prioritized_replay_beta_increment,
            'max_grad_norm': args.max_grad_norm
        },
        'env_params': {
            'success_reward': args.success_reward,
            'query_penalty': args.query_penalty,
            'diversity_bonus': args.diversity_bonus,
            'efficiency_bonus': args.efficiency_bonus,
            'confidence_threshold': args.confidence_threshold
        },
        'training_config': {
            'eval_freq': args.eval_freq,
            'model_dir': args.model_dir,
            'log_dir': args.log_dir
        },
        'early_stopping_params': {
            'patience': args.early_stopping_patience,
            'min_delta': args.early_stopping_min_delta
        }
    }
    
    # 保存训练配置 - 保存到config目录，同时在根目录保留一份（兼容性）
    config_dict = convert_numpy_types(config_dict)
    
    # 保存到config目录
    config_path = os.path.join(config_dir, f"training_config_{args.benchmark}.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.logger.info(f"💾 训练配置已保存到: {config_path}")
    
    # 同时在根目录保留一份（兼容性）
    config_path_root = os.path.join(args.model_dir, f"training_config_{args.benchmark}.json")
    with open(config_path_root, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # ========== 保存实验元数据 ==========
    import platform
    import socket
    experiment_metadata = {
        'experiment_id': os.path.basename(args.model_dir),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'benchmark': args.benchmark,
        'algorithm': 'RainbowDQN',
        'environment': {
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')
        },
        'data_info': {
            'train_samples': len(train_examples),
            'test_samples': len(test_examples),
            'validation_samples': len(validation_examples),
            'max_samples': args.max_samples,
            'train_ratio': args.train_ratio,
            'test_ratio': args.test_ratio,
            'validation_ratio': args.validation_ratio
        },
        'training_info': {
            'episodes': args.episodes,
            'training_time_seconds': training_time,
            'random_seed': args.random_seed
        }
    }
    experiment_metadata_path = os.path.join(config_dir, "experiment_metadata.json")
    with open(experiment_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_metadata, f, indent=2, ensure_ascii=False)
    logger.logger.info(f"💾 实验元数据已保存到: {experiment_metadata_path}")
    
    # ========== 保存Summary文件 ==========
    
    # 1. 保存训练汇总 (summary/training_summary.json)
    training_summary = {
        'total_episodes': args.episodes,
        'final_success_rate': training_results.get('final_eval', {}).get('success_rate', 0.0),
        'avg_episode_reward': np.mean(training_results.get('training_stats', {}).get('episode_rewards', [0])) if training_results.get('training_stats', {}).get('episode_rewards') else 0.0,
        'avg_queries_per_episode': np.mean(training_results.get('training_stats', {}).get('query_counts', [0])) if training_results.get('training_stats', {}).get('query_counts') else 0.0,
        'training_time': training_time,
        'best_checkpoint': 'best_model.pth' if os.path.exists(os.path.join(args.model_dir, 'best_model.pth')) else None,
        'early_stopping_triggered': False,  # 可以从training_results中获取
        'validation_scores': training_results.get('training_stats', {}).get('validation_scores', []),
        'episode_rewards': training_results.get('training_stats', {}).get('episode_rewards', []),
        'episode_success_rates': training_results.get('training_stats', {}).get('success_rates', []),
        # 计算每100个episode的平均值用于绘图
        'episode_rewards_avg_100': [float(np.mean(training_results.get('training_stats', {}).get('episode_rewards', [])[i:i+100])) 
                                   for i in range(0, len(training_results.get('training_stats', {}).get('episode_rewards', [])), 100)
                                   if i < len(training_results.get('training_stats', {}).get('episode_rewards', []))],
        'episode_success_rates_avg_100': [float(np.mean(training_results.get('training_stats', {}).get('success_rates', [])[i:i+100])) 
                                         for i in range(0, len(training_results.get('training_stats', {}).get('success_rates', [])), 100)
                                         if i < len(training_results.get('training_stats', {}).get('success_rates', []))]
    }
    training_summary = convert_numpy_types(training_summary)
    training_summary_path = os.path.join(summary_dir, "training_summary.json")
    with open(training_summary_path, 'w', encoding='utf-8') as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)
    logger.logger.info(f"💾 训练汇总已保存到: {training_summary_path}")
    
    # 2. 保存测试汇总 (summary/test_summary.json)
    test_summary = {
        'total_samples': test_results.get('total_samples', 0),
        'successful_attacks': test_results.get('successful_attacks', 0),
        'success_rate': test_results.get('success_rate', 0.0),
        'avg_queries': test_results.get('avg_queries', 0.0),
        'avg_queries_successful': test_results.get('avg_queries_successful', 0.0),
        'avg_reward': test_results.get('avg_reward', 0.0)
    }
    test_summary = convert_numpy_types(test_summary)
    test_summary_path = os.path.join(summary_dir, "test_summary.json")
    with open(test_summary_path, 'w', encoding='utf-8') as f:
        json.dump(test_summary, f, indent=2, ensure_ascii=False)
    logger.logger.info(f"💾 测试汇总已保存到: {test_summary_path}")
    
    # 3. 保存Episode统计CSV (summary/episode_statistics.csv)
    import csv
    episode_stats_path = os.path.join(summary_dir, "episode_statistics.csv")
    training_stats = training_results.get('training_stats', {})
    episode_rewards = training_stats.get('episode_rewards', [])
    episode_lengths = training_stats.get('episode_lengths', [])
    episode_success_rates = training_stats.get('success_rates', [])
    episode_query_counts = training_stats.get('query_counts', [])
    episode_losses = training_stats.get('training_losses', [])
    episode_q_values = training_stats.get('q_values', [])
    episode_epsilon = training_stats.get('epsilon_values', [])
    
    with open(episode_stats_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'length', 'success_rate', 'query_count', 'loss', 'avg_q_value', 'epsilon'])
        max_len = max(len(episode_rewards), len(episode_lengths), len(episode_success_rates), 
                     len(episode_query_counts), len(episode_losses), len(episode_q_values), len(episode_epsilon))
        for i in range(max_len):
            writer.writerow([
                i + 1,
                episode_rewards[i] if i < len(episode_rewards) else 0,
                episode_lengths[i] if i < len(episode_lengths) else 0,
                episode_success_rates[i] if i < len(episode_success_rates) else 0,
                episode_query_counts[i] if i < len(episode_query_counts) else 0,
                episode_losses[i] if i < len(episode_losses) else 0,
                episode_q_values[i] if i < len(episode_q_values) else 0,
                episode_epsilon[i] if i < len(episode_epsilon) else 0
            ])
    logger.logger.info(f"💾 Episode统计已保存到: {episode_stats_path}")
    
    # 4. 保存攻击使用统计 (summary/attack_usage_stats.json)
    # 从训练结果中获取攻击使用统计
    attack_usage_stats_training = training_results.get('attack_usage_stats', {})
    
    # 从测试结果中获取攻击使用统计
    attack_usage_stats_test = test_results.get('attack_usage_stats', {})
    
    attack_usage_stats = {
        'training': attack_usage_stats_training,
        'test': attack_usage_stats_test
    }
    attack_usage_stats = convert_numpy_types(attack_usage_stats)
    attack_usage_stats_path = os.path.join(summary_dir, "attack_usage_stats.json")
    with open(attack_usage_stats_path, 'w', encoding='utf-8') as f:
        json.dump(attack_usage_stats, f, indent=2, ensure_ascii=False)
    logger.logger.info(f"💾 攻击使用统计已保存到: {attack_usage_stats_path}")
    
    # 同时保存训练和测试的独立文件（从trainer保存的文件复制）
    attack_usage_training_src = os.path.join(args.model_dir, "attack_usage_stats_training.json")
    attack_usage_training_dst = os.path.join(summary_dir, "attack_usage_stats_training.json")
    if os.path.exists(attack_usage_training_src):
        import shutil
        shutil.copy2(attack_usage_training_src, attack_usage_training_dst)
        logger.logger.info(f"💾 训练攻击使用统计已复制到: {attack_usage_training_dst}")
    
    attack_usage_test_src = os.path.join(args.model_dir, "attack_usage_stats_test.json")
    attack_usage_test_dst = os.path.join(summary_dir, "attack_usage_stats_test.json")
    if os.path.exists(attack_usage_test_src):
        import shutil
        shutil.copy2(attack_usage_test_src, attack_usage_test_dst)
        logger.logger.info(f"💾 测试攻击使用统计已复制到: {attack_usage_test_dst}")
    
    # ========== 保存训练曲线数据 ==========
    training_stats = training_results.get('training_stats', {})
    training_curves = {
        'episode_rewards': episode_rewards,
        'episode_success_rates': episode_success_rates,
        'episode_query_counts': episode_query_counts,
        'validation_scores': training_stats.get('validation_scores', []),
        'validation_episodes': [i * args.eval_freq for i in range(len(training_stats.get('validation_scores', [])))],
        # 添加训练过程指标
        'training_losses': training_stats.get('training_losses', []),
        'q_values': training_stats.get('q_values', []),
        'epsilon_values': training_stats.get('epsilon_values', []),
        'max_q_values': training_stats.get('max_q_values', []),
        'min_q_values': training_stats.get('min_q_values', [])
    }
    training_curves = convert_numpy_types(training_curves)
    training_curves_path = os.path.join(training_dir, "training_curves.json")
    with open(training_curves_path, 'w', encoding='utf-8') as f:
        json.dump(training_curves, f, indent=2, ensure_ascii=False)
    logger.logger.info(f"💾 训练曲线数据已保存到: {training_curves_path}")
    
    # 保存trajectory数据（如果存在）
    trajectories_src = os.path.join(args.model_dir, "episode_trajectories.json")
    trajectories_dst = os.path.join(training_dir, "episode_trajectories.json")
    if os.path.exists(trajectories_src):
        import shutil
        shutil.copy2(trajectories_src, trajectories_dst)
        logger.logger.info(f"💾 Episode trajectories已复制到: {trajectories_dst}")
    
    # 将training_stats.json移动到summary目录（可选，也可以保留在根目录）
    training_stats_src = os.path.join(args.model_dir, "training_stats.json")
    training_stats_dst = os.path.join(summary_dir, "training_stats.json")
    if os.path.exists(training_stats_src):
        import shutil
        shutil.copy2(training_stats_src, training_stats_dst)
        logger.logger.info(f"💾 训练统计已复制到: {training_stats_dst}")
    
    # 创建README文件，说明目录结构
    readme_path = os.path.join(args.model_dir, "README.md")
    readme_content = f"""# RL训练实验结果

## 实验信息
- **Benchmark**: {args.benchmark}
- **算法**: Rainbow DQN
- **训练时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
- **训练耗时**: {training_time:.2f}秒
- **训练轮数**: {args.episodes}
- **总样本数**: {args.max_samples}

## 目录结构说明

```
{os.path.basename(args.model_dir)}/
├── config/                    # 配置和元数据
│   ├── training_config.json   # 完整超参数配置
│   ├── action_mapping.json    # 动作映射
│   └── experiment_metadata.json  # 实验元数据（时间、环境等）
│
├── summary/                   # 汇总数据（快速查看）
│   ├── training_summary.json  # 训练过程汇总
│   ├── test_summary.json      # 测试结果汇总
│   ├── attack_usage_stats.json  # 攻击方法使用统计
│   └── episode_statistics.csv   # Episode级别统计
│
├── training/                  # 训练过程数据
│   ├── checkpoints/          # 模型检查点
│   │   ├── fast_rl_attacker_*.pth
│   │   ├── best_model.pth
│   │   └── final_model.pth
│   └── training_curves.json  # 训练曲线数据
│
├── evaluation/               # 评估数据
│   └── test_samples/         # 测试样本详细记录
│       ├── successful/       # 成功攻击样本
│       ├── failed/           # 失败攻击样本
│       └── index.json        # 样本索引
│
├── judge_logs/               # Judge输入输出记录
│   ├── training/             # 训练阶段judge记录
│   └── test/                 # 测试阶段judge记录
│
├── logs/                     # 日志文件
│
├── fast_rl_attacker_{args.benchmark}.pth  # 模型文件
├── action_mapping_{args.benchmark}.json  # 动作映射（根目录备份）
└── training_config_{args.benchmark}.json # 训练配置（根目录备份）
```

## 快速查看结果

### 查看汇总结果
```bash
cat summary/training_summary.json
cat summary/test_summary.json
cat summary/attack_usage_stats.json
```

### 查看配置
```bash
cat config/training_config.json
```

### 查看日志
```bash
cat logs/training.log
```

## 实验结果

### 训练集表现
{training_results.get('final_eval', 'N/A')}

### 测试集表现（未见数据）
- 成功率: {test_results.get('success_rate', 0):.3f}
- 平均查询次数: {test_results.get('avg_queries', 0):.3f}
- 平均查询次数（成功）: {test_results.get('avg_queries_successful', 0):.3f}
- 平均奖励: {test_results.get('avg_reward', 0):.3f}
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    logger.logger.info(f"📝 README已保存到: {readme_path}")
    
    # 输出最终结果
    logger.logger.info("=" * 80)
    logger.logger.info("📊 最终训练结果")
    logger.logger.info("=" * 80)
    logger.logger.info(f"训练集表现: {training_results['final_eval']}")
    logger.logger.info(f"测试集表现（未见数据）: {test_results}")
    logger.logger.info("=" * 80)
    logger.logger.info(f"📁 所有结果文件已保存在: {args.model_dir}")
    logger.logger.info(f"📝 查看README了解目录结构: {readme_path}")
    
    return {
        'training_results': training_results,
        'test_results': test_results,
        'config': config_dict
    }


if __name__ == "__main__":
    main()

