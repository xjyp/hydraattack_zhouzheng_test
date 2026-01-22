#!/usr/bin/env python3
"""
RainbowDQN跨LLM泛化评估脚本
加载在某个source LLM上训练好的RainbowDQN模型，在target LLM上测试攻击的泛化效果
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
from evaluation.gemma_judge import create_gemma_judge
from evaluation.gemma_judge1b import create_gemma_judge as create_gemma_judge_1b
from evaluation.mistral_judge import create_mistral_judge
from evaluation.glm_judge import create_glm_judge
from evaluation.judge import create_online_judge
from rl.enhanced_environment import EnhancedHydraAttackEnv
from rl.agent import RainbowDQNAgent
from utils.logger import HydraLogger


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
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
    """Load benchmark test data"""
    test_file = f"data/split/{benchmark}_test.json"
    
    if not os.path.exists(test_file):
        print(f"❌ Test data not found: {test_file}")
        raise FileNotFoundError(f"Test data not found: {test_file}")
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded test data: {len(data)} samples")
    
    # Limit sample count
    if max_samples != "full" and max_samples is not None:
        try:
            max_samples_int = int(max_samples)
            if len(data) > max_samples_int:
                data = data[:max_samples_int]
                print(f"✅ Limited sample count to: {max_samples_int}")
        except (ValueError, TypeError):
            # If max_samples cannot be converted to int, ignore the limit
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
    """Get all attack method instances"""
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


def load_rainbowdqn_agent(model_path: str, config_path: str, logger: HydraLogger = None) -> RainbowDQNAgent:
    """Load trained RainbowDQN agent"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get RainbowDQN parameters
    rainbowdqn_params = config.get('rainbowdqn_params', {})
    env_params = config.get('env_params', {})
    
    # Create attack methods to get action space size
    attacks = get_attack_methods()
    total_actions = sum(attack.get_action_space_size() for attack in attacks)
    
    # Infer state dimension from saved model
    import torch
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Infer state dimension from the first linear layer's weight shape
    if 'q_network_state_dict' in checkpoint:
        state_dict = checkpoint['q_network_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Find the first linear layer's weight (usually 'feature.0.weight')
    first_layer_weight = None
    for key, value in state_dict.items():
        if 'weight' in key and len(value.shape) == 2:
            # Check if it's the first layer (feature.0.weight or 0.weight)
            if 'feature.0.weight' in key or (key.startswith('0.') and 'weight' in key):
                first_layer_weight = value
                break
    
    if first_layer_weight is not None:
        state_dim = first_layer_weight.shape[1]  # Input dimension
    else:
        state_dim = 512  # Default value
    
    if logger:
        logger.logger.info(f"   Inferred state dimension: {state_dim}")
    
    # Create RainbowDQN agent (using inferred state dimension)
    agent = RainbowDQNAgent(
        state_dim=state_dim,
        action_dim=total_actions,
        hidden_dim=rainbowdqn_params.get('hidden_dim', 384),
        learning_rate=rainbowdqn_params.get('learning_rate', 0.0001),
        gamma=rainbowdqn_params.get('gamma', 0.99),
        epsilon=0.0,  # Use greedy policy during evaluation
        epsilon_min=0.0,
        epsilon_decay=1.0,
        batch_size=rainbowdqn_params.get('batch_size', 64),
        memory_size=rainbowdqn_params.get('memory_size', 100000),
        target_update_freq=rainbowdqn_params.get('target_update_freq', 1500),
        prioritized_replay=rainbowdqn_params.get('prioritized_replay', True),
        prioritized_replay_alpha=rainbowdqn_params.get('prioritized_replay_alpha', 0.6),
        prioritized_replay_beta=rainbowdqn_params.get('prioritized_replay_beta', 0.4),
        prioritized_replay_beta_increment=rainbowdqn_params.get('prioritized_replay_beta_increment', 0.001),
        max_grad_norm=rainbowdqn_params.get('max_grad_norm', 1.0)
    )
    
    # Load model weights
    agent.load_model(model_path)
    
    if logger:
        logger.logger.info(f"✅ Successfully loaded RainbowDQN model: {model_path}")
        logger.logger.info(f"   State dimension: {state_dim}")
        logger.logger.info(f"   Action space size: {total_actions}")
        logger.logger.info(f"   Hidden dimension: {rainbowdqn_params.get('hidden_dim', 384)}")
        logger.logger.info(f"   Learning rate: {rainbowdqn_params.get('learning_rate', 0.0001)}")
    
    return agent


def evaluate_cross_llm_performance(agent, test_examples: List[PairwiseExample], 
                                 attacks: List[Any], judge, max_queries: int = 10,
                                 success_reward: float = 20.0, query_penalty: float = 0.5,
                                 diversity_bonus: float = 1.0, efficiency_bonus: float = 2.0,
                                 confidence_threshold: float = 0.7,
                                 logger: HydraLogger = None) -> Dict[str, float]:
    """Evaluate cross-LLM generalization performance"""
    
    if logger:
        logger.logger.info("🔍 Starting cross-LLM generalization evaluation...")
        logger.logger.info(f"   Target Judge: {judge.__class__.__name__}")
        logger.logger.info(f"   Test samples: {len(test_examples)}")
    
    success_count = 0
    total_queries = 0
    queries_successful = 0  # Total queries used in successful attacks
    total_reward = 0.0
    detailed_results = []
    
    for i, example in enumerate(test_examples):
        if logger and i % 10 == 0:
            logger.logger.info(f"  Evaluation progress: {i+1}/{len(test_examples)}")
        
        # 1. Get original preference first (for evaluation)
        try:
            original_response = judge.judge_pairwise(example)
            original_preference = original_response.preference
            original_confidence = original_response.confidence
        except Exception as e:
            if logger:
                logger.logger.warning(f"Failed to get original preference: {e}")
            continue  # Skip failed samples
        
        # 2. Create environment
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
        
        # 3. Reset environment (evaluation mode: use specific sample and pre-fetched preference)
        state, _ = env.reset(options={
            'use_specific_sample': True,
            'sample_idx': 0,
            'original_preference': original_preference,
            'original_confidence': original_confidence
        })
        episode_reward = 0
        queries_used = 0
        attack_sequence = []
        
        # 4. Execute attack sequence
        for step in range(max_queries):
            # Get action mask (prevent reusing failed actions)
            action_mask = env.get_action_mask()
            
            # Use trained agent to select action (greedy policy)
            action = agent.select_action(state, training=False, action_mask=action_mask)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Record attack information
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
        
        # 5. Check attack result
        original_preference = env.original_preference
        final_preference = info.get('current_preference', original_preference)
        # Use success information returned by environment, which already considers PositionAttack's special logic
        success = info.get('success', False)
        
        if success:
            success_count += 1
            queries_successful += queries_used
        
        total_queries += queries_used
        total_reward += episode_reward
        
        # Record detailed results
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
    
    # Calculate evaluation metrics
    success_rate = success_count / len(test_examples) if test_examples else 0.0
    avg_queries = total_queries / len(test_examples) if test_examples else 0.0
    avg_queries_successful = queries_successful / success_count if success_count > 0 else 0.0
    avg_reward = total_reward / len(test_examples) if test_examples else 0.0
    
    results = {
        'success_rate': success_rate,
        'avg_queries': avg_queries,
        'avg_queries_successful': avg_queries_successful,
        'avg_reward': avg_reward,
        'total_samples': len(test_examples),
        'successful_attacks': success_count,
        'total_queries': total_queries,
        'queries_successful': queries_successful,
        'detailed_results': detailed_results
    }
    
    if logger:
        logger.logger.info(f"📊 Cross-LLM generalization evaluation results:")
        logger.logger.info(f"  Success rate (ASR): {success_rate:.3f}")
        logger.logger.info(f"  Average queries: {avg_queries:.3f}")
        logger.logger.info(f"  Average queries (successful): {avg_queries_successful:.3f}")
        logger.logger.info(f"  Average reward: {avg_reward:.3f}")
        logger.logger.info(f"  Successful attacks: {success_count}/{len(test_examples)}")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="RainbowDQN Cross-LLM Generalization Evaluation")
    parser.add_argument("--benchmark", type=str, default="arena_hard", 
                       choices=["alpaca_eval", "arena_hard", "code_judge_bench"],
                       help="Benchmark to use")
    parser.add_argument("--max_samples", default="full", help="Number of test samples, use 'full' for all data")
    parser.add_argument("--judge_model_path", type=str, default="", help="Judge model path (target LLM, required for local judges)")
    parser.add_argument("--judge_type", type=str, default="qwen", 
                       choices=["qwen", "llama", "gemma", "mistral", "glm", "online"],
                       help="Judge type")
    parser.add_argument("--rainbowdqn_model_path", type=str, required=True, help="RainbowDQN model path")
    parser.add_argument("--rainbowdqn_config_path", type=str, required=True, help="RainbowDQN config file path")
    parser.add_argument("--max_queries", type=int, default=5, help="Maximum number of queries")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    
    # Environment reward parameters
    parser.add_argument("--success_reward", type=float, default=28.0, help="Success attack reward")
    parser.add_argument("--query_penalty", type=float, default=0.55, help="Query penalty")
    parser.add_argument("--diversity_bonus", type=float, default=1.8, help="Diversity bonus")
    parser.add_argument("--efficiency_bonus", type=float, default=3.5, help="Efficiency bonus")
    parser.add_argument("--confidence_threshold", type=float, default=0.68, help="Confidence threshold")
    parser.add_argument("--online_api_key", type=str, default=None, help="API key for OnlineJudge (fallback to env)")
    parser.add_argument("--online_base_url", type=str, default=None, help="Base URL for OnlineJudge")
    parser.add_argument("--online_model", type=str, default="gpt-4o-mini", help="Target online model name")
    parser.add_argument("--online_temperature", type=float, default=0.1, help="Sampling temperature for OnlineJudge")
    parser.add_argument("--online_max_tokens", type=int, default=32, help="Max tokens for OnlineJudge completion")
    parser.add_argument("--online_timeout", type=int, default=30, help="Request timeout for OnlineJudge")
    
    args = parser.parse_args()
    
    # Set random seed
    import random
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize logger
    logger = HydraLogger(log_dir=args.output_dir, results_dir=args.output_dir)
    
    logger.logger.info("🚀 Starting RainbowDQN Cross-LLM Generalization Evaluation")
    logger.logger.info(f"Benchmark: {args.benchmark}")
    target_model_info = args.judge_model_path or args.online_model
    logger.logger.info(f"Judge model (target LLM): {target_model_info}")
    logger.logger.info(f"Judge type: {args.judge_type}")
    logger.logger.info(f"RainbowDQN model: {args.rainbowdqn_model_path}")
    logger.logger.info(f"Test samples: {args.max_samples}")
    logger.logger.info(f"Max queries: {args.max_queries}")
    
    # Load test data
    test_examples = load_benchmark_data(args.benchmark, args.max_samples, args.random_seed)
    if not test_examples:
        logger.logger.error("❌ Failed to load test data")
        return
    
    logger.logger.info(f"✅ Loaded {len(test_examples)} test samples")
    
    # Create attack methods
    attacks = get_attack_methods()
    logger.logger.info(f"✅ Loaded {len(attacks)} attack methods")
    
    # Create Judge (target model)
    try:
        if args.judge_type == "qwen":
            if not args.judge_model_path:
                raise ValueError("judge_model_path is required for qwen judge")
            judge = create_qwen_judge(args.judge_model_path)
        elif args.judge_type == "llama":
            if not args.judge_model_path:
                raise ValueError("judge_model_path is required for llama judge")
            judge = create_llama_judge(args.judge_model_path)
        elif args.judge_type == "gemma":
            if not args.judge_model_path:
                raise ValueError("judge_model_path is required for gemma judge")
            # Check if it's 1B model (text-only)
            model_path_lower = args.judge_model_path.lower()
            if "1b" in model_path_lower or "gemma-3-1b" in model_path_lower:
                judge = create_gemma_judge_1b(args.judge_model_path)
            else:
                judge = create_gemma_judge(args.judge_model_path)
        elif args.judge_type == "mistral":
            if not args.judge_model_path:
                raise ValueError("judge_model_path is required for mistral judge")
            judge = create_mistral_judge(args.judge_model_path)
        elif args.judge_type == "glm":
            if not args.judge_model_path:
                raise ValueError("judge_model_path is required for glm judge")
            judge = create_glm_judge(args.judge_model_path)
        elif args.judge_type == "online":
            judge = create_online_judge(
                api_key=args.online_api_key,
                base_url=args.online_base_url,
                model=args.online_model,
                temperature=args.online_temperature,
                max_tokens=args.online_max_tokens,
                timeout=args.online_timeout,
            )
        logger.logger.info("✅ Judge initialized successfully")
    except Exception as e:
        logger.logger.error(f"❌ Judge initialization failed: {e}")
        return
    
    # Load RainbowDQN agent
    try:
        agent = load_rainbowdqn_agent(args.rainbowdqn_model_path, args.rainbowdqn_config_path, logger)
        logger.logger.info("✅ RainbowDQN agent loaded successfully")
    except Exception as e:
        logger.logger.error(f"❌ RainbowDQN agent loading failed: {e}")
        return
    
    # Start evaluation
    logger.logger.info("🎯 Starting cross-LLM generalization evaluation...")
    start_time = time.time()
    
    results = evaluate_cross_llm_performance(
        agent, test_examples, attacks, judge, args.max_queries,
        args.success_reward, args.query_penalty, args.diversity_bonus,
        args.efficiency_bonus, args.confidence_threshold, logger
    )
    
    evaluation_time = time.time() - start_time
    logger.logger.info(f"🎉 Evaluation completed!")
    logger.logger.info(f"⏱️  Evaluation time: {evaluation_time:.2f} seconds")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"cross_llm_evaluation_{args.benchmark}_{args.judge_type}_{timestamp}.json")
    
    # Prepare results to save (excluding detailed results to save space)
    save_results = {
        'benchmark': args.benchmark,
        'source_model': os.path.basename(args.rainbowdqn_model_path),
        'target_judge_model': target_model_info,
        'target_judge_type': args.judge_type,
        'rainbowdqn_model_path': args.rainbowdqn_model_path,
        'rainbowdqn_config_path': args.rainbowdqn_config_path,
        'max_samples': args.max_samples,
        'max_queries': args.max_queries,
        'evaluation_time': evaluation_time,
        'success_rate': results['success_rate'],
        'avg_queries': results['avg_queries'],
        'avg_queries_successful': results['avg_queries_successful'],
        'avg_reward': results['avg_reward'],
        'total_samples': results['total_samples'],
        'successful_attacks': results['successful_attacks'],
        'total_queries': results['total_queries'],
        'queries_successful': results['queries_successful'],
        'env_params': {
            'success_reward': args.success_reward,
            'query_penalty': args.query_penalty,
            'diversity_bonus': args.diversity_bonus,
            'efficiency_bonus': args.efficiency_bonus,
            'confidence_threshold': args.confidence_threshold
        }
    }
    
    # Convert numpy types
    save_results = convert_numpy_types(save_results)
    
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    logger.logger.info(f"💾 Results saved to: {results_file}")
    
    # Output final results
    logger.logger.info("=" * 80)
    logger.logger.info("📊 Cross-LLM Generalization Evaluation Results")
    logger.logger.info("=" * 80)
    logger.logger.info(f"Benchmark: {args.benchmark}")
    logger.logger.info(f"Source model: {os.path.basename(args.rainbowdqn_model_path)}")
    logger.logger.info(f"Target Judge model: {target_model_info} ({args.judge_type})")
    logger.logger.info(f"Success rate (ASR): {results['success_rate']:.3f}")
    logger.logger.info(f"Average queries: {results['avg_queries']:.3f}")
    logger.logger.info(f"Average queries (successful): {results['avg_queries_successful']:.3f}")
    logger.logger.info(f"Average reward: {results['avg_reward']:.3f}")
    logger.logger.info(f"Successful attacks: {results['successful_attacks']}/{results['total_samples']}")
    logger.logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    main()

