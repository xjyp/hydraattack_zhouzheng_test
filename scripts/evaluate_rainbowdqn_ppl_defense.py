#!/usr/bin/env python3
"""
RainbowDQN evaluation with PPL Defense
Load trained RainbowDQN models and evaluate with perplexity-based defense
"""

import argparse
import sys
import json
import time
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_types import PairwiseExample
from attacks import (
    FlipAttackFWO, FlipAttackFCW, FlipAttackFCS, UncertaintyAttack, PositionAttack, DistractorAttack,
    PromptInjectionAttack, MarkerInjectionAttack, FormattingAttack,
    AuthorityAttack, UnicodeAttack, CoTPoisoningAttack, EmojiAttack
)
from evaluation.gemma_judge import create_gemma_judge
try:
    from defense.ppl_defense import PPLDefense, PPLDefendedJudge
except ImportError:
    from src.defense.ppl_defense import PPLDefense, PPLDefendedJudge
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


def load_benchmark_data(benchmark: str, max_samples="full", random_seed: int = 42) -> List[PairwiseExample]:
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
    
    # Find the first linear layer's weight
    first_layer_weight = None
    for key, value in state_dict.items():
        if 'weight' in key and len(value.shape) == 2:
            if 'feature.0.weight' in key or (key.startswith('0.') and 'weight' in key):
                first_layer_weight = value
                break
    
    if first_layer_weight is not None:
        state_dim = first_layer_weight.shape[1]
    else:
        state_dim = 512
    
    if logger:
        logger.logger.info(f"   Inferred state dimension: {state_dim}")
    
    # Create RainbowDQN agent
    agent = RainbowDQNAgent(
        state_dim=state_dim,
        action_dim=total_actions,
        hidden_dim=rainbowdqn_params.get('hidden_dim', 384),
        learning_rate=rainbowdqn_params.get('learning_rate', 0.0001),
        gamma=rainbowdqn_params.get('gamma', 0.99),
        epsilon=0.0,
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
    
    return agent


def evaluate_with_ppl_defense(
    agent,
    test_examples: List[PairwiseExample],
    attacks: List[Any],
    judge,
    ppl_defense: PPLDefense,
    max_queries: int = 10,
    success_reward: float = 20.0,
    query_penalty: float = 0.5,
    diversity_bonus: float = 1.0,
    efficiency_bonus: float = 2.0,
    confidence_threshold: float = 0.7,
    logger: HydraLogger = None
) -> Dict[str, float]:
    """Evaluate with PPL defense"""
    
    if logger:
        logger.logger.info("🔍 Starting evaluation with PPL defense...")
        logger.logger.info(f"   Target Judge: {judge.__class__.__name__}")
        logger.logger.info(f"   Test samples: {len(test_examples)}")
        logger.logger.info(f"   PPL threshold: {ppl_defense.threshold:.2f}" if ppl_defense.threshold else "   PPL threshold: Not set")
    
    # Wrap judge with PPL defense
    defended_judge = PPLDefendedJudge(judge, ppl_defense, filter_mode="instruction")
    
    success_count = 0
    total_queries = 0
    queries_successful = 0
    total_reward = 0.0
    detailed_results = []
    
    for i, example in enumerate(test_examples):
        if logger and i % 10 == 0:
            logger.logger.info(f"  Evaluation progress: {i+1}/{len(test_examples)}")
        
        # Get original preference
        try:
            original_response = judge.judge_pairwise(example)
            original_preference = original_response.preference
            original_confidence = original_response.confidence
        except Exception as e:
            if logger:
                logger.logger.warning(f"Failed to get original preference: {e}")
            continue
        
        # Set original preference in defended judge so it returns original preference when filtered
        defended_judge.set_original_preference(original_preference)
        
        # Create environment with defended judge
        env = EnhancedHydraAttackEnv(
            examples=[example],
            attacks=attacks,
            judge=defended_judge,  # Use defended judge
            max_queries=max_queries,
            success_reward=success_reward,
            query_penalty=query_penalty,
            diversity_bonus=diversity_bonus,
            efficiency_bonus=efficiency_bonus,
            confidence_threshold=confidence_threshold
        )
        
        # Reset environment
        state, _ = env.reset(options={
            'use_specific_sample': True,
            'sample_idx': 0,
            'original_preference': original_preference,
            'original_confidence': original_confidence
        })
        episode_reward = 0
        attack_sequence = []
        
        # Execute attack sequence
        for step in range(max_queries):
            action_mask = env.get_action_mask()
            action = agent.select_action(state, training=False, action_mask=action_mask)
            next_state, reward, done, truncated, info = env.step(action)
            
            attack_info = {
                'step': step + 1,
                'action': int(action),
                'reward': float(reward),
                'done': done,
                'truncated': truncated,
                'query_count': info.get('query_count', step + 1),  # Use environment's query_count (includes filtered queries)
                'preference': info.get('current_preference', -1),
                'original_preference': info.get('original_preference', -1),
                'attack_successful': (info.get('current_preference', -1) != info.get('original_preference', -1)) if info.get('original_preference') is not None else False
            }
            attack_sequence.append(attack_info)
            
            episode_reward += reward
            
            if done or truncated:
                break
            
            state = next_state
        
        # Check attack result
        original_preference = env.original_preference
        final_preference = info.get('current_preference', original_preference)
        success = info.get('success', False)
        queries_used = info.get('query_count', 0)  # Use environment's query_count (includes filtered queries)
        
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
    
    # Get filter stats
    filter_stats = defended_judge.get_filter_stats()
    
    results = {
        'success_rate': success_rate,
        'avg_queries': avg_queries,
        'avg_queries_successful': avg_queries_successful,
        'avg_reward': avg_reward,
        'total_samples': len(test_examples),
        'successful_attacks': success_count,
        'total_queries': total_queries,
        'queries_successful': queries_successful,
        'ppl_defense_stats': filter_stats,
        'detailed_results': detailed_results
    }
    
    if logger:
        logger.logger.info(f"📊 Evaluation results with PPL defense:")
        logger.logger.info(f"  Success rate (ASR): {success_rate:.3f}")
        logger.logger.info(f"  Average queries: {avg_queries:.3f}")
        logger.logger.info(f"  Average queries (successful): {avg_queries_successful:.3f}")
        logger.logger.info(f"  Average reward: {avg_reward:.3f}")
        logger.logger.info(f"  Successful attacks: {success_count}/{len(test_examples)}")
        logger.logger.info(f"  PPL Defense filter rate: {filter_stats['filter_rate']:.3f}")
        logger.logger.info(f"  PPL Defense filtered: {filter_stats['filtered_count']}/{filter_stats['total_judgments']}")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="RainbowDQN Evaluation with PPL Defense")
    parser.add_argument("--benchmark", type=str, required=True,
                       choices=["alpaca_eval", "arena_hard", "code_judge_bench"],
                       help="Benchmark to use")
    parser.add_argument("--max_samples", default="full", help="Number of test samples, use 'full' for all data")
    parser.add_argument("--judge_model_path", type=str, required=True, help="Judge model path")
    parser.add_argument("--judge_type", type=str, default="gemma",
                       choices=["gemma"],
                       help="Judge type (only gemma supported for now)")
    parser.add_argument("--rainbowdqn_model_path", type=str, required=True, help="RainbowDQN model path")
    parser.add_argument("--rainbowdqn_config_path", type=str, required=True, help="RainbowDQN config file path")
    parser.add_argument("--ppl_model_path", type=str, default="/share/disk/llm_cache/gpt2",
                       help="PPL model path (GPT-2)")
    parser.add_argument("--ppl_threshold_multiplier", type=float, default=2.0,
                       help="PPL threshold multiplier (c in T = μ + c * σ)")
    parser.add_argument("--ppl_threshold", type=float, default=None,
                       help="Direct PPL threshold (overrides multiplier)")
    parser.add_argument("--ppl_threshold_method", type=str, default="fpr_based",
                       choices=["mean_std", "robust", "percentile", "iqr", "fpr_based"],
                       help="Method for computing threshold: 'mean_std' (standard), 'robust' (median+MAD), 'percentile', 'iqr', 'fpr_based' (FPR-based, recommended)")
    parser.add_argument("--ppl_target_fpr", type=float, default=0.01,
                       help="Target False Positive Rate for 'fpr_based' method (default 0.01 = 1%%)")
    parser.add_argument("--ppl_calibration_samples", type=int, default=100,
                       help="Number of clean samples to use for threshold calibration")
    parser.add_argument("--max_queries", type=int, default=5, help="Maximum number of queries")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    
    # Environment reward parameters
    parser.add_argument("--success_reward", type=float, default=30.0, help="Success attack reward")
    parser.add_argument("--query_penalty", type=float, default=0.60, help="Query penalty")
    parser.add_argument("--diversity_bonus", type=float, default=2.0, help="Diversity bonus")
    parser.add_argument("--efficiency_bonus", type=float, default=4.0, help="Efficiency bonus")
    parser.add_argument("--confidence_threshold", type=float, default=0.70, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Set random seed
    import random
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize logger
    logger = HydraLogger(log_dir=args.output_dir, results_dir=args.output_dir)
    
    logger.logger.info("🚀 Starting RainbowDQN Evaluation with PPL Defense")
    logger.logger.info(f"Benchmark: {args.benchmark}")
    logger.logger.info(f"Judge model: {args.judge_model_path}")
    logger.logger.info(f"Judge type: {args.judge_type}")
    logger.logger.info(f"RainbowDQN model: {args.rainbowdqn_model_path}")
    logger.logger.info(f"PPL model: {args.ppl_model_path}")
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
    
    # Create Judge
    try:
        if args.judge_type == "gemma":
            judge = create_gemma_judge(args.judge_model_path)
        else:
            raise ValueError(f"Unsupported judge type: {args.judge_type}")
        logger.logger.info("✅ Judge initialized successfully")
    except Exception as e:
        logger.logger.error(f"❌ Judge initialization failed: {e}")
        return
    
    # Initialize PPL Defense
    try:
        ppl_defense = PPLDefense(
            ppl_model_path=args.ppl_model_path,
            device="cuda",
            threshold_multiplier=args.ppl_threshold_multiplier,
            threshold=args.ppl_threshold,
            threshold_method=args.ppl_threshold_method,
            target_fpr=args.ppl_target_fpr
        )
        
        # Compute threshold from clean data if not directly provided
        if args.ppl_threshold is None:
            logger.logger.info("Computing PPL threshold from clean data...")
            # Use test data as clean data for calibration
            mean_ppl, std_ppl, threshold = ppl_defense.compute_threshold_from_clean_data(
                test_examples,
                num_samples=args.ppl_calibration_samples
            )
            ppl_defense.set_threshold(threshold)
        else:
            ppl_defense.set_threshold(args.ppl_threshold)
        
        logger.logger.info("✅ PPL Defense initialized successfully")
    except Exception as e:
        logger.logger.error(f"❌ PPL Defense initialization failed: {e}")
        return
    
    # Load RainbowDQN agent
    try:
        agent = load_rainbowdqn_agent(args.rainbowdqn_model_path, args.rainbowdqn_config_path, logger)
        logger.logger.info("✅ RainbowDQN agent loaded successfully")
    except Exception as e:
        logger.logger.error(f"❌ RainbowDQN agent loading failed: {e}")
        return
    
    # Start evaluation
    logger.logger.info("🎯 Starting evaluation with PPL defense...")
    start_time = time.time()
    
    results = evaluate_with_ppl_defense(
        agent, test_examples, attacks, judge, ppl_defense, args.max_queries,
        args.success_reward, args.query_penalty, args.diversity_bonus,
        args.efficiency_bonus, args.confidence_threshold, logger
    )
    
    evaluation_time = time.time() - start_time
    logger.logger.info(f"🎉 Evaluation completed!")
    logger.logger.info(f"⏱️  Evaluation time: {evaluation_time:.2f} seconds")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"ppl_defense_evaluation_{args.benchmark}_{timestamp}.json")
    
    # Prepare results to save
    save_results = {
        'benchmark': args.benchmark,
        'judge_model': args.judge_model_path,
        'judge_type': args.judge_type,
        'rainbowdqn_model_path': args.rainbowdqn_model_path,
        'rainbowdqn_config_path': args.rainbowdqn_config_path,
        'ppl_model_path': args.ppl_model_path,
        'ppl_threshold': ppl_defense.threshold,
        'ppl_threshold_multiplier': args.ppl_threshold_multiplier,
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
        'ppl_defense_stats': results['ppl_defense_stats'],
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
    logger.logger.info("📊 PPL Defense Evaluation Results")
    logger.logger.info("=" * 80)
    logger.logger.info(f"Benchmark: {args.benchmark}")
    logger.logger.info(f"Success rate (ASR): {results['success_rate']:.3f}")
    logger.logger.info(f"Average queries: {results['avg_queries']:.3f}")
    logger.logger.info(f"Average queries (successful): {results['avg_queries_successful']:.3f}")
    logger.logger.info(f"PPL Defense filter rate: {results['ppl_defense_stats']['filter_rate']:.3f}")
    logger.logger.info(f"Successful attacks: {results['successful_attacks']}/{results['total_samples']}")
    logger.logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    main()

