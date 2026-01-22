#!/usr/bin/env python3
"""
Analyze attack strategy usage statistics from RL generation results.
统计各个攻击策略的调用次数和占比
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def load_attack_stats(stats_file: str) -> Dict:
    """Load attack usage statistics from JSON file."""
    if not os.path.exists(stats_file):
        return {}
    with open(stats_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_single_dataset(dataset_dir: str, use_test: bool = True) -> Dict:
    """
    Analyze attack usage statistics for a single dataset.
    
    Args:
        dataset_dir: Path to the dataset directory
        use_test: If True, use test stats; if False, use training stats
    
    Returns:
        Dictionary with attack strategy statistics
    """
    stats_file = os.path.join(dataset_dir, 
                              'attack_usage_stats_test.json' if use_test 
                              else 'attack_usage_stats_training.json')
    
    if not os.path.exists(stats_file):
        return None
    
    stats = load_attack_stats(stats_file)
    if not stats:
        return None
    
    # Calculate total count
    total_count = sum(attack_data.get('count', 0) for attack_data in stats.values())
    
    # Build result dictionary
    result = {
        'dataset_name': os.path.basename(dataset_dir),
        'total_count': total_count,
        'attacks': {}
    }
    
    for attack_name, attack_data in stats.items():
        count = attack_data.get('count', 0)
        percentage = (count / total_count * 100) if total_count > 0 else 0.0
        
        result['attacks'][attack_name] = {
            'count': count,
            'percentage': percentage,
            'success_count': attack_data.get('success_count', 0),
            'success_rate': attack_data.get('success_rate', 0.0),
            'avg_reward': attack_data.get('avg_reward', 0.0)
        }
    
    return result

def analyze_multiple_datasets(base_dir: str, use_test: bool = True) -> Dict:
    """
    Analyze attack usage statistics across multiple datasets.
    
    Args:
        base_dir: Base directory containing multiple dataset subdirectories
        use_test: If True, use test stats; if False, use training stats
    
    Returns:
        Aggregated statistics across all datasets
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return None
    
    # Aggregate statistics
    all_attack_counts = defaultdict(int)
    all_success_counts = defaultdict(int)
    total_calls = 0
    dataset_results = []
    
    # Process each subdirectory
    for subdir in sorted(base_path.iterdir()):
        if not subdir.is_dir():
            continue
        
        result = analyze_single_dataset(str(subdir), use_test)
        if result is None:
            continue
        
        dataset_results.append(result)
        total_calls += result['total_count']
        
        # Aggregate attack counts
        for attack_name, attack_data in result['attacks'].items():
            all_attack_counts[attack_name] += attack_data['count']
            all_success_counts[attack_name] += attack_data['success_count']
    
    # Calculate percentages
    attack_percentages = {}
    for attack_name, count in all_attack_counts.items():
        percentage = (count / total_calls * 100) if total_calls > 0 else 0.0
        success_count = all_success_counts[attack_name]
        success_rate = (success_count / count * 100) if count > 0 else 0.0
        
        attack_percentages[attack_name] = {
            'count': count,
            'percentage': percentage,
            'success_count': success_count,
            'success_rate': success_rate
        }
    
    return {
        'total_datasets': len(dataset_results),
        'total_calls': total_calls,
        'aggregated_stats': attack_percentages,
        'dataset_results': dataset_results
    }

def print_statistics(results: Dict, use_test: bool = True):
    """Print statistics in a formatted way."""
    if results is None:
        print("No statistics found.")
        return
    
    stats_type = "Test" if use_test else "Training"
    print(f"\n{'='*80}")
    print(f"Attack Strategy Usage Statistics ({stats_type})")
    print(f"{'='*80}")
    print(f"Total Datasets: {results['total_datasets']}")
    print(f"Total Attack Calls: {results['total_calls']}")
    print(f"\n{'='*80}")
    print(f"{'Attack Strategy':<30} {'Count':<12} {'Percentage':<12} {'Success Count':<15} {'Success Rate':<12}")
    print(f"{'-'*80}")
    
    # Sort by count (descending)
    sorted_attacks = sorted(
        results['aggregated_stats'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    
    for attack_name, stats in sorted_attacks:
        if stats['count'] > 0:  # Only show attacks that were used
            print(f"{attack_name:<30} {stats['count']:<12} {stats['percentage']:>10.2f}%  "
                  f"{stats['success_count']:<15} {stats['success_rate']:>10.2f}%")
    
    print(f"{'='*80}\n")
    
    # Print per-dataset breakdown
    print(f"\nPer-Dataset Breakdown:")
    print(f"{'='*80}")
    for dataset_result in results['dataset_results']:
        print(f"\nDataset: {dataset_result['dataset_name']}")
        print(f"Total Calls: {dataset_result['total_count']}")
        print(f"{'Attack Strategy':<30} {'Count':<12} {'Percentage':<12}")
        print(f"{'-'*60}")
        
        sorted_dataset_attacks = sorted(
            dataset_result['attacks'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        for attack_name, attack_data in sorted_dataset_attacks:
            if attack_data['count'] > 0:
                print(f"{attack_name:<30} {attack_data['count']:<12} {attack_data['percentage']:>10.2f}%")

def main():
    """Main function."""
    import sys
    
    # Configuration
    # You can modify these parameters directly in the script
    BASE_DIR = "results_rainbowdqn/rl_generation_rainbowdqn_test_20251122_144706"
    USE_TEST = True  # Set to False to analyze training statistics
    
    # Allow command line override
    if len(sys.argv) > 1:
        BASE_DIR = sys.argv[1]
    if len(sys.argv) > 2:
        USE_TEST = sys.argv[2].lower() in ['true', '1', 'yes', 'test']
    
    # Analyze statistics
    results = analyze_multiple_datasets(BASE_DIR, USE_TEST)
    
    if results:
        print_statistics(results, USE_TEST)
        
        # Save results to JSON
        output_file = os.path.join(BASE_DIR, f"attack_usage_summary_{'test' if USE_TEST else 'training'}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print("Failed to analyze statistics.")

if __name__ == "__main__":
    main()


