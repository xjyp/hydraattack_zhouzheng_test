#!/usr/bin/env python3
"""
Find all available judge model data and extract Top-3 attack strategy usage statistics.
查找所有可用的judge model数据并提取Top-3攻击策略使用统计
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Judge model name mapping
JUDGE_MODEL_MAPPING = {
    'gemma': {
        '1b': 'Gemma-3-1B',
        '4b': 'Gemma-3-4B',
        '12b': 'Gemma-3-12B'
    },
    'mistral': 'Mistral-7B',
    'llama': 'Llama-3.1-8B',
    'glm': 'GLM-4-9B'
}

# Benchmark name mapping
BENCHMARK_MAPPING = {
    'arena_hard': 'Arena Hard',
    'alpaca_eval': 'Alpaca Eval',
    'code_judge_bench': 'Code Judge Bench'
}

# Attack strategy name mapping (from JSON keys to display names)
ATTACK_NAME_MAPPING = {
    'FlipAttackFCS': 'FlipAttack FCS',
    'FlipAttackFWO': 'FlipAttack FWO',
    'FlipAttackFCW': 'FlipAttack FCW',
    'UncertaintyAttack': 'Uncertainty Attack',
    'PositionAttack': 'Position Attack',
    'DistractorAttack': 'Distractor Attack',
    'PromptInjectionAttack': 'Prompt Injection',
    'MarkerInjectionAttack': 'Marker Injection',
    'FormattingAttack': 'Formatting Attack',
    'AuthorityAttack': 'Authority Attack',
    'UnicodeAttack': 'Unicode Attack',
    'CoTPoisoningAttack': 'CoT Poisoning',
    'EmojiAttack': 'Emoji Attack'
}

def extract_judge_model_from_path(path: str) -> Optional[str]:
    """Extract judge model name from directory path."""
    path_lower = path.lower()
    
    # Check for Gemma variants
    if 'gemma' in path_lower:
        if '1b' in path_lower or '1_b' in path_lower:
            return 'Gemma-3-1B'
        elif '12b' in path_lower or '12_b' in path_lower:
            return 'Gemma-3-12B'
        else:
            return 'Gemma-3-4B'  # Default to 4B
    
    # Check for other models
    if 'mistral' in path_lower:
        return 'Mistral-7B'
    elif 'llama' in path_lower:
        return 'Llama-3.1-8B'
    elif 'glm' in path_lower:
        return 'GLM-4-9B'
    
    return None

def extract_benchmark_from_path(path: str) -> Optional[str]:
    """Extract benchmark name from directory path."""
    path_lower = path.lower()
    
    if 'arena_hard' in path_lower or 'arena-hard' in path_lower:
        return 'Arena Hard'
    elif 'alpaca_eval' in path_lower or 'alpaca-eval' in path_lower:
        return 'Alpaca Eval'
    elif 'code_judge_bench' in path_lower or 'codejudgebench' in path_lower:
        return 'Code Judge Bench'
    
    return None

def load_attack_stats(stats_file: str) -> Optional[Dict]:
    """Load attack usage statistics from JSON file."""
    if not os.path.exists(stats_file):
        return None
    
    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Return test stats if available
            if 'test' in data:
                return data['test']
            return data
    except Exception as e:
        print(f"Error loading {stats_file}: {e}")
        return None

def get_top3_attacks(stats: Dict) -> List[Tuple[str, int, float]]:
    """Get top 3 attacks by count, returning (name, count, asr)."""
    attacks = []
    
    for attack_name, attack_data in stats.items():
        if isinstance(attack_data, dict):
            count = attack_data.get('count', 0)
            success_rate = attack_data.get('success_rate', 0.0)
            # Convert to percentage
            asr = success_rate * 100 if success_rate <= 1.0 else success_rate
            
            # Map attack name to display name
            display_name = ATTACK_NAME_MAPPING.get(attack_name, attack_name)
            attacks.append((display_name, count, asr))
    
    # Sort by count (descending) and return top 3
    attacks.sort(key=lambda x: x[1], reverse=True)
    return attacks[:3]

def find_all_judge_model_data(base_dir: str = "results_rainbowdqn") -> Dict:
    """Find all judge model data and extract statistics."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return {}
    
    # Structure: {judge_model: {benchmark: top3_attacks}}
    results = defaultdict(lambda: defaultdict(list))
    
    # Find all attack_usage_stats_test.json files
    stats_files = list(base_path.rglob("attack_usage_stats_test.json"))
    
    print(f"Found {len(stats_files)} stats files")
    
    for stats_file in stats_files:
        # Extract judge model and benchmark from path
        judge_model = extract_judge_model_from_path(str(stats_file))
        benchmark = extract_benchmark_from_path(str(stats_file))
        
        if not judge_model or not benchmark:
            continue
        
        # Load stats
        stats = load_attack_stats(str(stats_file))
        if not stats:
            continue
        
        # Get top 3 attacks
        top3 = get_top3_attacks(stats)
        if top3:
            results[judge_model][benchmark] = top3
            print(f"Found data: {judge_model} - {benchmark}: {len(top3)} attacks")
    
    return dict(results)

def print_results_table(results: Dict):
    """Print results in table format similar to the original table."""
    judge_models = sorted(results.keys())
    benchmarks = ['Arena Hard', 'Alpaca Eval', 'Code Judge Bench']
    
    print("\n" + "="*120)
    print("Top-3 Attack Strategy Usage Statistics by Benchmark and Judge Model")
    print("="*120)
    
    for judge_model in judge_models:
        print(f"\nJudge Model: {judge_model}")
        print("-" * 120)
        print(f"{'Rank':<6}", end="")
        for benchmark in benchmarks:
            print(f"{benchmark:^35}", end="")
        print()
        print(f"{'':<6}", end="")
        for benchmark in benchmarks:
            print(f"{'Strategy':<20} {'Count':<8} {'ASR':<7}", end="")
        print()
        print("-" * 120)
        
        # Print top 3 for each rank
        for rank in range(1, 4):
            print(f"{rank:<6}", end="")
            for benchmark in benchmarks:
                if benchmark in results[judge_model] and len(results[judge_model][benchmark]) >= rank:
                    attack_name, count, asr = results[judge_model][benchmark][rank - 1]
                    print(f"{attack_name:<20} {count:<8} {asr:>6.1f}%", end="")
                else:
                    print(f"{'N/A':<20} {'N/A':<8} {'N/A':<7}", end="")
            print()
        print()

def save_results_json(results: Dict, output_file: str = "judge_model_top3_stats.json"):
    """Save results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

def main():
    """Main function."""
    import sys
    
    # Configuration
    BASE_DIR = "results_rainbowdqn"
    
    # Allow command line override
    if len(sys.argv) > 1:
        BASE_DIR = sys.argv[1]
    
    # Find all judge model data
    results = find_all_judge_model_data(BASE_DIR)
    
    if results:
        print_results_table(results)
        save_results_json(results)
        
        # Print summary
        print(f"\nSummary:")
        print(f"Found data for {len(results)} judge models:")
        for judge_model in sorted(results.keys()):
            benchmarks = list(results[judge_model].keys())
            print(f"  - {judge_model}: {len(benchmarks)} benchmarks ({', '.join(benchmarks)})")
    else:
        print("No judge model data found.")

if __name__ == "__main__":
    main()

