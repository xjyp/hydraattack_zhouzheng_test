#!/usr/bin/env python3
"""
Extract Top-3 attack strategy data matching the LaTeX table format.
从汇总文件中提取与LaTeX表格匹配的Top-3攻击策略数据
"""

import json
from collections import defaultdict
from typing import Dict, List, Tuple

def load_summary():
    """Load summary file."""
    with open('results_rainbowdqn/rl_generation_rainbowdqn_test_20251122_144706/attack_usage_summary_test.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_judge_model(dataset_name: str) -> str:
    """Extract judge model from dataset name."""
    name_lower = dataset_name.lower()
    if 'gemma' in name_lower:
        return 'Gemma-3-4B'
    elif 'mistral' in name_lower:
        return 'Mistral-7B'
    elif 'llama' in name_lower:
        return 'Llama-3.1-8B'
    elif 'glm' in name_lower:
        return 'GLM-4-9B'
    elif 'qwen' in name_lower:
        return 'Qwen'  # Not in table, but exists
    return None

def extract_benchmark(dataset_name: str) -> str:
    """Extract benchmark from dataset name."""
    name_lower = dataset_name.lower()
    if 'arena_hard' in name_lower or 'arena-hard' in name_lower:
        return 'Arena Hard'
    elif 'alpaca_eval' in name_lower or 'alpaca-eval' in name_lower:
        return 'Alpaca Eval'
    elif 'code_judge_bench' in name_lower or 'codejudgebench' in name_lower:
        return 'Code Judge Bench'
    return None

def get_top3_attacks(attacks: Dict) -> List[Tuple[str, int, float]]:
    """Get top 3 attacks by count."""
    attack_list = []
    for attack_name, attack_data in attacks.items():
        if isinstance(attack_data, dict):
            count = attack_data.get('count', 0)
            success_rate = attack_data.get('success_rate', 0.0)
            # Convert to percentage
            asr = success_rate * 100 if success_rate <= 1.0 else success_rate
            
            # Map attack name
            attack_display = attack_name.replace('Attack', '').replace('FCS', ' FCS').replace('FWO', ' FWO').replace('FCW', ' FCW')
            if attack_display == 'FlipAttack FCS':
                attack_display = 'FlipAttack FCS'
            elif attack_display == 'Position':
                attack_display = 'Position Attack'
            elif attack_display == 'PromptInjection':
                attack_display = 'Prompt Injection'
            elif attack_display == 'MarkerInjection':
                attack_display = 'Marker Injection'
            elif attack_display == 'CoTPoisoning':
                attack_display = 'CoT Poisoning'
            elif attack_display == 'Uncertainty':
                attack_display = 'Uncertainty Attack'
            elif attack_display == 'Formatting':
                attack_display = 'Formatting Attack'
            elif attack_display == 'Authority':
                attack_display = 'Authority Attack'
            elif attack_display == 'Emoji':
                attack_display = 'Emoji Attack'
            elif attack_display == 'Distractor':
                attack_display = 'Distractor Attack'
            elif attack_display == 'Unicode':
                attack_display = 'Unicode Attack'
            
            attack_list.append((attack_display, count, asr))
    
    # Sort by count descending
    attack_list.sort(key=lambda x: x[1], reverse=True)
    return attack_list[:3]

def aggregate_by_judge_benchmark(summary_data: Dict) -> Dict:
    """Aggregate data by judge model and benchmark."""
    results = defaultdict(lambda: defaultdict(list))
    
    for dataset_result in summary_data.get('dataset_results', []):
        dataset_name = dataset_result['dataset_name']
        judge_model = extract_judge_model(dataset_name)
        benchmark = extract_benchmark(dataset_name)
        
        if not judge_model or not benchmark:
            continue
        
        # Get top 3 attacks for this dataset
        top3 = get_top3_attacks(dataset_result.get('attacks', {}))
        if top3:
            results[judge_model][benchmark].append(top3)
    
    # For each judge_model/benchmark combination, aggregate across multiple runs
    # Sum counts and calculate weighted ASR
    aggregated = defaultdict(lambda: defaultdict(list))
    
    for judge_model in results:
        for benchmark in results[judge_model]:
            # Aggregate all attacks across multiple runs
            all_attacks = defaultdict(lambda: {'count': 0, 'success_count': 0})
            
            for top3_list in results[judge_model][benchmark]:
                for attack_name, count, asr in top3_list:
                    all_attacks[attack_name]['count'] += count
                    # Estimate success_count from ASR and count
                    success_count = int(count * (asr / 100.0))
                    all_attacks[attack_name]['success_count'] += success_count
            
            # Convert to list and calculate ASR
            attack_list = []
            for attack_name, data in all_attacks.items():
                count = data['count']
                success_count = data['success_count']
                asr = (success_count / count * 100) if count > 0 else 0.0
                attack_list.append((attack_name, count, asr))
            
            # Sort and get top 3
            attack_list.sort(key=lambda x: x[1], reverse=True)
            aggregated[judge_model][benchmark] = attack_list[:3]
    
    return dict(aggregated)

def main():
    """Main function."""
    summary_data = load_summary()
    aggregated = aggregate_by_judge_benchmark(summary_data)
    
    # Print results
    judge_models = ['Gemma-3-4B', 'Mistral-7B', 'GLM-4-9B', 'Llama-3.1-8B']
    benchmarks = ['Arena Hard', 'Alpaca Eval', 'Code Judge Bench']
    
    print("\n" + "="*140)
    print("Top-3 Attack Strategy Usage Statistics by Benchmark and Judge Model")
    print("="*140)
    
    for judge_model in judge_models:
        if judge_model not in aggregated:
            continue
        
        print(f"\nJudge Model: {judge_model}")
        print("-" * 140)
        print(f"{'Rank':<6}", end="")
        for benchmark in benchmarks:
            print(f"{benchmark:^45}", end="")
        print()
        print(f"{'':<6}", end="")
        for benchmark in benchmarks:
            print(f"{'Strategy':<25} {'Count':<10} {'ASR':<10}", end="")
        print()
        print("-" * 140)
        
        for rank in range(1, 4):
            print(f"{rank:<6}", end="")
            for benchmark in benchmarks:
                if benchmark in aggregated[judge_model] and len(aggregated[judge_model][benchmark]) >= rank:
                    attack_name, count, asr = aggregated[judge_model][benchmark][rank - 1]
                    print(f"{attack_name:<25} {count:<10} {asr:>9.1f}%", end="")
                else:
                    print(f"{'N/A':<25} {'N/A':<10} {'N/A':<10}", end="")
            print()
        print()
    
    # Save to JSON
    output_data = {}
    for judge_model in aggregated:
        output_data[judge_model] = {}
        for benchmark in aggregated[judge_model]:
            output_data[judge_model][benchmark] = [
                [name, count, asr] for name, count, asr in aggregated[judge_model][benchmark]
            ]
    
    with open('table_data_from_summary.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nData saved to: table_data_from_summary.json")

if __name__ == "__main__":
    main()

