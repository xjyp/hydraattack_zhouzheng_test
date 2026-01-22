#!/usr/bin/env python3
"""
Find the data source for Table 5 (strategy_usage) in PAPER_dsej.tex
查找PAPER_dsej.tex中Table 5 (strategy_usage)的数据来源
"""

import json
from pathlib import Path
from collections import defaultdict

# Target data from the LaTeX table
TABLE_DATA = {
    'Gemma-3-4B': {
        'Arena Hard': {
            'AuthorityAttack': (152, 70.4),
            'PositionAttack': (43, 44.2),
            'PromptInjectionAttack': (23, 39.1)
        },
        'Alpaca Eval': {
            'FlipAttackFCS': (161, 31.1),
            'MarkerInjectionAttack': (153, 55.6),
            'PositionAttack': (28, 21.4)
        },
        'Code Judge Bench': {
            'AuthorityAttack': (180, 70.0),
            'CoTPoisoningAttack': (178, 73.6),
            'PositionAttack': (176, 68.8)
        }
    },
    'Mistral-7B': {
        'Arena Hard': {
            'PromptInjectionAttack': (301, 29.2),
            'PositionAttack': (49, 77.6),
            'UncertaintyAttack': (35, 8.6)
        },
        'Alpaca Eval': {
            'PositionAttack': (96, 77.1),
            'FormattingAttack': (95, 20.0),
            'MarkerInjectionAttack': (54, 63.0)
        },
        'Code Judge Bench': {
            'AuthorityAttack': (398, 53.8),
            'MarkerInjectionAttack': (147, 84.4),
            'PositionAttack': (101, 46.5)
        }
    }
}

def find_data_sources():
    """Find potential data sources matching the table."""
    results_dirs = [
        Path('results_rainbowdqn'),
        Path('results_rainbowdqn_ablation'),
        Path('results_rainbowdqn_crossllm'),
    ]
    
    matches = defaultdict(list)
    
    for results_dir in results_dirs:
        if not results_dir.exists():
            continue
        
        for stats_file in results_dir.rglob('attack_usage_stats_test.json'):
            try:
                with open(stats_file, 'r') as f:
                    data = json.load(f)
                
                # Determine judge and benchmark from path
                path_str = str(stats_file).lower()
                judge = None
                benchmark = None
                
                if 'gemma' in path_str:
                    judge = 'Gemma-3-4B'
                elif 'mistral' in path_str:
                    judge = 'Mistral-7B'
                else:
                    continue
                
                if 'arena_hard' in path_str:
                    benchmark = 'Arena Hard'
                elif 'alpaca_eval' in path_str:
                    benchmark = 'Alpaca Eval'
                elif 'code_judge_bench' in path_str:
                    benchmark = 'Code Judge Bench'
                else:
                    continue
                
                if judge not in TABLE_DATA or benchmark not in TABLE_DATA[judge]:
                    continue
                
                # Check matches
                match_count = 0
                match_details = []
                
                for attack_key, (target_count, target_asr) in TABLE_DATA[judge][benchmark].items():
                    attack_data = data.get(attack_key, {})
                    if isinstance(attack_data, dict):
                        count = attack_data.get('count', 0)
                        success_rate = attack_data.get('success_rate', 0) * 100
                        
                        count_match = (count == target_count)
                        asr_match = abs(success_rate - target_asr) < 0.2
                        
                        if count_match and asr_match:
                            match_count += 1
                        
                        match_details.append({
                            'attack': attack_key,
                            'count': count,
                            'target_count': target_count,
                            'asr': success_rate,
                            'target_asr': target_asr,
                            'count_match': count_match,
                            'asr_match': asr_match
                        })
                
                if match_count > 0:
                    matches[(judge, benchmark)].append({
                        'file': str(stats_file),
                        'matches': match_count,
                        'details': match_details
                    })
            except Exception as e:
                pass
    
    return matches

def print_report(matches):
    """Print a report of findings."""
    print("="*100)
    print("Data Source Search Report for Table 5 (strategy_usage)")
    print("="*100)
    print()
    
    if not matches:
        print("No matching data found. The table data might be:")
        print("1. From a specific experiment run not found in current directories")
        print("2. Manually aggregated from multiple sources")
        print("3. From a different results directory")
        return
    
    print("Found potential data sources (sorted by match count):")
    print()
    
    # Sort by match count
    sorted_matches = []
    for key, file_list in matches.items():
        for item in file_list:
            sorted_matches.append((key, item))
    
    sorted_matches.sort(key=lambda x: x[1]['matches'], reverse=True)
    
    for (judge, benchmark), item in sorted_matches:
        print(f"Judge: {judge}, Benchmark: {benchmark}")
        print(f"  File: {item['file']}")
        print(f"  Matches: {item['matches']}/3")
        print(f"  Details:")
        for detail in item['details']:
            status = "✓" if detail['count_match'] and detail['asr_match'] else "✗"
            print(f"    {status} {detail['attack']}:")
            print(f"      Count: {detail['count']} (target: {detail['target_count']}) {'MATCH' if detail['count_match'] else ''}")
            print(f"      ASR: {detail['asr']:.1f}% (target: {detail['target_asr']:.1f}%) {'MATCH' if detail['asr_match'] else ''}")
        print()
    
    # Check for full matches
    full_matches = [x for x in sorted_matches if x[1]['matches'] == 3]
    if full_matches:
        print("="*100)
        print("FULL MATCHES FOUND (all 3 attacks match):")
        print("="*100)
        for (judge, benchmark), item in full_matches:
            print(f"  {judge} - {benchmark}: {item['file']}")
    else:
        print("="*100)
        print("No full matches found. Closest matches shown above.")
        print("="*100)

def main():
    """Main function."""
    print("Searching for data sources...")
    matches = find_data_sources()
    print_report(matches)
    
    # Save results
    output = {}
    for (judge, benchmark), file_list in matches.items():
        key = f"{judge}_{benchmark}"
        output[key] = file_list
    
    with open('table_data_source_search.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: table_data_source_search.json")

if __name__ == "__main__":
    main()

