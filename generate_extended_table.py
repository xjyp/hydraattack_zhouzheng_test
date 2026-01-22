#!/usr/bin/env python3
"""
Generate extended table with all available judge model data.
生成包含所有可用judge model数据的扩展表格
"""

import json
from typing import Dict, List

def load_stats():
    """Load statistics from JSON file."""
    with open('judge_model_top3_stats.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_latex_table(stats: Dict):
    """Generate LaTeX table format."""
    judge_models = ['Gemma-3-4B', 'Mistral-7B', 'GLM-4-9B', 'Llama-3.1-8B']
    benchmarks = ['Arena Hard', 'Alpaca Eval', 'Code Judge Bench']
    
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\small")
    print("\\caption{Top-3 Attack Strategy Usage Statistics by Benchmark and Judge Model}")
    print("\\label{tab:top3_attack_usage}")
    print("\\begin{tabular}{l|c|ccc|ccc|ccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{Judge Model} & \\multirow{2}{*}{Rank} & \\multicolumn{3}{c|}{Arena Hard} & \\multicolumn{3}{c|}{Alpaca Eval} & \\multicolumn{3}{c}{Code Judge Bench} \\\\")
    print("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}")
    print(" & & Strategy & Count $\\downarrow$ & ASR & Strategy & Count $\\downarrow$ & ASR & Strategy & Count $\\downarrow$ & ASR \\\\")
    print("\\midrule")
    
    for judge_model in judge_models:
        if judge_model not in stats:
            continue
        
        for rank in range(1, 4):
            row = [judge_model if rank == 1 else "", str(rank)]
            
            for benchmark in benchmarks:
                if benchmark in stats[judge_model] and len(stats[judge_model][benchmark]) >= rank:
                    attack_name, count, asr = stats[judge_model][benchmark][rank - 1]
                    # Clean attack name for LaTeX
                    attack_name_clean = attack_name.replace('&', '\\&')
                    row.extend([attack_name_clean, str(count), f"{asr:.1f}\\%"])
                else:
                    row.extend(["N/A", "N/A", "N/A"])
            
            print(" & ".join(row) + " \\\\")
        
        if judge_model != judge_models[-1]:
            print("\\midrule")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")

def generate_markdown_table(stats: Dict):
    """Generate Markdown table format."""
    judge_models = ['Gemma-3-4B', 'Mistral-7B', 'GLM-4-9B', 'Llama-3.1-8B']
    benchmarks = ['Arena Hard', 'Alpaca Eval', 'Code Judge Bench']
    
    print("\n## Top-3 Attack Strategy Usage Statistics by Benchmark and Judge Model\n")
    
    # Create header
    header = "| Judge Model | Rank |"
    separator = "|:---|:---:|"
    for benchmark in benchmarks:
        header += f" {benchmark} Strategy | {benchmark} Count ↓ | {benchmark} ASR |"
        separator += ":---:|:---:|:---:|"
    
    print(header)
    print(separator)
    
    for judge_model in judge_models:
        if judge_model not in stats:
            continue
        
        for rank in range(1, 4):
            row = [judge_model if rank == 1 else "", str(rank)]
            
            for benchmark in benchmarks:
                if benchmark in stats[judge_model] and len(stats[judge_model][benchmark]) >= rank:
                    attack_name, count, asr = stats[judge_model][benchmark][rank - 1]
                    row.extend([attack_name, str(count), f"{asr:.1f}%"])
                else:
                    row.extend(["N/A", "N/A", "N/A"])
            
            print("| " + " | ".join(row) + " |")

def generate_text_table(stats: Dict):
    """Generate formatted text table."""
    judge_models = ['Gemma-3-4B', 'Mistral-7B', 'GLM-4-9B', 'Llama-3.1-8B']
    benchmarks = ['Arena Hard', 'Alpaca Eval', 'Code Judge Bench']
    
    print("\n" + "="*140)
    print("Top-3 Attack Strategy Usage Statistics by Benchmark and Judge Model")
    print("="*140)
    
    for judge_model in judge_models:
        if judge_model not in stats:
            continue
        
        print(f"\nJudge Model: {judge_model}")
        print("-" * 140)
        
        # Header
        print(f"{'Rank':<6}", end="")
        for benchmark in benchmarks:
            print(f"{benchmark:^45}", end="")
        print()
        
        print(f"{'':<6}", end="")
        for benchmark in benchmarks:
            print(f"{'Strategy':<25} {'Count':<10} {'ASR':<10}", end="")
        print()
        
        print("-" * 140)
        
        # Data rows
        for rank in range(1, 4):
            print(f"{rank:<6}", end="")
            for benchmark in benchmarks:
                if benchmark in stats[judge_model] and len(stats[judge_model][benchmark]) >= rank:
                    attack_name, count, asr = stats[judge_model][benchmark][rank - 1]
                    print(f"{attack_name:<25} {count:<10} {asr:>9.1f}%", end="")
                else:
                    print(f"{'N/A':<25} {'N/A':<10} {'N/A':<10}", end="")
            print()
        print()

def main():
    """Main function."""
    import sys
    
    stats = load_stats()
    
    format_type = 'text'
    if len(sys.argv) > 1:
        format_type = sys.argv[1].lower()
    
    if format_type == 'latex':
        generate_latex_table(stats)
    elif format_type == 'markdown':
        generate_markdown_table(stats)
    else:
        generate_text_table(stats)
        
        # Print summary
        print("\n" + "="*140)
        print("Summary:")
        print("="*140)
        print(f"Found data for {len(stats)} judge models:")
        for judge_model in sorted(stats.keys()):
            benchmarks = list(stats[judge_model].keys())
            print(f"  - {judge_model}: {len(benchmarks)} benchmarks")
            for benchmark in benchmarks:
                top3 = stats[judge_model][benchmark]
                print(f"    * {benchmark}: Top strategy = {top3[0][0]} (Count: {top3[0][1]}, ASR: {top3[0][2]:.1f}%)")

if __name__ == "__main__":
    main()

