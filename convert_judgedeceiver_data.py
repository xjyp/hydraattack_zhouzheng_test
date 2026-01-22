#!/usr/bin/env python3
"""
Convert judgedeceiver CSV data to JSON format matching arena_hard format.
Merge train and test data, then re-split with 8:2 ratio to align with other datasets.
"""

import csv
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any


def extract_instruction(instruction_str: str) -> str:
    """Extract instruction content after '# Instruction: ' marker."""
    marker = "# Instruction: "
    if marker in instruction_str:
        idx = instruction_str.find(marker)
        return instruction_str[idx + len(marker):].strip()
    return instruction_str.strip()


def parse_target(target_str: str) -> str:
    """Parse target string to extract 'Output (a)' or 'Output (b)'."""
    target_str = target_str.strip()
    if target_str == "Output (a) is better.":
        return "Output (a)"
    elif target_str == "Output (b) is better.":
        return "Output (b)"
    else:
        # Handle other possible formats
        if "Output (a)" in target_str:
            return "Output (a)"
        elif "Output (b)" in target_str:
            return "Output (b)"
        return target_str


def convert_csv_to_json_format(csv_file_path: str, dataset_name: str, start_id: int = 0) -> List[Dict[str, Any]]:
    """Convert a CSV file to JSON format entries."""
    entries = []
    current_id = start_id
    
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract instruction
            instruction = extract_instruction(row['instruction'])
            
            # Create entry
            entry = {
                "question_id": f"{dataset_name}_{current_id}",
                "instruction": instruction,
                "response_a": row['text1'],
                "response_b": row['text2'],
                "model_a": "model_a",
                "model_b": "model_b",
                "metadata": {
                    "target": parse_target(row['target'])
                }
            }
            entries.append(entry)
            current_id += 1
    
    return entries


def merge_csv_files(csv_dir: str, dataset_name: str) -> List[Dict[str, Any]]:
    """Merge all CSV files in a directory."""
    all_entries = []
    csv_files = sorted(Path(csv_dir).glob("*.csv"))
    
    current_id = 0
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        entries = convert_csv_to_json_format(str(csv_file), dataset_name, start_id=current_id)
        all_entries.extend(entries)
        current_id += len(entries)
    
    return all_entries


def create_split_info(dataset_name: str, train_entries: List[Dict], test_entries: List[Dict], output_dir: str, random_seed: int = 42, train_ratio: float = 0.8, test_ratio: float = 0.2) -> None:
    """Create split_info.json file."""
    total_samples = len(train_entries) + len(test_entries)
    train_samples = len(train_entries)
    test_samples = len(test_entries)
    
    split_info = {
        "benchmark": dataset_name,
        "total_samples": total_samples,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "random_seed": random_seed,
        "data_files": {
            "train": f"data/split/{dataset_name}_train.json",
            "test": f"data/split/{dataset_name}_test.json"
        }
    }
    
    output_path = Path(output_dir) / f"{dataset_name}_split_info.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"Created {output_path}")


def main():
    base_dir = Path("/home/wzdou/project/hydraattack_share")
    raw_data_dir = base_dir / "raw_data" / "judgedeceiver"
    output_dir = base_dir / "data" / "split"
    
    # Configuration: align with arena_hard, alpaca_eval, code_judge_bench
    train_ratio = 0.8
    test_ratio = 0.2
    random_seed = 42
    
    datasets = [
        {
            "name": "llmbar",
            "train_dir": raw_data_dir / "llmbar_train",
            "test_dir": raw_data_dir / "llmbar_test"
        },
        {
            "name": "mtbench",
            "train_dir": raw_data_dir / "mtbench_train",
            "test_dir": raw_data_dir / "mtbench_test"
        }
    ]
    
    for dataset in datasets:
        dataset_name = dataset["name"]
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Step 1: Process all train CSV files
        print(f"\nStep 1: Processing train CSV files...")
        train_entries = merge_csv_files(str(dataset["train_dir"]), dataset_name)
        print(f"  Total train entries from CSV: {len(train_entries)}")
        
        # Step 2: Process all test CSV files
        print(f"\nStep 2: Processing test CSV files...")
        test_entries = []
        csv_files = sorted(dataset["test_dir"].glob("*.csv"))
        current_id = len(train_entries)  # Continue ID from train entries
        for csv_file in csv_files:
            print(f"  Processing {csv_file.name}...")
            entries = convert_csv_to_json_format(str(csv_file), dataset_name, start_id=current_id)
            test_entries.extend(entries)
            current_id += len(entries)
        print(f"  Total test entries from CSV: {len(test_entries)}")
        
        # Step 3: Merge all entries
        print(f"\nStep 3: Merging train and test data...")
        all_entries = train_entries + test_entries
        print(f"  Total merged entries: {len(all_entries)}")
        
        # Step 4: Shuffle and re-split with 8:2 ratio
        print(f"\nStep 4: Shuffling and re-splitting with {train_ratio:.0%}:{test_ratio:.0%} ratio (random_seed={random_seed})...")
        random.seed(random_seed)
        shuffled_entries = all_entries.copy()
        random.shuffle(shuffled_entries)
        
        # Re-assign IDs after shuffling
        for idx, entry in enumerate(shuffled_entries):
            entry["question_id"] = f"{dataset_name}_{idx}"
        
        # Split data
        train_end = int(len(shuffled_entries) * train_ratio)
        new_train_entries = shuffled_entries[:train_end]
        new_test_entries = shuffled_entries[train_end:]
        
        print(f"  New train entries: {len(new_train_entries)} samples ({len(new_train_entries)/len(shuffled_entries)*100:.1f}%)")
        print(f"  New test entries: {len(new_test_entries)} samples ({len(new_test_entries)/len(shuffled_entries)*100:.1f}%)")
        
        # Step 5: Write train JSON
        train_output_path = output_dir / f"{dataset_name}_train.json"
        os.makedirs(output_dir, exist_ok=True)
        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump(new_train_entries, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Created {train_output_path} ({len(new_train_entries)} entries)")
        
        # Step 6: Write test JSON
        test_output_path = output_dir / f"{dataset_name}_test.json"
        with open(test_output_path, 'w', encoding='utf-8') as f:
            json.dump(new_test_entries, f, indent=2, ensure_ascii=False)
        print(f"✅ Created {test_output_path} ({len(new_test_entries)} entries)")
        
        # Step 7: Create split_info.json
        create_split_info(dataset_name, new_train_entries, new_test_entries, str(output_dir), random_seed, train_ratio, test_ratio)
        
        print(f"\n✅ Completed processing {dataset_name}")
        print(f"  Original: {len(train_entries)} train + {len(test_entries)} test = {len(all_entries)} total")
        print(f"  Re-split: {len(new_train_entries)} train + {len(new_test_entries)} test = {len(shuffled_entries)} total")
        print(f"  Ratio: {train_ratio:.0%}:{test_ratio:.0%} (aligned with arena_hard, alpaca_eval, code_judge_bench)")
    
    print(f"\n{'='*60}")
    print("🎉 All datasets processed successfully!")
    print(f"{'='*60}")
    print(f"✅ All datasets now use {train_ratio:.0%}:{test_ratio:.0%} split with random_seed={random_seed}")
    print(f"✅ Data alignment: llmbar, mtbench, arena_hard, alpaca_eval, code_judge_bench")


if __name__ == "__main__":
    main()

