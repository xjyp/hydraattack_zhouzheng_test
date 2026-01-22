#!/usr/bin/env python3
"""
Verify that the average values (last column) are correctly calculated.
"""

# Data from the script
data = {
    'Arena Hard': {
        'GCG': [5.3, 26.3, 18.9, 83.8, 26.3, 20.0, 30.1],
        'AutoDAN': [72.7, 63.6, 67.6, 63.4, 64.9, 90.9, 70.5],
        'JudgeDeceiver': [73.3, 46.0, 48.7, 52.0, 68.0, 31.3, 53.2],
        'PAIR': [86.0, 97.3, 18.0, 86.0, 85.3, 88.7, 76.9],
        'HydraAttack': [89.3, 96.0, 95.3, 92.7, 86.7, 96.7, 92.8]
    },
    'Alpaca Eval': {
        'GCG': [66.7, 70.0, 27.0, 85.7, 58.0, 66.7, 62.3],
        'AutoDAN': [36.4, 45.5, 81.1, 36.3, 43.2, 90.9, 55.6],
        'JudgeDeceiver': [88.2, 57.1, 30.4, 89.4, 45.3, 27.3, 56.3],
        'PAIR': [85.7, 96.2, 21.1, 81.4, 98.1, 98.1, 80.1],
        'HydraAttack': [93.2, 90.7, 97.5, 88.8, 88.8, 95.7, 92.5]
    },
    'Code Judge Bench': {
        'GCG': [5.2, 52.6, 32.4, 85.7, 18.8, 18.2, 35.5],
        'AutoDAN': [63.6, 54.5, 45.5, 63.6, 52.9, 91.9, 62.0],
        'JudgeDeceiver': [75.5, 34.4, 58.2, 59.4, 36.8, 35.4, 50.0],
        'PAIR': [72.7, 97.6, 21.9, 85.5, 93.8, 94.3, 77.6],
        'HydraAttack': [97.4, 97.4, 89.1, 97.6, 93.6, 93.6, 94.8]
    }
}

methods = ['GCG', 'AutoDAN', 'JudgeDeceiver', 'PAIR', 'HydraAttack']
benchmarks = ['Arena Hard', 'Alpaca Eval', 'Code Judge Bench']

print("=" * 80)
print("验证平均值计算")
print("=" * 80)

all_correct = True

for benchmark in benchmarks:
    print(f"\n{benchmark}:")
    print("-" * 80)
    
    for method in methods:
        values = data[benchmark][method]
        # First 6 values are judge model results
        judge_values = values[:6]
        # Last value is the claimed average
        claimed_avg = values[6]
        
        # Calculate actual average
        actual_avg = sum(judge_values) / len(judge_values)
        
        # Check if they match (with rounding to 1 decimal place)
        claimed_avg_rounded = round(claimed_avg, 1)
        actual_avg_rounded = round(actual_avg, 1)
        
        status = "✓" if abs(claimed_avg_rounded - actual_avg_rounded) < 0.01 else "✗"
        
        if abs(claimed_avg_rounded - actual_avg_rounded) >= 0.01:
            all_correct = False
        
        print(f"{method:15s} | 前6个值: {judge_values}")
        print(f"{'':15s} | 计算平均值: {actual_avg:.6f} → {actual_avg_rounded:.1f}")
        print(f"{'':15s} | 代码中的值: {claimed_avg:.1f} {status}")
        if abs(claimed_avg_rounded - actual_avg_rounded) >= 0.01:
            print(f"{'':15s} | ⚠️  错误！应该是 {actual_avg_rounded:.1f}")
        print()

print("=" * 80)
if all_correct:
    print("✓ 所有平均值计算正确！")
else:
    print("✗ 发现错误！请检查上面的结果。")
print("=" * 80)

