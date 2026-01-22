#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlpacaEval Pairwise LLM-as-a-Judge 示例
演示如何使用AlpacaEval进行pairwise evaluation的具体输入输出格式
"""

import json
import os
from pathlib import Path

def show_alpaca_eval_example():
    """
    展示AlpacaEval进行pairwise LLM-as-a-judge评估的具体示例
    """
    print("=" * 80)
    print("AlpacaEval Pairwise LLM-as-a-Judge 评估示例")
    print("=" * 80)
    
    # 1. 输入数据格式示例
    print("\n1. 模型输出数据格式 (model_outputs.json):")
    print("-" * 50)
    
    model_output_example = {
        "dataset": "helpful_base",
        "instruction": "What are the names of some famous actors that started their careers on Broadway?",
        "output": "1. Robert De Niro\n2. Al Pacino\n3. Denzel Washington\n4. Matthew Broderick\n5. Jane Fonda\n6. Liza Minnelli\n7. Catherine Zeta-Jones\n8. Hugh Jackman\n9. Kevin Spacey\n10. Whoopi Goldberg",
        "generator": "airoboros-33b"
    }
    
    print(json.dumps(model_output_example, indent=2, ensure_ascii=False))
    
    # 2. 参考模型输出格式
    print("\n2. 参考模型输出数据格式 (reference_outputs.json):")
    print("-" * 50)
    
    reference_output_example = {
        "dataset": "helpful_base", 
        "instruction": "What are the names of some famous actors that started their careers on Broadway?",
        "output": "Some famous actors who started their careers on Broadway include:\n\n• Hugh Jackman - Known for roles in musicals like 'Oklahoma!' and 'The Boy from Oz'\n• Sarah Jessica Parker - Started in 'Annie' as a child\n• Nathan Lane - Broadway veteran in shows like 'The Producers'\n• Idina Menzel - Known for 'Rent' and 'Wicked'\n• Lin-Manuel Miranda - Creator and star of 'Hamilton'\n• Sutton Foster - Tony Award winner for multiple Broadway shows",
        "generator": "text-davinci-003"
    }
    
    print(json.dumps(reference_output_example, indent=2, ensure_ascii=False))
    
    # 3. Pairwise Evaluation Prompt
    print("\n3. GPT-4 Judge的评估Prompt:")
    print("-" * 50)
    
    evaluation_prompt = """<|im_start|>system
You are a helpful assistant, that ranks models by the quality of their answers.
<|im_end|>
<|im_start|>user
I want you to create a leaderboard of different of large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

Here is the prompt:
{
    "instruction": "What are the names of some famous actors that started their careers on Broadway?"
}

Here are the outputs of the models:
[
    {
        "model": "model_1",
        "answer": "1. Robert De Niro\n2. Al Pacino\n3. Denzel Washington\n4. Matthew Broderick\n5. Jane Fonda\n6. Liza Minnelli\n7. Catherine Zeta-Jones\n8. Hugh Jackman\n9. Kevin Spacey\n10. Whoopi Goldberg"
    },
    {
        "model": "model_2", 
        "answer": "Some famous actors who started their careers on Broadway include:\n\n• Hugh Jackman - Known for roles in musicals like 'Oklahoma!' and 'The Boy from Oz'\n• Sarah Jessica Parker - Started in 'Annie' as a child\n• Nathan Lane - Broadway veteran in shows like 'The Producers'\n• Idina Menzel - Known for 'Rent' and 'Wicked'\n• Lin-Manuel Miranda - Creator and star of 'Hamilton'\n• Sutton Foster - Tony Award winner for multiple Broadway shows"
    }
]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
    {'model': <model-name>, 'rank': <model-rank>},
    {'model': <model-name>, 'rank': <model-rank>}
]

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give.
<|im_end|>"""
    
    print(evaluation_prompt)
    
    # 4. GPT-4 Judge的输出示例
    print("\n4. GPT-4 Judge的评估输出:")
    print("-" * 50)
    
    judge_output_example = [
        {'model': 'model_2', 'rank': 1},
        {'model': 'model_1', 'rank': 2}
    ]
    
    print(json.dumps(judge_output_example, indent=2))
    
    # 5. 解析后的偏好结果
    print("\n5. 解析后的偏好结果:")
    print("-" * 50)
    
    preference_result = {
        "instruction": "What are the names of some famous actors that started their careers on Broadway?",
        "output_1": model_output_example["output"],
        "output_2": reference_output_example["output"],
        "preference": 2,  # 偏好model_2 (reference model)
        "annotator": "alpaca_eval_gpt4",
        "price_per_example": 0.0136,
        "time_per_example": 1.455
    }
    
    print(json.dumps(preference_result, indent=2, ensure_ascii=False))
    
    # 6. 最终的Win Rate计算
    print("\n6. Win Rate计算方式:")
    print("-" * 50)
    
    print("""
Win Rate = (模型被偏好的次数) / (总比较次数)

例如，如果在805个样本中：
- airoboros-33b被偏好: 580次
- text-davinci-003被偏好: 225次

则airoboros-33b的Win Rate = 580/805 = 72.0%

这意味着在72%的情况下，GPT-4 judge认为airoboros-33b的输出
比text-davinci-003的输出更好。
    """)
    
    # 7. 评估指标
    print("\n7. 评估指标说明:")
    print("-" * 50)
    
    metrics_explanation = """
• Human Agreement: 与人类标注的一致性百分比
• Price: 每1000个样本的评估成本(美元)
• Time: 每1000个样本的评估时间(秒)
• Bias: 评估器的系统性偏差
• Variance: 评估器的方差(一致性)
• Proba. prefer longer: 偏好更长输出的概率

示例 - alpaca_eval_gpt4评估器:
- Human Agreement: 69.2%
- Price: $13.6/1000 examples  
- Time: 1455 seconds/1000 examples
- Bias: 28.4
- Variance: 14.6
- Proba. prefer longer: 0.68
    """
    
    print(metrics_explanation)
    
    # 8. 使用命令示例
    print("\n8. 使用AlpacaEval的命令示例:")
    print("-" * 50)
    
    command_examples = """
# 基本评估命令
alpaca_eval --model_outputs 'model_outputs.json' \
           --annotators_config 'alpaca_eval_gpt4' \
           --name 'my_model'

# 使用不同的judge
alpaca_eval --model_outputs 'model_outputs.json' \
           --annotators_config 'claude' \
           --name 'my_model'

# 分析评估器性能
alpaca_eval analyze_evaluators --annotators_config 'alpaca_eval_gpt4'

# 生成排行榜
alpaca_eval make_leaderboard --leaderboard_path 'my_leaderboard.csv'
    """
    
    print(command_examples)
    
    print("\n" + "=" * 80)
    print("总结: AlpacaEval通过让强大的LLM(如GPT-4)对两个模型的输出进行")
    print("成对比较，计算目标模型相对于参考模型的胜率，从而评估模型性能。")
    print("=" * 80)

def load_real_example():
    """
    加载真实的AlpacaEval数据示例
    """
    alpaca_dir = Path("alpaca_eval")
    
    if not alpaca_dir.exists():
        print("AlpacaEval目录不存在，请确保已下载相关数据")
        return
    
    # 加载一个真实的模型输出示例
    results_dir = alpaca_dir / "results"
    if results_dir.exists():
        model_dirs = list(results_dir.iterdir())
        if model_dirs:
            model_output_file = model_dirs[0] / "model_outputs.json"
            if model_output_file.exists():
                with open(model_output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"\n真实示例来自: {model_dirs[0].name}")
                    print(json.dumps(data[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    show_alpaca_eval_example()
    load_real_example()