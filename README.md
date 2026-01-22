# hydraattack_zhouzheng_test
师兄，这个项目主要跑：run1_baseline_gcg.sh与run1_baseline_tap.sh

更换模型主要修改这个路径
JUDGE_MODEL_PATH="/root/autodl-tmp/Qwen2.5-7B-Instruct"


run1_baseline_gcg.sh中切换以下模型
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct
Qwen2.5-1.5B-Instruct
Qwen2.5-0.5B-Instruct

conda create -n Zhouzheng python=3.12

source /etc/network_turbo

pip install -r requirements.txt
