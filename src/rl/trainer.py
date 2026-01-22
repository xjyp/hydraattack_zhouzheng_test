"""
强化学习训练器
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm


try:
    from data_types import PairwiseExample, RLEpisode, RLConfig
except ImportError:
    # 如果data_types模块不存在，定义简单的类
    class PairwiseExample:
        def __init__(self, instruction, response_a, response_b, preference):
            self.instruction = instruction
            self.response_a = response_a
            self.response_b = response_b
            self.preference = preference
    
    class RLEpisode:
        def __init__(self, state, action, reward, next_state, done, info=None):
            self.state = state
            self.action = action
            self.reward = reward
            self.next_state = next_state
            self.done = done
            self.info = info or {}
    
    class RLConfig:
        def __init__(self, **kwargs):
            # Set default values for common RL parameters
            self.algorithm = kwargs.get('algorithm', 'dqn')
            self.total_episodes = kwargs.get('total_timesteps', 10000)  # 保持向后兼容，但实际表示episode数量
            self.learning_rate = kwargs.get('learning_rate', 1e-3)
            self.batch_size = kwargs.get('batch_size', 32)
            self.gamma = kwargs.get('gamma', 0.99)
            self.epsilon_start = kwargs.get('epsilon_start', 1.0)
            self.epsilon_end = kwargs.get('epsilon_end', 0.01)
            self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
            self.target_update_freq = kwargs.get('target_update_freq', 100)
            self.save_freq = kwargs.get('save_freq', 100)
            self.eval_freq = kwargs.get('eval_freq', 50)
            self.eval_episodes = kwargs.get('eval_episodes', 10)
            
            # PPO specific parameters
            self.gae_lambda = kwargs.get('gae_lambda', 0.95)
            self.clip_range = kwargs.get('clip_range', 0.2)
            self.value_loss_coef = kwargs.get('value_loss_coef', 0.5)
            self.entropy_coef = kwargs.get('entropy_coef', 0.01)
            self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
            self.n_epochs = kwargs.get('n_epochs', 10)
            
            # Evaluation parameters for evaluate_on_unseen_data
            self.eval_sample_size = kwargs.get('eval_sample_size', 50)  # 周期性评估使用的样本数量
            self.max_queries = kwargs.get('max_queries', 10)
            self.success_reward = kwargs.get('success_reward', 20.0)
            self.query_penalty = kwargs.get('query_penalty', 0.5)
            self.diversity_bonus = kwargs.get('diversity_bonus', 1.0)
            self.efficiency_bonus = kwargs.get('efficiency_bonus', 2.0)
            self.confidence_threshold = kwargs.get('confidence_threshold', 0.7)
            
            # Set any additional parameters
            for key, value in kwargs.items():
                if not hasattr(self, key):
                    setattr(self, key, value)

from .enhanced_environment import EnhancedHydraAttackEnv
from .agent import DQNAgent, PPOAgent, GRPOAgent, RainbowDQNAgent

try:
    from ..attacks.base import BaseAttack
except ImportError:
    class BaseAttack:
        def __init__(self, **kwargs):
            pass

try:
    from ..evaluation.judge import BaseJudge
except ImportError:
    class BaseJudge:
        def __init__(self, **kwargs):
            pass


def evaluate_on_unseen_data(agent, test_examples: List[PairwiseExample], 
                           attacks: List[Any], judge, max_queries: int = 10,
                           success_reward: float = 20.0, query_penalty: float = 0.5,
                           diversity_bonus: float = 1.0, efficiency_bonus: float = 2.0,
                           confidence_threshold: float = 0.7,
                           logger = None, save_detailed_samples: bool = False,
                           save_dir: str = None) -> Dict[str, float]:
    """在未见数据上评估模型（优化版本：简化逻辑，避免重复获取偏好）"""
    
    # 安全检查：test_examples不能为None
    if test_examples is None:
        if logger:
            logger.logger.warning("⚠️ test_examples为None，返回空结果")
        return {
            'success_rate': 0.0,
            'avg_queries': 0.0,
            'avg_queries_successful': 0.0,
            'avg_reward': 0.0,
            'total_samples': 0,
            'successful_attacks': 0,
            'attack_usage_stats': {}
        }
    
    if logger:
        logger.logger.info("🔍 开始在未见测试数据上评估模型...")
    
    success_count = 0
    total_queries = 0  # All attacks (success + failure)
    total_queries_successful = 0  # Only successful attacks
    total_reward = 0.0
    
    # 攻击使用统计（测试过程中）
    attack_usage_stats_test = {
        attack_idx: {
            "count": 0,
            "success_count": 0,
            "total_reward": 0.0,
            "avg_reward": 0.0
        }
        for attack_idx in range(len(attacks)) if attacks and len(attacks) > 0
    }
    
    # 用于保存详细样本记录
    detailed_samples = [] if save_detailed_samples else None
    successful_samples_dir = None
    failed_samples_dir = None
    
    if save_detailed_samples and save_dir:
        import os
        successful_samples_dir = os.path.join(save_dir, "evaluation", "test_samples", "successful")
        failed_samples_dir = os.path.join(save_dir, "evaluation", "test_samples", "failed")
        os.makedirs(successful_samples_dir, exist_ok=True)
        os.makedirs(failed_samples_dir, exist_ok=True)
    
    for i, example in enumerate(test_examples):
        if logger and i % 10 == 0:
            logger.logger.info(f"  评估进度: {i+1}/{len(test_examples)}")
        
        # 1. 先获取原始偏好（用于评估）
        try:
            original_response = judge.judge_pairwise(example)
            original_preference_val = original_response.preference
            original_confidence_val = original_response.confidence
        except Exception as e:
            if logger:
                logger.logger.warning(f"获取原始偏好失败: {e}")
            raise Exception(f"获取原始偏好失败: {e}")
        
        # 2. 创建环境
        env = EnhancedHydraAttackEnv(
            examples=[example],
            attacks=attacks,
            judge=judge,
            max_queries=max_queries,
            success_reward=success_reward,
            query_penalty=query_penalty,
            diversity_bonus=diversity_bonus,
            efficiency_bonus=efficiency_bonus,
            confidence_threshold=confidence_threshold
        )
        
        # 3. 重置环境（评估模式：使用特定样本和预先获取的偏好）
        state, _ = env.reset(options={
            'use_specific_sample': True,
            'sample_idx': 0,
            'original_preference': original_preference_val,
            'original_confidence': original_confidence_val
        })
        episode_reward = 0
        queries_used = 0
        attack_sequence = []  # 记录攻击序列（trajectory）
        
        # 3. 执行攻击序列
        for step in range(max_queries):
            # 获取action mask（禁止重复使用失败的action）
            action_mask = env.get_action_mask()
            
            # 使用训练好的智能体选择动作
            if hasattr(agent, 'select_action') and 'training' in agent.select_action.__code__.co_varnames:
                action = agent.select_action(state, training=False, action_mask=action_mask)
            else:
                action, _, _ = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            # 记录攻击序列（trajectory）- 始终记录
            attack_history = info.get('attack_history', [])
            if attack_history and len(attack_history) > 0:
                last_attack = attack_history[-1]
                attack_idx = last_attack.get('attack_idx', -1)
                action_id = last_attack.get('action_id', -1)
                
                # 安全检查attack_idx
                if attack_idx >= 0 and attack_idx < len(attacks) if attacks else False:
                    attack_name = attacks[attack_idx].__class__.__name__
                else:
                    attack_name = "Unknown"
                
                # 更新攻击使用统计
                if attack_idx >= 0 and attack_idx in attack_usage_stats_test:
                    attack_usage_stats_test[attack_idx]["count"] += 1
                    attack_usage_stats_test[attack_idx]["total_reward"] += reward
                
                # 始终记录trajectory（用于分析）
                attack_sequence.append({
                    'step': step + 1,
                    'action': int(action),
                    'attack_idx': attack_idx,
                    'attack_method': attack_name,
                    'action_id': action_id,
                    'reward': float(reward),
                    'confidence': last_attack.get('confidence', 0.0),
                    'preference': last_attack.get('preference', -1),
                    'success': info.get('success', False)
                })
            
            episode_reward += reward
            queries_used += 1
            
            if done or truncated:
                break
            
            state = next_state
        
        # 4. 检查攻击结果
        # 从环境信息中获取原始偏好和最终偏好
        env_original_preference = env.original_preference
        final_preference = info.get('current_preference', env_original_preference)
        final_confidence = info.get('confidence', original_confidence_val)
        # 使用环境返回的success信息，它已经考虑了PositionAttack的特殊逻辑
        success = info.get('success', False)
        
        if success:
            success_count += 1
            total_queries_successful += queries_used  # Only count queries for successful attacks
            # 更新成功攻击的统计（记录最后使用的攻击方法）
            # 注意：attack_history在循环中可能被更新，需要从info中重新获取
            final_attack_history = info.get('attack_history', [])
            if final_attack_history and len(final_attack_history) > 0:
                last_attack = final_attack_history[-1]
                attack_idx = last_attack.get('attack_idx', -1)
                if attack_idx >= 0 and attack_idx in attack_usage_stats_test:
                    attack_usage_stats_test[attack_idx]["success_count"] += 1
        
        total_queries += queries_used  # Count queries for all attacks
        total_reward += episode_reward
        
        # 保存详细样本记录（如果启用）
        if save_detailed_samples:
            sample_id = f"sample_{i+1:03d}"
            sample_data = {
                'sample_id': sample_id,
                'question_id': example.question_id,
                'instruction': example.instruction,
                'response_a': example.response_a,
                'response_b': example.response_b,
                'model_a': example.model_a,
                'model_b': example.model_b,
                'original_preference': 'A' if original_preference_val == 0 else 'B',
                'original_confidence': float(original_confidence_val),
                'final_preference': 'A' if final_preference == 0 else 'B',
                'final_confidence': float(final_confidence),
                'success': success,
                'queries_used': queries_used,
                'episode_reward': float(episode_reward),
                'attack_sequence': attack_sequence,  # 完整的trajectory
                'metadata': example.metadata if hasattr(example, 'metadata') else {}
            }
            detailed_samples.append(sample_data)
            
            # 保存到对应的目录
            if success and successful_samples_dir:
                sample_file = os.path.join(successful_samples_dir, f"{sample_id}.json")
            elif not success and failed_samples_dir:
                sample_file = os.path.join(failed_samples_dir, f"{sample_id}.json")
            else:
                sample_file = None
            
            if sample_file:
                import json
                with open(sample_file, 'w', encoding='utf-8') as f:
                    json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    # 计算攻击使用统计的平均值
    for attack_idx in attack_usage_stats_test:
        stats = attack_usage_stats_test[attack_idx]
        if stats["count"] > 0:
            stats["avg_reward"] = stats["total_reward"] / stats["count"]
            stats["success_rate"] = stats["success_count"] / stats["count"]
        else:
            stats["avg_reward"] = 0.0
            stats["success_rate"] = 0.0
    
    # 添加攻击方法名称
    attack_usage_with_names = {}
    for attack_idx, stats in attack_usage_stats_test.items():
        if attacks and attack_idx < len(attacks):
            attack_name = attacks[attack_idx].__class__.__name__
        else:
            attack_name = "Unknown"
        attack_usage_with_names[attack_name] = {
            "attack_idx": attack_idx,
            **stats
        }
    
    # 计算评估指标
    if test_examples and len(test_examples) > 0:
        success_rate = success_count / len(test_examples)
        avg_queries = total_queries / len(test_examples)  # AQA: Average Queries per Attack (all attempts)
        avg_reward = total_reward / len(test_examples)
    else:
        success_rate = 0.0
        avg_queries = 0.0
        avg_reward = 0.0
    avg_queries_successful = total_queries_successful / success_count if success_count > 0 else 0.0  # AQSA: Average Queries per Successful Attack
    
    results = {
        'success_rate': success_rate,
        'avg_queries': avg_queries,  # AQA: Average Queries per Attack (all attempts)
        'avg_queries_successful': avg_queries_successful,  # AQSA: Average Queries per Successful Attack
        'avg_reward': avg_reward,
        'total_samples': len(test_examples),
        'successful_attacks': success_count,
        'attack_usage_stats': attack_usage_with_names
    }
    
    # 保存攻击使用统计（测试）
    if save_dir:
        import os
        attack_stats_path = os.path.join(save_dir, "attack_usage_stats_test.json")
        with open(attack_stats_path, 'w', encoding='utf-8') as f:
            json.dump(attack_usage_with_names, f, indent=2, ensure_ascii=False)
        if logger:
            logger.logger.info(f"💾 测试攻击使用统计已保存到: {attack_stats_path}")
    
    # 如果保存了详细样本，创建索引文件
    if save_detailed_samples and detailed_samples and save_dir:
        import os
        import json
        index_data = {
            'total_samples': len(test_examples),
            'successful_samples': success_count,
            'failed_samples': len(test_examples) - success_count,
            'successful_ids': [s['sample_id'] for s in detailed_samples if s['success']],
            'failed_ids': [s['sample_id'] for s in detailed_samples if not s['success']],
            'sample_metadata': {
                s['sample_id']: {
                    'question_id': s['question_id'],
                    'success': s['success'],
                    'queries_used': s['queries_used'],
                    'file_path': f"{'successful' if s['success'] else 'failed'}/{s['sample_id']}.json"
                }
                for s in detailed_samples
            }
        }
        index_path = os.path.join(save_dir, "evaluation", "test_samples", "index.json")
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        if logger:
            logger.logger.info(f"💾 测试样本索引已保存到: {index_path}")
            logger.logger.info(f"  - 成功样本: {success_count} 个")
            logger.logger.info(f"  - 失败样本: {len(test_examples) - success_count} 个")
    
    if logger:
        logger.logger.info(f"📊 未见数据评估结果:")
        logger.logger.info(f"  成功率: {success_rate:.3f}")
        logger.logger.info(f"  平均查询次数(所有攻击): {avg_queries:.3f} (AQA)")
        logger.logger.info(f"  平均查询次数(成功攻击): {avg_queries_successful:.3f} (AQSA)")
        logger.logger.info(f"  平均奖励: {avg_reward:.3f}")
        logger.logger.info(f"  总样本数: {len(test_examples)}")
        logger.logger.info(f"  成功攻击数: {success_count}")
    
    return results


class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self, 
                 env: EnhancedHydraAttackEnv,
                 agent: Any,  # DQNAgent or PPOAgent
                 config: RLConfig,
                 save_dir: str = "./models",
                 test_env: EnhancedHydraAttackEnv = None,
                 test_examples: List[PairwiseExample] = None,
                 validation_examples: List[PairwiseExample] = None,
                 attacks: List[Any] = None,
                 judge: BaseJudge = None,
                 logger = None):
        
        self.env = env
        self.agent = agent
        self.config = config
        self.save_dir = save_dir
        self.test_env = test_env
        self.test_examples = test_examples
        self.validation_examples = validation_examples
        self.attacks = attacks
        self.judge = judge
        self.logger = logger
        
        # Early stopping相关
        self.best_validation_score = -float('inf')
        self.patience_counter = 0
        self.patience = getattr(config, 'early_stopping_patience', 10)  # 默认patience为10
        self.min_delta = getattr(config, 'early_stopping_min_delta', 0.001)  # 默认最小改善
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        
        # 训练统计
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "success_rates": [],
            "query_counts": [],
            "validation_scores": [],  # 添加验证分数记录
            "training_losses": [],  # 训练损失
            "q_values": [],  # Q值统计
            "epsilon_values": []  # 探索率变化
        }
        
        # 攻击使用统计（训练过程中）
        self.attack_usage_stats_training = {
            attack_idx: {
                "count": 0,
                "success_count": 0,
                "total_reward": 0.0,
                "avg_reward": 0.0
            }
            for attack_idx in range(len(attacks)) if attacks is not None
        }
        
        # Episode trajectories（用于分析）
        self.episode_trajectories = []  # 存储每个episode的trajectory
    
    def train(self, total_episodes: int) -> Dict[str, Any]:
        """训练模型"""
        print(f"Starting RL training for {total_episodes} episodes...")
        
        episode_count = 0
        total_steps = 0
        recent_episodes = []
        
        pbar = tqdm(total=total_episodes, desc="Training")
        
        while episode_count < total_episodes:
            # 重置环境
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_success = False
            episode_queries = 0
            
            # 存储episode数据（用于PPO）
            episode_data = []
            
            # 记录当前episode的trajectory
            episode_trajectory = []
            
            # 记录当前episode的Q值和Loss
            episode_losses = []
            episode_q_values = []
            episode_epsilon = None
            
            done = False
            while not done:
                # 获取action mask（禁止重复使用失败的action）
                action_mask = self.env.get_action_mask()
                
                # 选择动作
                if isinstance(self.agent, (DQNAgent, RainbowDQNAgent)):
                    action = self.agent.select_action(state, training=True, action_mask=action_mask)
                    # 记录当前epsilon（探索率）
                    if hasattr(self.agent, 'epsilon'):
                        episode_epsilon = self.agent.epsilon
                else:  # PPOAgent or GRPOAgent
                    action, log_prob, value = self.agent.select_action(state)
                
                # 执行动作
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # 记录trajectory信息
                attack_history = info.get('attack_history', [])
                if attack_history and len(attack_history) > 0:
                    last_attack = attack_history[-1]
                    attack_idx = last_attack.get('attack_idx', -1)
                    action_id = last_attack.get('action_id', -1)
                    
                    # 安全检查attack_idx
                    if attack_idx >= 0 and attack_idx < len(self.attacks) if self.attacks else False:
                        attack_name = self.attacks[attack_idx].__class__.__name__
                    else:
                        attack_name = "Unknown"
                    
                    trajectory_step = {
                        'step': episode_length + 1,
                        'action': int(action),
                        'attack_idx': attack_idx,
                        'attack_method': attack_name,
                        'action_id': action_id,
                        'reward': float(reward),
                        'confidence': last_attack.get('confidence', 0.0),
                        'preference': last_attack.get('preference', -1),
                        'success': info.get('success', False)
                    }
                    episode_trajectory.append(trajectory_step)
                    
                    # 更新攻击使用统计
                    if attack_idx >= 0 and attack_idx in self.attack_usage_stats_training:
                        self.attack_usage_stats_training[attack_idx]["count"] += 1
                        self.attack_usage_stats_training[attack_idx]["total_reward"] += reward
                        if info.get('success', False):
                            self.attack_usage_stats_training[attack_idx]["success_count"] += 1
                
                # 存储经验
                if isinstance(self.agent, (DQNAgent, RainbowDQNAgent)):
                    self.agent.store_experience(state, action, reward, next_state, done)
                else:  # PPOAgent or GRPOAgent
                    # 安全转换state和next_state为list
                    state_list = state.tolist() if hasattr(state, 'tolist') else list(state) if isinstance(state, (list, tuple, np.ndarray)) else state
                    next_state_list = next_state.tolist() if hasattr(next_state, 'tolist') else list(next_state) if isinstance(next_state, (list, tuple, np.ndarray)) else next_state
                    episode_data.append(RLEpisode(
                        state=state_list,
                        action=action,
                        reward=reward,
                        next_state=next_state_list,
                        done=done,
                        info=info
                    ))
                
                # 更新状态
                state = next_state
                episode_reward += reward
                episode_length += 1
                episode_queries = info.get('query_count', 0)
                episode_success = info.get('success', False)
                
                total_steps += 1
                
                # 训练（DQN or RainbowDQN）
                if isinstance(self.agent, (DQNAgent, RainbowDQNAgent)) and len(self.agent.memory) >= self.agent.batch_size:
                    train_stats = self.agent.train()
                    # 收集训练指标
                    if train_stats:
                        if 'loss' in train_stats:
                            episode_losses.append(train_stats['loss'])
                        if 'avg_q_value' in train_stats:
                            episode_q_values.append(train_stats['avg_q_value'])
                        if 'max_q_value' in train_stats:
                            if 'max_q_values' not in self.training_stats:
                                self.training_stats["max_q_values"] = []
                            self.training_stats["max_q_values"].append(train_stats['max_q_value'])
                        if 'min_q_value' in train_stats:
                            if 'min_q_values' not in self.training_stats:
                                self.training_stats["min_q_values"] = []
                            self.training_stats["min_q_values"].append(train_stats['min_q_value'])
            
            # 训练（PPO or GRPO）
            if (isinstance(self.agent, PPOAgent) or isinstance(self.agent, GRPOAgent)) and episode_data:
                train_stats = self.agent.train(episode_data)
                # 收集训练指标
                if train_stats:
                    if 'policy_loss' in train_stats:
                        if 'policy_losses' not in self.training_stats:
                            self.training_stats["policy_losses"] = []
                        self.training_stats["policy_losses"].append(train_stats['policy_loss'])
                    if 'value_loss' in train_stats:
                        if 'value_losses' not in self.training_stats:
                            self.training_stats["value_losses"] = []
                        self.training_stats["value_losses"].append(train_stats['value_loss'])
            
            # 更新统计
            episode_count += 1
            pbar.update(1)  # 每个episode完成后更新进度条
            self.training_stats["episode_rewards"].append(episode_reward)
            self.training_stats["episode_lengths"].append(episode_length)
            self.training_stats["success_rates"].append(1.0 if episode_success else 0.0)
            self.training_stats["query_counts"].append(episode_queries)
            
            # 记录训练指标（每个episode的平均值）
            if episode_losses:
                avg_loss = sum(episode_losses) / len(episode_losses)
                self.training_stats["training_losses"].append(avg_loss)
            else:
                self.training_stats["training_losses"].append(0.0)
            
            if episode_q_values:
                avg_q = sum(episode_q_values) / len(episode_q_values)
                self.training_stats["q_values"].append(avg_q)
            else:
                self.training_stats["q_values"].append(0.0)
            
            # 记录epsilon（每个episode都记录，保持长度一致）
            if episode_epsilon is not None:
                self.training_stats["epsilon_values"].append(episode_epsilon)
            elif isinstance(self.agent, (DQNAgent, RainbowDQNAgent)) and hasattr(self.agent, 'epsilon'):
                # 如果episode中没有训练步骤，仍然记录当前的epsilon值
                self.training_stats["epsilon_values"].append(self.agent.epsilon)
            else:
                # 对于PPO等算法，如果没有epsilon，记录0以保持长度一致
                self.training_stats["epsilon_values"].append(0.0)
            
            # 保存trajectory（每N个episode保存一次，避免内存过大）
            if episode_count % 100 == 0 or episode_success:  # 每100个episode或成功时保存
                self.episode_trajectories.append({
                    'episode': episode_count,
                    'success': episode_success,
                    'reward': episode_reward,
                    'queries': episode_queries,
                    'trajectory': episode_trajectory
                })
            
            
            # 定期评估
            if episode_count % self.config.eval_freq == 0:
                # 优先使用验证集进行周期性验证
                eval_samples = None
                eval_type = ""
                
                if (self.validation_examples is not None and len(self.validation_examples) > 0 and 
                    self.attacks is not None and self.judge is not None):
                    # 使用验证集（使用全部验证样本，不限制数量）
                    eval_samples = self.validation_examples
                    eval_type = "validation set"
                elif (self.test_examples is not None and self.attacks is not None and 
                      self.judge is not None):
                    # 回退到测试集（使用前50个样本）
                    eval_samples = self.test_examples[:50]  # 固定使用前50个测试样本
                    eval_type = "test set"
                
                if eval_samples is not None:
                    # Use test max_queries for validation/test set evaluation, fallback to training max_queries
                    eval_max_queries = getattr(self.config, 'max_queries_test', None)
                    if eval_max_queries is None:
                        eval_max_queries = getattr(self.config, 'max_queries', 10)
                    
                    eval_stats = evaluate_on_unseen_data(
                        agent=self.agent,
                        test_examples=eval_samples,
                        attacks=self.attacks,
                        judge=self.judge,
                        max_queries=eval_max_queries,
                        success_reward=getattr(self.config, 'success_reward', 20.0),
                        query_penalty=getattr(self.config, 'query_penalty', 0.5),
                        diversity_bonus=getattr(self.config, 'diversity_bonus', 1.0),
                        efficiency_bonus=getattr(self.config, 'efficiency_bonus', 2.0),
                        confidence_threshold=getattr(self.config, 'confidence_threshold', 0.7),
                        logger=self.logger
                    )
                    
                    print(f"\nEvaluation at episode {episode_count} (using {eval_type}):")
                    print(f"  Success rate: {eval_stats['success_rate']:.3f}")
                    print(f"  Avg reward: {eval_stats['avg_reward']:.3f}")
                    print(f"  Avg queries: {eval_stats['avg_queries']:.3f}")
                    print(f"  Evaluation samples: {len(eval_samples)}")
                    
                    # 记录验证分数用于early stopping
                    validation_score = eval_stats['success_rate']  # 使用成功率作为验证分数
                    self.training_stats["validation_scores"].append(validation_score)
                    
                    # Early stopping检查
                    if self.validation_examples is not None and len(self.validation_examples) > 0:
                        if validation_score > self.best_validation_score + self.min_delta:
                            self.best_validation_score = validation_score
                            self.patience_counter = 0
                            # 保存最佳模型
                            self.save_model("best_model")
                            print(f"  🎯 New best validation score: {validation_score:.3f}")
                        else:
                            self.patience_counter += 1
                            print(f"  ⏳ Patience counter: {self.patience_counter}/{self.patience}")
                            
                            if self.patience_counter >= self.patience:
                                print(f"  🛑 Early stopping triggered at episode {episode_count}")
                                print(f"  📊 Best validation score: {self.best_validation_score:.3f}")
                                break
                else:
                    # 回退到原来的评估方法
                    eval_stats = self.evaluate(n_episodes=self.config.eval_episodes)
                    print(f"\nEvaluation at episode {episode_count} (using original evaluate method):")
                    print(f"  Success rate: {eval_stats['success_rate']:.3f}")
                    print(f"  Avg reward: {eval_stats['avg_reward']:.3f}")
                    print(f"  Avg queries: {eval_stats['avg_queries']:.3f}")
                
            
            # 定期保存模型 - 已禁用，只在训练结束时保存
            # if episode_count % self.config.save_freq == 0:
            #     self.save_model(f"model_episode_{episode_count}")
        
        pbar.close()
        
        # 最终评估
        final_eval = self.evaluate(n_episodes=self.config.eval_episodes)
        print(f"\nFinal evaluation:")
        print(f"  Success rate: {final_eval['success_rate']:.3f}")
        print(f"  Avg reward: {final_eval['avg_reward']:.3f}")
        print(f"  Avg queries: {final_eval['avg_queries']:.3f}")
        
        # 保存最终模型
        self.save_model("final_model")
        
        # 保存训练统计
        self.save_training_stats()
        
        # 保存攻击使用统计
        self.save_attack_usage_stats()
        
        # 保存trajectory数据（采样保存，避免文件过大）
        self.save_trajectories()
        
        
        return {
            "final_eval": final_eval,
            "training_stats": self.training_stats,
            "attack_usage_stats": self.attack_usage_stats_training
        }
    
    def evaluate(self, n_episodes: int = 100) -> Dict[str, float]:
        """评估模型"""
        success_count = 0
        total_reward = 0
        total_queries = 0  # All attacks (success + failure)
        total_queries_successful = 0  # Only successful attacks
        
        # 使用测试环境进行评估（如果提供），否则使用训练环境
        eval_env = self.test_env if self.test_env is not None else self.env
        
        for _ in range(n_episodes):
            state, _ = eval_env.reset()
            episode_reward = 0
            episode_queries = 0
            
            done = False
            while not done:
                if isinstance(self.agent, (DQNAgent, RainbowDQNAgent)):
                    action = self.agent.select_action(state, training=False)
                else:  # PPOAgent or GRPOAgent
                    action, _, _ = self.agent.select_action(state)
                
                next_state, reward, done, truncated, info = eval_env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_queries = info.get('query_count', 0)
                
                if info.get('success', False):
                    success_count += 1
                    break
            
            # Count queries for all attacks
            total_reward += episode_reward
            total_queries += episode_queries
            
            # Only count queries for successful attacks
            if info.get('success', False):
                total_queries_successful += episode_queries
        
        # Calculate AQA and AQSA
        if n_episodes > 0:
            avg_queries = total_queries / n_episodes  # AQA: Average Queries per Attack (all attempts)
            success_rate = success_count / n_episodes
            avg_reward = total_reward / n_episodes
        else:
            avg_queries = 0.0
            success_rate = 0.0
            avg_reward = 0.0
        avg_queries_successful = total_queries_successful / success_count if success_count > 0 else 0.0  # AQSA: Average Queries per Successful Attack
        
        return {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_queries": avg_queries,  # AQA: Average Queries per Attack (all attempts)
            "avg_queries_successful": avg_queries_successful  # AQSA: Average Queries per Successful Attack
        }
    
    def save_model(self, name: str):
        """保存模型"""
        model_path = os.path.join(self.save_dir, f"{name}.pth")
        self.agent.save(model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, name: str):
        """加载模型"""
        model_path = os.path.join(self.save_dir, f"{name}.pth")
        self.agent.load(model_path)
        print(f"Model loaded from {model_path}")
    
    def save_training_stats(self):
        """保存训练统计"""
        stats_path = os.path.join(self.save_dir, "training_stats.json")
        # 转换numpy类型
        def convert_numpy_types(obj):
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
            return obj
        
        stats_to_save = convert_numpy_types(self.training_stats)
        with open(stats_path, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        print(f"Training stats saved to {stats_path}")
    
    def save_attack_usage_stats(self):
        """保存攻击使用统计"""
        if self.attacks is None or not self.attack_usage_stats_training:
            return
        
        # 计算平均奖励
        for attack_idx in self.attack_usage_stats_training:
            stats = self.attack_usage_stats_training[attack_idx]
            if stats["count"] > 0:
                stats["avg_reward"] = stats["total_reward"] / stats["count"]
                stats["success_rate"] = stats["success_count"] / stats["count"]
            else:
                stats["avg_reward"] = 0.0
                stats["success_rate"] = 0.0
        
        # 添加攻击方法名称
        attack_usage_with_names = {}
        for attack_idx, stats in self.attack_usage_stats_training.items():
            if self.attacks and attack_idx < len(self.attacks):
                attack_name = self.attacks[attack_idx].__class__.__name__
            else:
                attack_name = "Unknown"
            attack_usage_with_names[attack_name] = {
                "attack_idx": attack_idx,
                **stats
            }
        
        stats_path = os.path.join(self.save_dir, "attack_usage_stats_training.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(attack_usage_with_names, f, indent=2, ensure_ascii=False)
        print(f"Attack usage stats (training) saved to {stats_path}")
    
    def save_trajectories(self):
        """保存trajectory数据（采样保存）"""
        if not self.episode_trajectories:
            return
        
        # 只保存部分trajectory（每10个episode保存一个，以及所有成功的）
        sampled_trajectories = []
        for traj in self.episode_trajectories:
            if traj['episode'] % 10 == 0 or traj['success']:
                sampled_trajectories.append(traj)
        
        trajectories_path = os.path.join(self.save_dir, "episode_trajectories.json")
        with open(trajectories_path, 'w', encoding='utf-8') as f:
            json.dump(sampled_trajectories, f, indent=2, ensure_ascii=False)
        print(f"Episode trajectories saved to {trajectories_path} (sampled: {len(sampled_trajectories)}/{len(self.episode_trajectories)})")


def create_trainer(examples: List[PairwiseExample],
                  attacks: List[BaseAttack],
                  judge: BaseJudge,
                  config: RLConfig,
                  algorithm: str = "PPO",
                  test_examples: List[PairwiseExample] = None,
                  validation_examples: List[PairwiseExample] = None,
                  logger = None,
                  **kwargs) -> RLTrainer:
    """创建训练器"""
    
    # 创建环境
    env = EnhancedHydraAttackEnv(
        examples=examples,
        attacks=attacks,
        judge=judge,
        max_queries=config.max_queries,
        **kwargs
    )
    
    # 创建Agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if algorithm == "DQN":
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            **kwargs
        )
    elif algorithm == "PPO":
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            **kwargs
        )
    elif algorithm == "GRPO":
        agent = GRPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            group_size=kwargs.get('group_size', 4),
            relative_weight=kwargs.get('relative_weight', 0.3),
            memory_size=kwargs.get('memory_size', 10000),
            target_update_freq=kwargs.get('target_update_freq', 100),
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    return RLTrainer(
        env=env, 
        agent=agent, 
        config=config, 
        test_examples=test_examples,
        validation_examples=validation_examples,
        attacks=attacks,
        judge=judge,
        logger=logger,
        **kwargs
    )
