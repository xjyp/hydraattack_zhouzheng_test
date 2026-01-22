"""
强化学习Agent
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from typing import List, Tuple, Dict, Any
import random
from collections import deque
import pickle
import os

try:
    from data_types import RLEpisode
except ImportError:
    # 如果data_types模块不存在，定义一个简单的RLEpisode类
    class RLEpisode:
        def __init__(self, state, action, reward, next_state, done):
            self.state = state
            self.action = action
            self.reward = reward
            self.next_state = next_state
            self.done = done


class DQNAgent:
    """DQN Agent实现"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # 神经网络
        self.q_network = self._build_network(state_dim, action_dim, hidden_dim)
        self.target_network = self._build_network(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.memory = deque(maxlen=memory_size)
        
        # 更新目标网络
        self.update_target_network()
        self.step_count = 0
    
    def _build_network(self, state_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
        """构建神经网络"""
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def select_action(self, state: np.ndarray, training: bool = True, action_mask: np.ndarray = None) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            training: 是否在训练模式
            action_mask: 动作掩码，1表示可用，0表示禁用。如果为None，则所有动作都可用。
        """
        # If epsilon >= 1.0, always use random selection (for ablation studies like Random Selection)
        if self.epsilon >= 1.0:
            if action_mask is not None:
                available_actions = np.where(action_mask > 0)[0]
                if len(available_actions) > 0:
                    return np.random.choice(available_actions)
                else:
                    return 0
            return random.randint(0, self.action_dim - 1)
        
        if training and random.random() < self.epsilon: # Epsilon (ε) 是探索率参数，控制智能体在训练过程中随机探索的概率。
            # 探索阶段：如果有掩码，只从可用action中随机选择
            if action_mask is not None:
                available_actions = np.where(action_mask > 0)[0]
                if len(available_actions) > 0:
                    return np.random.choice(available_actions)
                else:
                    # 如果没有可用action，返回第一个（理论上不应该发生）
                    return 0
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0) 
            q_values = self.q_network(state_tensor)
            
            # 应用掩码：将禁用action的Q值设为-inf
            if action_mask is not None:
                q_values = q_values.clone()
                q_values[0, action_mask == 0] = float('-inf')
            
            # 如果所有action都被禁用，返回第一个可用action（理论上不应该发生）
            if action_mask is not None and q_values.max() == float('-inf'):
                available_actions = np.where(action_mask > 0)[0]
                if len(available_actions) > 0:
                    return available_actions[0]
                return 0
            
            return q_values.argmax().item() # 选择Q值最大的动作
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self) -> Dict[str, float]:
        """训练模型"""
        if len(self.memory) < self.batch_size:
            return {"loss": 0.0}
        
        # 采样批次数据
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        # 确保张量尺寸匹配，避免广播警告
        current_q_vals = current_q_values.squeeze()
        if current_q_vals.dim() == 0:  # 如果是标量，添加维度
            current_q_vals = current_q_vals.unsqueeze(0)
        loss = nn.MSELoss()(current_q_vals, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 计算Q值统计
        with torch.no_grad():
            avg_q_value = current_q_vals.mean().item()
            max_q_value = current_q_vals.max().item()
            min_q_value = current_q_vals.min().item()
            avg_target_q = target_q_values.mean().item()
            td_errors = target_q_values - current_q_vals
        
        # 更新epsilon，越来越小，直到达到epsilon_min，不再探索，只进行利用。
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 更新目标网络
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        return {
            "loss": loss.item(),
            "avg_q_value": avg_q_value,
            "max_q_value": max_q_value,
            "min_q_value": min_q_value,
            "avg_target_q": avg_target_q,
            "epsilon": self.epsilon,
            "td_error_mean": td_errors.abs().mean().item()
        }
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
    
    def save_model(self, filepath: str):
        """保存模型（别名方法，用于兼容性）"""
        self.save(filepath)
    
    def load_model(self, filepath: str):
        """加载模型（别名方法，用于兼容性）"""
        self.load(filepath)


class RainbowDQNAgent:
    """Rainbow DQN Agent实现
    集成了Double DQN、Dueling Network Architecture和优先经验回放
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100,
                 prioritized_replay: bool = True,
                 prioritized_replay_alpha: float = 0.6,
                 prioritized_replay_beta: float = 0.4,
                 prioritized_replay_beta_increment: float = 0.001,
                 max_grad_norm: float = 1.0):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.prioritized_replay = prioritized_replay
        self.max_grad_norm = max_grad_norm
        
        # 神经网络 - 使用Dueling架构
        self.q_network = self._build_dueling_network(state_dim, action_dim, hidden_dim)
        self.target_network = self._build_dueling_network(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 优先经验回放
        if prioritized_replay:
            self.memory = []  # 存储经验：[(state, action, reward, next_state, done, td_error), ...]
            self.max_memory_size = memory_size
            self.prioritized_replay_alpha = prioritized_replay_alpha
            self.prioritized_replay_beta = prioritized_replay_beta
            self.prioritized_replay_beta_increment = prioritized_replay_beta_increment
            self.max_priority = 1.0  # 初始优先级
        else:
            self.memory = deque(maxlen=memory_size)
        
        # 更新目标网络
        self.update_target_network()
        self.step_count = 0
    
    def _build_dueling_network(self, state_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
        """构建Dueling网络架构（分离价值函数和优势函数）"""
        class DuelingNetwork(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim):
                super().__init__()
                self.feature = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
                # 价值函数 V(s)
                self.value = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1)
                )
                # 优势函数 A(s, a)
                self.advantage = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, action_dim)
                )
            
            def forward(self, x):
                features = self.feature(x)
                value = self.value(features)
                advantage = self.advantage(features)
                # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
                q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
                return q_values
        
        return DuelingNetwork(state_dim, action_dim, hidden_dim)
    
    def select_action(self, state: np.ndarray, training: bool = True, action_mask: np.ndarray = None) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            training: 是否在训练模式
            action_mask: 动作掩码，1表示可用，0表示禁用。如果为None，则所有动作都可用。
        """
        # If epsilon >= 1.0, always use random selection (for ablation studies like Random Selection)
        if self.epsilon >= 1.0:
            if action_mask is not None:
                available_actions = np.where(action_mask > 0)[0]
                if len(available_actions) > 0:
                    return np.random.choice(available_actions)
                else:
                    return 0
            return random.randint(0, self.action_dim - 1)
        
        if training and random.random() < self.epsilon:
            # 探索阶段：如果有掩码，只从可用action中随机选择
            if action_mask is not None:
                available_actions = np.where(action_mask > 0)[0]
                if len(available_actions) > 0:
                    return np.random.choice(available_actions)
                else:
                    # 如果没有可用action，返回第一个（理论上不应该发生）
                    return 0
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            
            # 应用掩码：将禁用action的Q值设为-inf
            if action_mask is not None:
                q_values = q_values.clone()
                q_values[0, action_mask == 0] = float('-inf')
            
            # 如果所有action都被禁用，返回第一个可用action（理论上不应该发生）
            if action_mask is not None and q_values.max() == float('-inf'):
                available_actions = np.where(action_mask > 0)[0]
                if len(available_actions) > 0:
                    return available_actions[0]
                return 0
            
            return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """存储经验（带优先级）"""
        if self.prioritized_replay:
            # 使用最大优先级（新经验）
            priority = self.max_priority
            self.memory.append((state, action, reward, next_state, done, priority))
            if len(self.memory) > self.max_memory_size:
                self.memory.pop(0)  # 移除最老的样本
        else:
            self.memory.append((state, action, reward, next_state, done))
    
    def _sample_batch(self):
        """采样批次数据（优先经验回放或均匀采样）"""
        if self.prioritized_replay and len(self.memory) > 0:
            # 计算优先级
            priorities = np.array([exp[5] for exp in self.memory])  # TD误差（优先级）
            probabilities = priorities ** self.prioritized_replay_alpha
            probabilities /= probabilities.sum()
            
            # 采样
            indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
            batch = [self.memory[i] for i in indices]
            weights = (len(self.memory) * probabilities[indices]) ** (-self.prioritized_replay_beta)
            weights /= weights.max()  # 归一化重要性采样权重
            
            return batch, indices, torch.FloatTensor(weights)
        else:
            # 均匀采样
            batch = random.sample(self.memory, self.batch_size)
            indices = None
            weights = torch.ones(self.batch_size)
            return batch, indices, weights
    
    def train(self) -> Dict[str, float]:
        """训练模型（使用Double DQN和优先经验回放）"""
        if len(self.memory) < self.batch_size:
            return {"loss": 0.0}
        
        # 采样批次数据
        batch, indices, weights = self._sample_batch()
        states, actions, rewards, next_states, dones, *_ = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: 使用主网络选择动作，目标网络评估
        with torch.no_grad():
            # 使用主网络选择最佳动作
            next_actions = self.q_network(next_states).argmax(1)
            # 使用目标网络评估这些动作的Q值
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算TD误差和损失（带重要性采样权重）
        current_q_vals = current_q_values.squeeze()
        if current_q_vals.dim() == 0:
            current_q_vals = current_q_vals.unsqueeze(0)
        
        td_errors = target_q_values - current_q_vals
        loss = (weights * td_errors.pow(2)).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（防止梯度爆炸，提高训练稳定性）
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        
        # 更新优先经验回放的优先级
        if self.prioritized_replay and indices is not None:
            with torch.no_grad():
                td_errors_abs = td_errors.abs().detach().cpu().numpy()
                for i, idx in enumerate(indices):
                    self.memory[idx] = (*self.memory[idx][:5], float(td_errors_abs[i] + 1e-6))
            
            # 更新最大优先级
            self.max_priority = max([exp[5] for exp in self.memory])
            
            # 更新beta（逐渐增加到1.0）
            self.prioritized_replay_beta = min(1.0, 
                self.prioritized_replay_beta + self.prioritized_replay_beta_increment)
        
        # 计算Q值统计
        with torch.no_grad():
            avg_q_value = current_q_vals.mean().item()
            max_q_value = current_q_vals.max().item()
            min_q_value = current_q_vals.min().item()
            avg_target_q = target_q_values.mean().item()
        
        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 更新目标网络
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        return {
            "loss": loss.item(),
            "avg_q_value": avg_q_value,
            "max_q_value": max_q_value,
            "min_q_value": min_q_value,
            "avg_target_q": avg_target_q,
            "epsilon": self.epsilon,
            "td_error_mean": td_errors.abs().mean().item()
        }
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'prioritized_replay_beta': self.prioritized_replay_beta if self.prioritized_replay else None
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        if self.prioritized_replay and 'prioritized_replay_beta' in checkpoint:
            self.prioritized_replay_beta = checkpoint['prioritized_replay_beta']
    
    def save_model(self, filepath: str):
        """保存模型（别名方法，用于兼容性）"""
        self.save(filepath)
    
    def load_model(self, filepath: str):
        """加载模型（别名方法，用于兼容性）"""
        self.load(filepath)


class PPOAgent:
    """PPO Agent实现"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 n_epochs: int = 10,
                 batch_size: int = 64):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # 策略网络和价值网络
        self.policy_network = self._build_policy_network(state_dim, action_dim, hidden_dim)
        self.value_network = self._build_value_network(state_dim, hidden_dim)
        
        self.optimizer = optim.Adam(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            lr=learning_rate
        )
    
    def _build_policy_network(self, state_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
        """构建策略网络"""
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def _build_value_network(self, state_dim: int, hidden_dim: int) -> nn.Module:
        """构建价值网络"""
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.policy_network(state_tensor)
            value = self.value_network(state_tensor)
            
            # 采样动作
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   next_value: float, dones: List[bool]) -> Tuple[List[float], List[float]]:
        """计算GAE优势估计"""
        advantages = []
        returns = []
        
        # 确保rewards是列表类型
        if not isinstance(rewards, list):
            print(f"Warning: rewards is not a list, type: {type(rewards)}, value: {rewards}")
            rewards = [rewards] if isinstance(rewards, (int, float)) else list(rewards)
        
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[i]
                next_val = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_val * next_non_terminal - values[i]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return advantages, returns
    
    def train(self, episodes: List[RLEpisode]) -> Dict[str, float]:
        """训练模型"""
        if not episodes:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # 提取数据
        states = torch.FloatTensor([ep.state for ep in episodes])
        actions = torch.LongTensor([ep.action for ep in episodes])
        rewards = [ep.reward for ep in episodes]
        next_states = torch.FloatTensor([ep.next_state for ep in episodes])
        dones = [ep.done for ep in episodes]
        
        # 计算价值和优势
        with torch.no_grad():
            values = self.value_network(states).squeeze()
            # 确保values是列表类型
            if values.dim() == 0:  # 如果是标量
                values = [values.item()]
            else:
                values = values.tolist()
            next_value = self.value_network(next_states[-1:]).item() if not dones[-1] else 0.0
        
        advantages, returns = self.compute_gae(rewards, values, next_value, dones)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # 标准化优势
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            # 如果只有一个样本，不进行标准化
            advantages = advantages - advantages.mean()
        
        # 训练多个epoch
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(self.n_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(episodes))
            
            for start_idx in range(0, len(episodes), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(episodes))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 前向传播
                action_probs = self.policy_network(batch_states)
                values = self.value_network(batch_states).squeeze()
                
                # 计算损失
                action_dist = torch.distributions.Categorical(action_probs)
                log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy().mean()
                
                # 策略损失
                ratio = torch.exp(log_probs - log_probs.detach())
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                # 确保张量尺寸匹配，避免广播警告
                if values.dim() == 0:  # 如果是标量，添加维度
                    values = values.unsqueeze(0)
                if batch_returns.dim() == 0:  # 如果是标量，添加维度
                    batch_returns = batch_returns.unsqueeze(0)
                value_loss = nn.MSELoss()(values, batch_returns)
                
                # 总损失
                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_network.parameters()) + list(self.value_network.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        n_batches = self.n_epochs * (len(episodes) + self.batch_size - 1) // self.batch_size
        
        return {
            "policy_loss": total_policy_loss / n_batches,
            "value_loss": total_value_loss / n_batches,
            "entropy": total_entropy / n_batches
        }
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def save_model(self, filepath: str):
        """保存模型（别名方法，用于兼容性）"""
        self.save(filepath)
    
    def load_model(self, filepath: str):
        """加载模型（别名方法，用于兼容性）"""
        self.load(filepath)


class GRPOAgent:
    """GRPO (Group Relative Policy Optimization) Agent实现
    结合DQN和PPO的优点，使用组相对策略优化
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 group_size: int = 4,
                 relative_weight: float = 0.3,
                 memory_size: int = 10000,
                 target_update_freq: int = 100):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.group_size = group_size
        self.relative_weight = relative_weight
        self.target_update_freq = target_update_freq
        
        # 策略网络和价值网络
        self.policy_network = self._build_policy_network(state_dim, action_dim, hidden_dim)
        self.value_network = self._build_value_network(state_dim, hidden_dim)
        self.target_value_network = self._build_value_network(state_dim, hidden_dim)
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            lr=learning_rate
        )
        
        # 经验回放（借鉴DQN）
        self.memory = deque(maxlen=memory_size)
        
        # 更新目标网络
        self.update_target_network()
        self.step_count = 0
        
        # 组策略历史（用于相对策略优化）
        self.group_policy_history = deque(maxlen=group_size * 10)
    
    def _build_policy_network(self, state_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
        """构建策略网络"""
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def _build_value_network(self, state_dim: int, hidden_dim: int) -> nn.Module:
        """构建价值网络"""
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.policy_network(state_tensor)
            value = self.value_network(state_tensor)
            
            # 采样动作
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """存储经验（借鉴DQN）"""
        self.memory.append((state, action, reward, next_state, done))
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   next_value: float, dones: List[bool]) -> Tuple[List[float], List[float]]:
        """计算GAE优势估计（借鉴PPO）"""
        advantages = []
        returns = []
        
        # 确保rewards是列表类型
        if not isinstance(rewards, list):
            rewards = [rewards] if isinstance(rewards, (int, float)) else list(rewards)
        
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[i]
                next_val = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_val * next_non_terminal - values[i]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return advantages, returns
    
    def compute_group_relative_advantages(self, advantages: torch.Tensor, 
                                        group_indices: List[int]) -> torch.Tensor:
        """计算组相对优势（GRPO核心特性）"""
        if len(group_indices) < 2:
            return advantages
        
        # 将优势按组分组
        group_advantages = []
        for i in range(0, len(advantages), self.group_size):
            group_adv = advantages[i:i+self.group_size]
            if len(group_adv) > 0:
                # 计算组内相对优势
                group_mean = group_adv.mean()
                relative_adv = group_adv - group_mean
                group_advantages.append(relative_adv)
        
        if group_advantages:
            # 合并所有组的相对优势
            relative_advantages = torch.cat(group_advantages)
            # 结合原始优势和相对优势
            combined_advantages = (1 - self.relative_weight) * advantages + \
                                self.relative_weight * relative_advantages
            return combined_advantages
        else:
            return advantages
    
    def train(self, episodes: List[RLEpisode]) -> Dict[str, float]:
        """训练模型（结合DQN和PPO的方法）"""
        if not episodes:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "group_loss": 0.0}
        
        # 提取数据
        states = torch.FloatTensor([ep.state for ep in episodes])
        actions = torch.LongTensor([ep.action for ep in episodes])
        rewards = [ep.reward for ep in episodes]
        next_states = torch.FloatTensor([ep.next_state for ep in episodes])
        dones = [ep.done for ep in episodes]
        
        # 计算价值和优势
        with torch.no_grad():
            values = self.value_network(states).squeeze()
            # 确保values是列表类型
            if values.dim() == 0:  # 如果是标量
                values = [values.item()]
            else:
                values = values.tolist()
            next_value = self.value_network(next_states[-1:]).item() if not dones[-1] else 0.0
        
        advantages, returns = self.compute_gae(rewards, values, next_value, dones)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # 计算组相对优势（GRPO核心）
        group_indices = list(range(len(episodes)))
        relative_advantages = self.compute_group_relative_advantages(advantages, group_indices)
        
        # 标准化优势
        if len(relative_advantages) > 1:
            relative_advantages = (relative_advantages - relative_advantages.mean()) / (relative_advantages.std() + 1e-8)
        else:
            relative_advantages = relative_advantages - relative_advantages.mean()
        
        # 训练多个epoch
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_group_loss = 0
        
        for epoch in range(self.n_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(episodes))
            
            for start_idx in range(0, len(episodes), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(episodes))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = relative_advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 前向传播
                action_probs = self.policy_network(batch_states)
                values = self.value_network(batch_states).squeeze()
                
                # 计算损失
                action_dist = torch.distributions.Categorical(action_probs)
                log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy().mean()
                
                # 策略损失（使用相对优势）
                old_log_probs = log_probs.detach()
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                if values.dim() == 0:
                    values = values.unsqueeze(0)
                if batch_returns.dim() == 0:
                    batch_returns = batch_returns.unsqueeze(0)
                value_loss = nn.MSELoss()(values, batch_returns)
                
                # 组一致性损失（GRPO特有）
                group_loss = 0.0
                if len(batch_states) >= self.group_size:
                    # 计算组内策略一致性
                    group_states = batch_states[:self.group_size]
                    group_probs = self.policy_network(group_states)
                    group_entropy = action_dist.entropy()[:self.group_size]
                    # 组内策略应该相对一致
                    group_loss = -group_entropy.mean() * 0.1  # 鼓励组内策略多样性
                
                # 总损失
                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy + group_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_network.parameters()) + list(self.value_network.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_group_loss += group_loss
        
        # 更新目标网络（借鉴DQN）
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        n_batches = self.n_epochs * (len(episodes) + self.batch_size - 1) // self.batch_size
        
        return {
            "policy_loss": total_policy_loss / n_batches,
            "value_loss": total_value_loss / n_batches,
            "entropy": total_entropy / n_batches,
            "group_loss": total_group_loss / n_batches
        }
    
    def update_target_network(self):
        """更新目标网络（借鉴DQN）"""
        self.target_value_network.load_state_dict(self.value_network.state_dict())
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'target_value_network_state_dict': self.target_value_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        self.target_value_network.load_state_dict(checkpoint['target_value_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
    
    def save_model(self, filepath: str):
        """保存模型（别名方法，用于兼容性）"""
        self.save(filepath)
    
    def load_model(self, filepath: str):
        """加载模型（别名方法，用于兼容性）"""
        self.load(filepath)
