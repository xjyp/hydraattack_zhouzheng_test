"""
随机种子设置工具
确保所有随机性来源都被正确设置，保证实验的可重现性
"""

import random
import numpy as np
import torch
import os
from typing import Optional


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    设置所有随机种子以确保可重现性
    
    Args:
        seed: 随机种子值
        deterministic: 是否启用确定性模式（可能影响性能）
    """
    # Python内置random模块
    random.seed(seed)
    
    # NumPy随机性
    np.random.seed(seed)
    
    # PyTorch CPU随机性
    torch.manual_seed(seed)
    
    # PyTorch CUDA随机性
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 设置CUDA确定性模式
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 设置PyTorch的确定性模式（如果可用）
        if hasattr(torch, 'set_deterministic'):
            torch.set_deterministic(True)
    
    # 设置环境变量以确保某些库的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✅ 随机种子已设置为: {seed} (确定性模式: {deterministic})")


def get_random_state() -> dict:
    """
    获取当前所有随机状态
    
    Returns:
        包含所有随机状态的字典
    """
    state = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda_random'] = torch.cuda.get_rng_state()
        if torch.cuda.device_count() > 1:
            state['torch_cuda_all_random'] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict) -> None:
    """
    恢复随机状态
    
    Args:
        state: 之前保存的随机状态字典
    """
    random.setstate(state['python_random'])
    np.random.set_state(state['numpy_random'])
    torch.set_rng_state(state['torch_random'])
    
    if torch.cuda.is_available() and 'torch_cuda_random' in state:
        torch.cuda.set_rng_state(state['torch_cuda_random'])
        if 'torch_cuda_all_random' in state:
            torch.cuda.set_rng_state_all(state['torch_cuda_all_random'])


def ensure_reproducibility(seed: int = 42, deterministic: bool = True) -> None:
    """
    确保实验可重现性的便捷函数
    
    Args:
        seed: 随机种子值，默认为42
        deterministic: 是否启用确定性模式
    """
    set_seed(seed, deterministic)
    
    # 额外的可重现性设置
    if torch.cuda.is_available():
        # 确保CUDA操作的顺序一致性
        torch.cuda.synchronize()
    
    print(f"🔒 实验可重现性已确保 (seed={seed})")


# 为了向后兼容，提供简化的函数名
def set_all_seeds(seed: int) -> None:
    """设置所有随机种子的简化接口"""
    set_seed(seed, deterministic=True)
