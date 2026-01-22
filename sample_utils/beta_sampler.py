import numpy as np
import torch

def sample_beta(batch_size=1, sampling_method='normal', range_limit=3.0, device='cpu'):
    """
    对SMPL模型的beta参数进行采样，返回1x10的PyTorch张量。

    参数:
        batch_size (int): 采样样本数量，默认为1。
        sampling_method (str): 采样方法，'uniform'（均匀采样）或'normal'（正态分布采样）。默认为'normal'。
        range_limit (float): beta参数的范围限制，默认为3.0（即[-3, 3]）。
        device (torch.device): 计算设备 (e.g., 'cuda' or 'cpu')

    返回:
        torch.Tensor: 形状为(batch_size, 10)的beta参数张量。

    异常:
        ValueError: 如果sampling_method不是'uniform'或'normal'。
    """
    beta_dim = 10  # SMPL beta参数维度
    
    if sampling_method == 'uniform':
        # 均匀分布采样
        beta = np.random.uniform(low=-range_limit, high=range_limit, size=(batch_size, beta_dim))
    elif sampling_method == 'normal':
        # 正态分布采样
        beta = np.random.normal(loc=0, scale=2, size=(batch_size, beta_dim))
        beta = np.clip(beta, -range_limit, range_limit)  # 限制在[-range_limit, range_limit]
    else:
        raise ValueError("sampling_method must be 'uniform' or 'normal'")
    
    # 转换为PyTorch张量
    beta_tensor = torch.tensor(beta, dtype=torch.float32).to(device)
    
    return beta_tensor

