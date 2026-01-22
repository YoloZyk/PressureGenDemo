import torch
import os
from pathlib import Path
import numpy as np
import json
import sys

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(PROJECT_ROOT))

from .lib.model.unet1d import UNet1D
from .lib.model.flow_matching import FlowMatching

class PoseSampler:
    def __init__(self, 
                 device='cpu', 
                 dataset='pp'):
        """
        初始化 PoseSampler 类。
        在此处加载配置、统计量和模型，确保这些重型操作只执行一次。
        """
        self.device = torch.device(device)
        self.checkpoint_path = "sample_utils/lib/ckpt/fm_20251117_172225/checkpoints/best.pt"
        self.dataset = dataset
        self.stats_dir = "sample_utils/lib/data_stats"
        
        # 1. 自动检测 run_dir 并加载配置
        self.run_dir = self._detect_run_dir(self.checkpoint_path)
        self.config = self._load_config(self.run_dir)
        
        # 2. 加载数据统计量 (Mean/Std)
        self.pose_mean, self.pose_std = self._load_pose_stats()
        
        # 3. 加载模型
        self.model = self._load_model()
        print(f"Pose Sampler loaded successfully from {self.checkpoint_path}")

    def _detect_run_dir(self, checkpoint_path):
        """内部辅助方法：从 checkpoint 路径推断 run 目录"""
        abs_path = os.path.abspath(checkpoint_path)
        if 'checkpoints' in abs_path:
            checkpoint_dir = os.path.dirname(abs_path)
            if os.path.basename(checkpoint_dir) == 'checkpoints':
                return os.path.dirname(checkpoint_dir)
        return os.path.dirname(abs_path)

    def _load_config(self, run_dir):
        """内部辅助方法：加载 config.json"""
        config_path = os.path.join(run_dir, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    def _load_pose_stats(self):
        """内部辅助方法：加载 Pose 统计量"""
        if self.dataset == "pp":
            filename = "pose_stats.pt"
        else:
            filename = "t_pose_stats.pt"
            
        file_path = os.path.join(self.stats_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Stats file not found at {file_path}")

        stats = torch.load(file_path, map_location=self.device)
        return stats['mean'], stats['std']

    def _load_model(self):
        """内部辅助方法：初始化并加载模型权重"""
        unet = UNet1D(
            pose_dim=72,
            base_channels=self.config['base_channels'],
            channel_multipliers=self.config['channel_multipliers'],
            time_emb_dim=self.config['time_emb_dim'],
            mid_structure=self.config['mid_structure'],
            mid_num_heads=self.config['mid_num_heads']
        ).to(self.device)

        model = FlowMatching(
            model=unet,
            sigma=0.0
        ).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    def _unnormalize_pose(self, pose_norm):
        """内部辅助方法：反归一化"""
        return pose_norm * self.pose_std + self.pose_mean

    @torch.no_grad()
    def sample(self, batch_size=1, num_steps=100, method="euler", verbose=False):
        """
        采样方法。
        每次调用只需执行推理，无需重新加载模型。
        """
        samples_norm = self.model.sample(
            sample_shape=(batch_size, 72),
            device=self.device,
            num_steps=num_steps,
            method=method,
            verbose=verbose
        )
        
        samples_raw = self._unnormalize_pose(samples_norm)
        return samples_raw

# 使用示例
if __name__ == '__main__':
    # 1. 实例化 Sampler (只加载一次模型，耗时操作在这里)
    print("Initializing sampler...")
    sampler = PoseSampler(
        device='cpu',
        dataset='pp'
    )
    
    # 2. 多次采样 (非常快)
    print("Sampling batch 1...")
    pose_batch_1 = sampler.sample(batch_size=1)
    
    print("Sampling batch 2 (with different batch size)...")
    pose_batch_2 = sampler.sample(batch_size=4) # 甚至可以改变 batch size
    
    print(f"Batch 1 shape: {pose_batch_1.shape}")
    print(f"Batch 2 shape: {pose_batch_2.shape}")
    
    import pdb; pdb.set_trace()
    