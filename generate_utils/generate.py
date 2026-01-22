import os
import yaml
import smplx
import torch

from .lib.model.cvae import SMPL2PressureCVAE

class PressureGenerator:
    # 静态配置数据
    DATASET_META = {
        'tip': {
            'max_p': 512.0,
            'crop_size': [56, 40], 
            'path': "/workspace/zyk/public_data/wzy_opt_dataset_w_feats"
        }, 
        'pressurepose': {
            'max_p': 100.0, 
            'crop_size': [64, 27], 
            'path': "/workspace/zyk/public_data/pressurepose/synth"
        }, 
        'moyo': {
            'max_p': 64.0, 
            'crop_size': [110, 37], 
            'path': "/workspace/zyk/public_data/moyo"
        }
    }

    def __init__(self, 
                 ckpt_dir="generate_utils/lib/ckpt/pressurepose_20251222_180032",
                 smpl_model_dir="E:/pyku/smpl_models",
                 device="cpu"):
        """
        初始化生成器：加载配置、权重和 SMPL 模型。
        """
        self.device = torch.device(device)
        self.ckpt_dir = ckpt_dir
        self.smpl_model_dir = smpl_model_dir

        # 1. 加载配置
        self.cfg = self._load_config()
        
        # 2. 设置数据集相关参数
        dataset_name = self.cfg['dataset']['name']
        if dataset_name not in self.DATASET_META:
             raise ValueError(f"Unknown dataset name: {dataset_name}")
             
        self.max_pressure = self.DATASET_META[dataset_name]['max_p']
        self.is_normalized = self.cfg['dataset'].get('normal', False)

        # 3. 加载 CVAE 模型
        self.cvae_model = self._load_cvae()

        # 4. 加载 SMPL 模型
        self.smpl_model = self._load_smpl()
        
        print(f"Pressure Generator loaded successfully from {self.ckpt_dir}")

    def _load_config(self):
        config_path = os.path.join(self.ckpt_dir, 'config.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_cvae(self):
        model = SMPL2PressureCVAE(self.cfg).to(self.device)
        ckpt_path = os.path.join(self.ckpt_dir, 'ckpts', 'best_model.pth')
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def _load_smpl(self):
        # 创建 SMPL 模型 (这是一个比较耗时的操作)
        smpl = smplx.create(
            self.smpl_model_dir, 
            model_type='smpl', 
            gender='neutral', 
            ext='pkl'
        ).to(self.device)
        return smpl

    @torch.no_grad()
    def generate(self, betas, transl, poses):
        """
        执行推理。
        输入参数应该是 Tensor, 维度需符合模型要求 (Batch Size, ...)。
        """
        # 确保输入在正确的设备上
        if betas.device != self.device: betas = betas.to(self.device)
        if transl.device != self.device: transl = transl.to(self.device)
        if poses.device != self.device: poses = poses.to(self.device)

        # 1. 获取 SMPL 顶点 (Vertices)
        output = self.smpl_model(
            betas=betas,
            global_orient=poses[:, :3], # 前3位是全局旋转
            body_pose=poses[:, 3:],     # 后69位是身体姿态
            transl=transl,
        )
        vertices = output.vertices

        # 2. 预测压力图
        pred_pmap = self.cvae_model.inference(vertices)

        # 3. 后处理 (反归一化 & 阈值过滤)
        if self.is_normalized:
            pred_pmap = pred_pmap * self.max_pressure
        
        # 这里的 0.1 是硬编码的阈值，也可以提取为参数
        pred_pmap[pred_pmap < 0.1] = 0

        return pred_pmap

