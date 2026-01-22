import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from viz_utils import PyRenderVisualizer
from sample_utils import *
from generate_utils import PressureGenerator

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

pose_sampler = PoseSampler(
    device=device,
    dataset='pp'
)

generator = PressureGenerator(device=device)

pv = PyRenderVisualizer()

# save = False

for i in range(6):
    # betas = sample_beta(batch_size=1, device=device)

    betas_file = f"static_stats/betas_{i}.npz"
    betas = np.load(betas_file)['betas']
    betas = torch.from_numpy(betas).unsqueeze(0).to(device)

    transl = sample_transl4pp(batch_size=1, device=device)

    # pose = pose_sampler.sample(batch_size=1)

    pose_file = f"static_stats/pose_{i}.npz"
    pose = np.load(pose_file)['pose']
    pose = torch.from_numpy(pose).unsqueeze(0).to(device)

    pv.visualize_mesh(
        global_orient=pose[:, :3], 
        body_pose=pose[:, 3:], 
        betas=betas, 
        transl=transl, 
    )

    pmap = generator.generate(
        betas=betas, 
        transl=transl, 
        poses=pose, 
    )

    plt.imshow(pmap.squeeze().detach().cpu().numpy(), cmap='viridis')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # import pdb; pdb.set_trace()

    print(f'{i} start')

    filename = os.path.join("static_stats", f"betas_{i}.npz")
    np.savez(filename, betas=betas.squeeze().detach().cpu().numpy())

    filename = os.path.join("static_stats", f"pose_{i}_tmp.npz")
    np.savez(filename, pose=pose.squeeze().detach().cpu().numpy())

    print(f"{i} done")


