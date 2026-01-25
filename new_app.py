# 【功能描述】
# 本项目基于 SMPL 模型，通过随机生成或从库中加载参数，分析人体姿态与压力分布的关系。
# 主要功能包括：
# 1. 随机生成人体参数（包括 betas 和 pose）
# 2. 从预定义库中加载已保存的参数
# 3. 利用 SMPL 模型生成人体网格和压力分布
# 4. 可视化展示生成的人体模型和压力分布
# 5. 允许用户保存新生成的参数到库中【新增功能】

# streamlit run new_app.py --server.port 8501

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt

from sample_utils import PoseSampler, sample_beta, sample_transl4pp
from generate_utils import PressureGenerator

# --- 1. 初始化 (必须放在最前面) ---
st.set_page_config(page_title="SMPL2Pressure", layout="wide")

# 强制注入 CSS 减少顶部空白
st.markdown("""
    <style>
    .block-container { padding-top: 3rem; padding-bottom: 0rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stMetric"] { margin-top: -1.1rem; margin-bottom: -2rem; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def init_models(_device):
    p_sampler = PoseSampler(device=_device, dataset='pp')
    p_gen = PressureGenerator(device=_device)
    return p_sampler, p_gen

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_sampler, generator = init_models(DEVICE)
# print(f"Model initialized on device: {DEVICE}")

# 初始化 Session State
if 'betas' not in st.session_state:
    st.session_state.betas = torch.zeros((1, 10)).to(DEVICE)
    st.session_state.pose = torch.zeros((1, 72)).to(DEVICE)
    st.session_state.transl = sample_transl4pp(batch_size=1, device=DEVICE)
    st.session_state.run_trigger = 0

# --- 2. 辅助函数 ---
@st.cache_data(show_spinner="Calculating Body Mesh...")
def compute_results(_betas, _pose, _transl, trigger=0):
    pmap = generator.generate(betas=_betas, transl=_transl, poses=_pose)
    pmap = pmap.flip(1)

    with torch.no_grad():
        output = generator.smpl_model(
            betas=_betas, 
            global_orient=_pose[:, :3], 
            body_pose=_pose[:, 3:], 
            transl=_transl
        )
        verts = output.vertices[0].cpu().numpy()
        faces = generator.smpl_model.faces
    return verts, faces, pmap.squeeze().cpu().numpy()

def get_static_list(prefix):
    if not os.path.exists("static_stats"): return ["dummy_1.npz"]
    files = [f for f in os.listdir("static_stats") if f.startswith(prefix) and f.endswith(".npz")]
    return sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))

# --- 3. 侧边栏与控制逻辑 ---
with st.sidebar:
    st.title("🎛️ Controls")
    mode = st.radio("Input Source", ["Random", "Library"], horizontal=True)
    st.divider()
    
    if mode == "Library":
        beta_files = get_static_list("betas_")
        pose_files = get_static_list("pose_")
        sel_beta = st.selectbox("Body Shape (Betas)", beta_files) 
        sel_pose = st.selectbox("Human Pose", pose_files)
        
        # 【关键修复】Library 模式下直接读取文件并更新 state
        b_data = np.load(f"static_stats/{sel_beta}")['betas']
        st.session_state.betas = torch.from_numpy(b_data).float().unsqueeze(0).to(DEVICE)
        p_data = np.load(f"static_stats/{sel_pose}")['pose']
        st.session_state.pose = torch.from_numpy(p_data).float().unsqueeze(0).to(DEVICE)
    else:
        st.info("Random mode active. Click 'Generate' in the main panel.")

# --- 4. 主界面布局与按钮响应 ---
head_c1, head_c2, head_c3 = st.columns([3, 1, 1])
with head_c1:
    st.subheader("🔬 Pressure Map Synthesis via SMPL Model")

with head_c2:
    # 按钮点击后，更新 Session State
    if st.button("🔄 Generate / Resample", type="primary", use_container_width=True):
        if mode == "Random":
            # 只有在 Random 模式下才重新采样 betas 和 pose
            st.session_state.betas = sample_beta(batch_size=1, device=DEVICE)
            st.session_state.pose = pose_sampler.sample(batch_size=1)
            pass 
        
        # 无论哪种模式，通常位移和触发器都要更新
        st.session_state.transl = sample_transl4pp(batch_size=1, device=DEVICE)
        st.session_state.run_trigger += 1 

with head_c3:
    if mode == "Random":
        if st.button("Save to Library", type="secondary", use_container_width=True):
            try:
                # 1. 确定保存目录
                save_dir = "static_stats"
                os.makedirs(save_dir, exist_ok=True)
                
                # 2. 获取当前最大的序号
                existing_files = [f for f in os.listdir(save_dir) if f.startswith("betas_")]
                indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files if '_' in f]
                next_idx = max(indices) + 1 if indices else 1
                
                # 3. 保存 betas 和 pose (从 tensor 转回 numpy)
                beta_path = os.path.join(save_dir, f"betas_{next_idx}.npz")
                pose_path = os.path.join(save_dir, f"pose_{next_idx}.npz")
                
                np.savez(beta_path, betas=st.session_state.betas.cpu().numpy().squeeze(0))
                np.savez(pose_path, pose=st.session_state.pose.cpu().numpy().squeeze(0))
                
                st.toast(f"Saved as index {next_idx}!", icon='✅')
                
                # 4. 强制刷新下拉列表缓存
                st.cache_data.clear() 
            except Exception as e:
                st.error(f"Save failed: {e}")
    else:
        # Library 模式下显示一个占位符或禁用按钮
        st.button("Save to Library", disabled=True, use_container_width=True)


# --- 5. 计算逻辑 ---
verts, faces, pmap_np = compute_results(
    st.session_state.betas, 
    st.session_state.pose, 
    st.session_state.transl,
    trigger=st.session_state.run_trigger
)

# --- 6. 渲染 (保持之前的 7:3 布局) ---
view_c1, view_c2 = st.columns([7.2, 2.8])
DISPLAY_HEIGHT = 730

with view_c1:
    st.subheader("🌐 3D Mesh")
    fig_3d = go.Figure(data=[go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='LightBlue',
        opacity=1.0,
        flatshading=False,
        lighting=dict(ambient=0.5, diffuse=0.9, specular=0.5, roughness=0.6),
        lightposition=dict(x=100, y=200, z=150)
    )])
    
    fig_3d.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            camera=dict(
                eye=dict(x=0, y=0, z=2.0),     # Positioned high on Z-axis
                up=dict(x=0, y=1, z=0),        # Y-axis points "up" on the screen
                center=dict(x=0., y=0., z=0.)     # Looking at the origin
            )
        ),
        height=DISPLAY_HEIGHT, 
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with view_c2:
    m_c1, m_c2 = st.columns(2)
    m_c1.metric("Peak Pressure", f"{pmap_np.max():.2f}")
    m_c2.metric("Contact Pixels", f"{(pmap_np > 0.5).sum()}")

    # with st.container():
    # aspect='equal' ensures no distortion
    fig_2d, ax = plt.subplots(figsize=(5, 8)) 
    im = ax.imshow(pmap_np, cmap='viridis', origin='lower', aspect='equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    # Use container width helps alignment
    st.pyplot(fig_2d, use_container_width=True)




