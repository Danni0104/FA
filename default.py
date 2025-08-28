# config/default.py

import torch
from easydict import EasyDict

config = EasyDict({
    # —— 环境初始化参数 —— #
    "env": {
        "num_agvs": 5,
        "num_jobs": 20,
        "charger_capacity": 2,
        "max_steps": 400,
        "reward_scale": 0.1,   # 奖励缩放因子（训练时统一缩放）
    },

    # —— PPO 算法超参数 —— #
    "ppo": {
        "learning_rate": 1e-4,
        "gamma": 0.95,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_steps": 256,
        "batch_size": 512,
        "n_epochs": 4,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "target_kl": 0.03,
        "n_episodes": 1000,        # 训练总回合数
    },

    # —— Transformer 网络结构参数 —— #
    "transformer": {
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
    },

    # —— 运行设备 —— #
    "device": "cuda" if torch.cuda.is_available() else "cpu",
})

