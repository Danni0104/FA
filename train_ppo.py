"""

使用 Stable-Baselines3 + PPO 训练 AGVEnv。

    • DummyVecEnv + VecNormalize：支持并行、观测 / 奖励归一化
    • 学习率预热 + 线性衰减（自定义调度）
    • CheckpointCallback：定期保存模型
    • TensorBoard：可选开关（设为 None 即可关闭）

"""
# model/ppo_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class ActorNet(nn.Module):
    def __init__(self, feat_dim, hidden_dim, action_dim):
        super().__init__()
        # 简单的 shared MLP，用于每个 AGV 的行特征
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        # x: [B, feat_dim] -> logits [B, action_dim]
        return self.net(x)


class CriticNet(nn.Module):
    def __init__(self, t, f, hidden_dim):
        super().__init__()
        in_dim = t * f
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B, T, F] -> flatten -> [B, 1]
        B = x.shape[0]
        flat = x.view(B, -1)
        return self.net(flat).squeeze(-1)  # [B]


class PPOAgent:
    def __init__(self,
                 feat_dim,          # F
                 T,                 # number of token rows (T)
                 action_dim,        # choices per AGV (n_options)
                 num_agvs,          # number of AGVs
                 hidden_dim=128,
                 lr_actor=3e-4,
                 lr_critic=1e-3,
                 device=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.feat_dim = feat_dim
        self.T = T
        self.action_dim = action_dim
        self.num_agvs = num_agvs

        # actor applied per-AGV row (input feat_dim -> output action_dim)
        self.actor = ActorNet(feat_dim, hidden_dim, action_dim).to(self.device)
        # critic takes whole flattened tokens (T*F) -> scalar
        self.critic = CriticNet(T, feat_dim, hidden_dim).to(self.device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    # --- helper to convert numpy obs to torch ---
    def _to_tensor_obs(self, tokens):
        # tokens can be numpy array (T, F) or torch tensor
        if isinstance(tokens, np.ndarray):
            t = torch.from_numpy(tokens).float().to(self.device)
        elif torch.is_tensor(tokens):
            t = tokens.float().to(self.device)
        else:
            raise TypeError("tokens must be np.ndarray or torch.Tensor")
        return t

    # choose_action used during rollout (single step)
        def choose_action(self, tokens, mask=None, action_mask=None):
            t = self._to_tensor_obs(tokens)  # [T, F]
            agv_feats = t[:self.num_agvs, :]  # [num_agvs, F]

            # —— 第2步会改这里（加 job 摘要），先保留 ——
            logits = self.actor(agv_feats)  # [num_agvs, action_dim]
            logits = self._apply_action_mask(logits, action_mask)  # ★ 打掩码

            dists = Categorical(logits=logits)
            acts = dists.sample()
            logp_per_agent = dists.log_prob(acts)
            ent_per_agent = dists.entropy()

            actions_list = acts.detach().cpu().numpy().astype(int).tolist()
            logp = logp_per_agent.sum()
            entropy = ent_per_agent.sum()
            return actions_list, logp, entropy

        # convert to python list for env.step
        actions_list = acts.cpu().numpy().astype(int).tolist()

        # produce aggregated logp/entropy as sum across agents (single scalar)
        logp = logp_per_agent.sum()      # scalar tensor on device
        entropy = ent_per_agent.sum()    # scalar tensor on device

        return actions_list, logp, entropy

    # get single value for one tokens (single step)
    def get_value(self, tokens, mask=None):
        """
        tokens: numpy array (T, F) or torch tensor (T, F)
        returns: scalar float (or torch tensor on device)
        """
        t = self._to_tensor_obs(tokens).unsqueeze(0)  # [1, T, F]
        with torch.no_grad():
            v = self.critic(t)  # [1]
        # return python float or 0-d tensor - training code expects to store tensor, so return tensor
        return v.squeeze(0)

    # evaluate used during PPO update: states: [N, T, F], actions: [N, num_agvs] (torch)
    def evaluate(self, states, actions, mask=None):
        """
        states: torch tensor on device [N, T, F]
        actions: torch tensor on device, shape [N, num_agvs] (dtype long)
        returns:
            logp_new: tensor [N]
            entropy: tensor [N]
            values:  tensor [N]
        """
        if not torch.is_tensor(states):
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
        else:
            states = states.to(self.device).float()

        # critic values: [N]
        values = self.critic(states)  # [N]

        N = states.shape[0]

        # Prepare actor input: per-agent features across batch
        # states[:, :num_agvs, :] -> [N, num_agvs, F] -> reshape to [N*num_agvs, F]
        agv_feats = states[:, :self.num_agvs, :].contiguous()
        B_agv = agv_feats.shape[0] * agv_feats.shape[1]
        agv_feats_flat = agv_feats.view(-1, self.feat_dim)  # [N * num_agvs, F]

        logits = self.actor(agv_feats_flat)  # [N*num_agvs, action_dim]
        dists = Categorical(logits=logits)

        # actions may be [N, num_agvs] or flattened [N*num_agvs]
        if actions.dim() == 2:
            acts_flat = actions.view(-1)  # [N * num_agvs]
        else:
            acts_flat = actions

        # compute log_prob per agent (flatten)
        logp_per_agent = dists.log_prob(acts_flat)  # [N * num_agvs]
        ent_per_agent = dists.entropy()              # [N * num_agvs]

        # reshape back to [N, num_agvs]
        logp_per_agent = logp_per_agent.view(N, self.num_agvs)   # [N, num_agvs]
        ent_per_agent = ent_per_agent.view(N, self.num_agvs)     # [N, num_agvs]

        # aggregate per-sample (sum over agents) -> [N]
        logp_new = logp_per_agent.sum(dim=1)
        entropy = ent_per_agent.sum(dim=1)

        return logp_new, entropy, values.to(self.device)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
