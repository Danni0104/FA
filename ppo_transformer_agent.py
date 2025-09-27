# model/ppo_transformer_agent.py
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from typing import Optional, List, Union
from model.transformer_model import TransformerEncoder

class PPOAgent:
    def __init__(self,
                 feat_dim: int,
                 T: int,
                 action_dim: int,
                 num_agvs: int,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 device: Optional[str] = None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.feat_dim = feat_dim
        self.T = T
        self.action_dim = action_dim
        self.num_agvs = num_agvs
        self.hidden_dim = hidden_dim

        self.encoder = TransformerEncoder(
            input_dim=feat_dim,
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        ).to(self.device)

        self.actor_head = nn.Sequential(
            nn.Linear(self.num_agvs * hidden_dim, hidden_dim * self.num_agvs),
            nn.ReLU(),
            nn.Linear(hidden_dim * self.num_agvs, self.num_agvs * action_dim)
        ).to(self.device)

        self.critic_head = nn.Sequential(
            nn.Linear(self.T * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        self.optimizer_actor = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor_head.parameters()),
            lr=lr_actor
        )
        self.optimizer_critic = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.critic_head.parameters()),
            lr=lr_critic
        )

    def _to_tensor_obs(self, tokens: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(tokens, np.ndarray):
            t = torch.from_numpy(tokens).float().to(self.device)
        elif torch.is_tensor(tokens):
            t = tokens.float().to(self.device)
        else:
            raise TypeError("tokens must be np.ndarray or torch.Tensor")
        return t

    def _apply_action_mask(self, logits: torch.Tensor, action_mask: Optional[np.ndarray]) -> torch.Tensor:
        if action_mask is not None:
            mask_t = torch.from_numpy(action_mask).bool().to(logits.device)
            logits.masked_fill_(~mask_t, float('-inf'))
        return logits

    def choose_action(self, tokens, mask=None, action_mask=None):
        t = self._to_tensor_obs(tokens).unsqueeze(0)

        with torch.no_grad():
            encoded_tokens = self.encoder(t)  # shape: [1, T, hidden_dim]

            # 从编码后的 tokens 中提取AGV的特征并展平
            agv_encoded = encoded_tokens[:, :self.num_agvs, :]  # shape: [1, num_agvs, hidden_dim]
            flat_agv_encoded = agv_encoded.view(1, -1)  # shape: [1, num_agvs * hidden_dim]

            # 将展平后的特征输入actor_head，得到所有AGV的动作logits
            all_logits = self.actor_head(flat_agv_encoded)  # shape: [1, num_agvs * action_dim]

        # 将扁平化的logits重新整理成 [num_agvs, action_dim] 的形状
        logits = all_logits.view(self.num_agvs, self.action_dim)

        # 应用动作掩码
        if action_mask is not None:
            mask_t = torch.from_numpy(action_mask).bool().to(self.device)
            logits.masked_fill_(~mask_t, float('-inf'))

        # 为每个AGV创建一个独立的Categorical分布
        dists = Categorical(logits=logits)
        acts = dists.sample()

        # 计算 log_prob 和 entropy
        logp_per_agent = dists.log_prob(acts)
        ent_per_agent = dists.entropy()

        actions_list = acts.detach().cpu().numpy().astype(int).tolist()
        logp = logp_per_agent.sum()
        entropy = ent_per_agent.sum()

        return actions_list, logp, entropy

    def get_value(self, tokens: Union[np.ndarray, torch.Tensor], mask: Optional[np.ndarray] = None) -> torch.Tensor:
        t = self._to_tensor_obs(tokens).unsqueeze(0)
        with torch.no_grad():
            encoded_tokens = self.encoder(t)
            flat_encoded = encoded_tokens.view(1, -1)
            v = self.critic_head(flat_encoded)
        return v.squeeze()

    '''def evaluate(self, states: torch.Tensor, actions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    '''

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 action_mask: Optional[torch.Tensor] = None) -> (
            torch.Tensor, torch.Tensor, torch.Tensor):
        # 1. 编码所有tokens
        encoded_tokens = self.encoder(states)
        N = states.shape[0]

        # 提取 AGV 特征并展平
        agv_encoded = encoded_tokens[:, :self.num_agvs, :]
        flat_agv_encoded = agv_encoded.view(N, -1)

        # 获取值函数 (critic_head)
        flat_all_tokens = encoded_tokens.view(N, -1)
        values = self.critic_head(flat_all_tokens).squeeze()

        # 获取策略网络的 logits (actor_head)
        all_logits = self.actor_head(flat_agv_encoded)

        # 2. 将扁平化的 logits 重新整理成 [N, num_agvs, action_dim] 的形状
        logits = all_logits.view(N, self.num_agvs, self.action_dim)

        # 3. 应用动作掩码
        if action_mask is not None:
            # 确保 action_mask 的形状是 [N, num_agvs, action_dim]
            mask_t = action_mask.bool().to(logits.device)
            logits.masked_fill_(~mask_t, float('-inf'))

        # 4. 计算 log_prob 和 entropy
        dists = Categorical(logits=logits)

        # actions 的形状应为 [N, num_agvs]，与 dists 的形状匹配
        logp_per_agent = dists.log_prob(actions)
        ent_per_agent = dists.entropy()

        # 5. 聚合 log_prob 和 entropy
        logp_new = logp_per_agent.sum(dim=1)
        entropy = ent_per_agent.sum(dim=1)

        return logp_new, entropy, values

    def save(self, path: str):
        torch.save({
            'actor_head': self.actor_head.state_dict(),
            'critic_head': self.critic_head.state_dict(),
            'encoder': self.encoder.state_dict()
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor_head.load_state_dict(ckpt['actor_head'])
        self.critic_head.load_state_dict(ckpt['critic_head'])
        self.encoder.load_state_dict(ckpt['encoder'])
