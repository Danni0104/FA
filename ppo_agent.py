# model/ppo_agent.py
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ============ utils: running mean/var for state norm ============
class RunningMeanStd:
    def __init__(self, shape=(1,), eps: float = 1e-4, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.eps = eps

    def update(self, x: torch.Tensor):
        if x.numel() == 0:
            return
        x = x.to(self.device)
        batch_mean = torch.mean(x, dim=0)
        batch_var  = torch.var(x, dim=0, unbiased=False)
        batch_count = x.size(0)

        if float(self.count.item()) == 0.0:
            self.mean  = batch_mean
            self.var   = batch_var
            self.count = torch.tensor(float(batch_count), device=self.device)
            return

        delta     = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean  = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2  = m_a + m_b + (delta**2) * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count

        self.mean  = new_mean
        self.var   = new_var
        self.count = tot_count


def normalize_columns(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor,
                          cols: List[int]) -> torch.Tensor:

    """只对连续列做标准化；x: [N,T,F] 或 [1,T,F]；mean/var: [F]"""
    x = x.clone()
    m = mean.view(1, 1, -1)
    s = torch.sqrt(var + 1e-8).view(1, 1, -1)
    x[..., cols] = (x[..., cols] - m[..., cols]) / s[..., cols]
    return x

# ============ MLP Actor / Critic（原来那套，保留） ============
class SimpleActor(nn.Module):
    """[N, T, F] -> 取前 num_agvs 行 -> MLP(逐AGV) -> [N, num_agvs, action_dim]"""
    def __init__(self, feat_dim: int, num_agvs: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.num_agvs   = int(num_agvs)
        self.action_dim = int(action_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        agv = x[:, :self.num_agvs, :]            # [N, num_agvs, F]
        h   = self.mlp(agv)                      # [N, num_agvs, H]
        logits = self.head(h)                    # [N, num_agvs, A]
        return logits

class SimpleCritic(nn.Module):
    """[N, T, F] -> mean-pool(T) -> MLP -> value [N,1]"""
    def __init__(self, feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled = x.mean(dim=1)                  # [N, F]
        h = F.relu(self.fc1(pooled))
        h = F.relu(self.fc2(h))
        v = self.out(h)                         # [N, 1]
        return v


# ============ Transformer 版 Actor / Critic ============

class SinePositionalEncoding(nn.Module):
    """标准正弦位置编码（batch_first=True）。"""
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [L, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, L, D]
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerBackbone(nn.Module):
    """线性投影 -> 位置编码 -> TransformerEncoder (batch_first=True)。"""
    def __init__(self, feat_dim: int, d_model: int, n_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.posenc = SinePositionalEncoding(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [N, T, F]
        h = self.in_proj(x)                 # [N, T, D]
        h = self.posenc(h)                  # [N, T, D]
        # key_padding_mask: [N, T] -> True means to mask (ignore)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return h  # [N, T, D]


class TransformerActor(nn.Module):
    """
    [N, T, F] -> Transformer -> 取前 num_agvs 行（AGV行）-> MLP -> logits [N, num_agvs, action_dim]
    """
    def __init__(self, feat_dim: int, num_agvs: int, action_dim: int,
                 d_model: int = 128, n_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_agvs = int(num_agvs)
        self.action_dim = int(action_dim)
        self.backbone = TransformerBackbone(feat_dim, d_model, n_heads, num_layers, dropout)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # mask (if provided) 作为 key_padding_mask：True=pad/忽略
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()  # 你的 mask 是可用=1，不可用=0，需翻转

        h = self.backbone(x, key_padding_mask)     # [N, T, D]
        agv_tokens = h[:, :self.num_agvs, :]       # [N, num_agvs, D]
        logits = self.proj(agv_tokens)             # [N, num_agvs, action_dim]
        return logits


class TransformerCritic(nn.Module):
    """[N, T, F] -> Transformer -> mean-pool(T) -> MLP -> value [N,1]"""
    def __init__(self, feat_dim: int, d_model: int = 128, n_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = TransformerBackbone(feat_dim, d_model, n_heads, num_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()
        h = self.backbone(x, key_padding_mask)     # [N, T, D]
        pooled = h.mean(dim=1)                     # [N, D]
        v = self.head(pooled)                      # [N, 1]
        return v


# ============ PPOAgent（可切换后端） ============

class PPOAgent:
    """
    兼容版 PPOAgent：可在 MLP 与 Transformer 间切换。
    关键构造参数：
      - use_transformer: bool  是否启用 Transformer（默认 False=MLP）
      - hidden_dim / embed_dim: MLP 隐层或 Transformer 的 d_model
      - n_heads, num_layers: Transformer 的多头与层数
      - num_agvs, action_dim, input_dim/feat_dim: 与环境维度一致
    """
    def __init__(self,
                 input_dim: int = None,
                 action_dim: int = None,
                 num_agvs: int = 5,
                 hidden_dim: int = 128,         # MLP hidden 或 Transformer d_model
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 device: Optional[str] = None,
                 use_transformer: bool = False,
                 n_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 **kwargs):

        # 兼容别名 feat_dim/embed_dim
        if input_dim is None:
            input_dim = kwargs.get("feat_dim", None)
        # 允许从 config['transformer']['embed_dim'] 传进来
        hidden_dim = kwargs.get("hidden_dim", hidden_dim)
        hidden_dim = kwargs.get("embed_dim", hidden_dim)

        if action_dim is None:
            action_dim = kwargs.get("action_dim", None)

        if input_dim is None or action_dim is None:
            raise TypeError(f"PPOAgent requires input_dim and action_dim. "
                            f"Got input_dim={input_dim}, action_dim={action_dim}")

        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.input_dim = int(input_dim)   # F
        self.action_dim = int(action_dim)
        self.num_agvs = int(num_agvs)
        self.use_transformer = bool(use_transformer)
        self.d_model = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        # 选择后端
        if self.use_transformer:
            self.actor = TransformerActor(
                feat_dim=self.input_dim,
                num_agvs=self.num_agvs,
                action_dim=self.action_dim,
                d_model=self.d_model,
                n_heads=self.n_heads,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)

            self.critic = TransformerCritic(
                feat_dim=self.input_dim,
                d_model=self.d_model,
                n_heads=self.n_heads,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)
        else:
            self.actor = SimpleActor(
                feat_dim=self.input_dim,
                num_agvs=self.num_agvs,
                action_dim=self.action_dim,
                hidden_dim=self.d_model,
            ).to(self.device)

            self.critic = SimpleCritic(
                feat_dim=self.input_dim,
                hidden_dim=self.d_model,
            ).to(self.device)

        # 优化器
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 状态归一化（按特征维）
        self.state_rms = RunningMeanStd(shape=(self.input_dim,), device=self.device)
        self.norm_cols = [0, 2]  # 只规范化连续列：0列(battery/duration)、2列(due_norm)

    # ---------- choose action ----------
    def choose_action(self, tokens, mask=None, action_mask=None) -> Tuple[
        List[int], torch.Tensor, torch.Tensor]:
        """
        tokens: [T, F]
        mask:   [T]，1=有效, 0=padding
        action_mask:
            - int: 只允许 [0..int-1]
            - [action_dim]: 1/0（所有 AGV 相同）
            - [num_agvs, action_dim]: 1/0（逐 AGV）
        返回: actions(list[int] 长度=num_agvs), joint_logp(标量), mean_entropy(标量)
        """
        import numpy as _np

        # ---- to tensor ----
        if isinstance(tokens, _np.ndarray):
            arr = tokens.astype(_np.float32)
        else:
            arr = _np.array(tokens, dtype=_np.float32)
        T, F = arr.shape
        x = torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, T, F]
        m = None
        if mask is not None:
            m = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)  # [1, T]

        # ---- state norm：更新统计 + 仅归一化连续列 ----
        self.state_rms.update(x.view(-1, F))  # 更新统计
        mean = self.state_rms.mean.to(self.device)
        var = self.state_rms.var.to(self.device)
        xn = normalize_columns(x, mean, var, cols=self.norm_cols)

        # ---- 采样阶段禁用梯度 + 支持二维动作掩码 ----
        with torch.no_grad():
            logits = self.actor(xn, mask=m).squeeze(0)  # [num_agvs, action_dim]

            if action_mask is not None:
                if isinstance(action_mask, int):
                    idxs = torch.arange(self.action_dim, device=self.device)
                    invalid = idxs >= int(action_mask)  # [action_dim] bool
                    logits = logits.masked_fill(invalid.unsqueeze(0), float("-1e9"))
                else:
                    am = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
                    if am.dim() == 1:  # [action_dim]
                        am = am.unsqueeze(0).expand(self.num_agvs, -1)  # -> [num_agvs, action_dim]
                    elif am.dim() == 2:  # [num_agvs, action_dim]
                        if am.size(0) != self.num_agvs or am.size(1) != self.action_dim:
                            raise ValueError(
                                f"action_mask shape {am.shape} != [{self.num_agvs},{self.action_dim}]")
                    else:
                        raise ValueError(f"Unsupported action_mask dim: {am.dim()}")
                    logits = logits.masked_fill(~am, float("-1e9"))  # 不可用动作置为 -inf

            dist = Categorical(logits=logits)  # 独立对每个 AGV 采样
            acts = dist.sample()  # [num_agvs]
            logp_per = dist.log_prob(acts)  # [num_agvs]
            ent_per = dist.entropy()  # [num_agvs]

        joint_logp = logp_per.sum()  # 标量
        mean_ent = ent_per.mean()  # 标量
        a_list = [int(x) for x in acts.detach().cpu().numpy().tolist()]
        return a_list, joint_logp.to(self.device), mean_ent.to(self.device)

    # ---------- value ----------
    def get_value(self, tokens, mask=None) -> float:
        import numpy as _np
        if isinstance(tokens, _np.ndarray):
            arr = tokens.astype(_np.float32)
        else:
            arr = _np.array(tokens, dtype=_np.float32)
        x = torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, T, F]
        m = None
        if mask is not None:
            m = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        mean = self.state_rms.mean.to(self.device)
        var = self.state_rms.var.to(self.device)
        xn = normalize_columns(x, mean, var, cols=self.norm_cols)

        with torch.no_grad():
            v = self.critic(xn, mask=m)  # [1, 1]
        return float(v.item())

    # ---------- evaluate (PPO update) ----------
    def evaluate(
            self,
            states: torch.Tensor,  # [N, T, F]
            actions: torch.Tensor,  # [N] or [N, num_agvs]
            mask: Optional[torch.Tensor] = None,  # [N, T] (sequence key_padding)
            action_mask: Optional[torch.Tensor] = None,  # [N,num_agvs,A] / [num_agvs,A] / [A] / int
    ):
        """
        返回: (logp [N], entropy [N], values [N])
        """
        device = self.device
        states = states.to(device)
        m = mask.to(device) if mask is not None else None

        N, T, F = states.shape

        # 归一化
        mean = self.state_rms.mean.to(device)
        var = self.state_rms.var.to(device)
        xn = normalize_columns(states, mean, var, cols=self.norm_cols)

        # 前向
        logits = self.actor(xn, mask=m)  # [N, num_agvs, action_dim]

        # ===== 与 choose_action 一致的动作掩码 =====
        if action_mask is not None:
            if isinstance(action_mask, int):
                idxs = torch.arange(self.action_dim, device=device)
                invalid = idxs >= int(action_mask)  # [A] bool (True=禁用)
                invalid = invalid.view(1, 1, -1).expand(N, self.num_agvs, -1)
                logits = logits.masked_fill(invalid, float("-1e9"))
            else:
                am = torch.as_tensor(action_mask, dtype=torch.bool, device=device)  # True=可用
                if am.dim() == 1:  # [A]
                    am = am.view(1, 1, -1).expand(N, self.num_agvs, -1)
                elif am.dim() == 2:  # [num_agvs, A]
                    if am.size(0) != self.num_agvs or am.size(1) != self.action_dim:
                        raise ValueError(
                            f"action_mask shape {am.shape} != [{self.num_agvs},{self.action_dim}]")
                    am = am.view(1, self.num_agvs, -1).expand(N, -1, -1)
                elif am.dim() == 3:  # [N, num_agvs, A]
                    if (am.size(0) != N) or (am.size(1) != self.num_agvs) or (
                            am.size(2) != self.action_dim):
                        raise ValueError(
                            f"action_mask shape {am.shape} != [{N},{self.num_agvs},{self.action_dim}]")
                else:
                    raise ValueError(f"Unsupported action_mask dim: {am.dim()}")
                logits = logits.masked_fill(~am, float("-1e9"))  # 不可用置 -inf
        # ===== 掩码结束 =====

        # 统一动作形状
        if actions.dim() == 1:
            actions = actions.unsqueeze(1).expand(-1, self.num_agvs).to(device)
        elif actions.dim() == 2:
            actions = actions.to(device)
        else:
            raise ValueError(f"Unsupported actions shape: {actions.shape}")

        # 联合 logp / entropy（对每个 AGV 独立，然后合并）
        logits_flat = logits.reshape(-1, logits.shape[-1])  # [N*num_agvs, A]
        acts_flat = actions.reshape(-1)  # [N*num_agvs]

        dist = Categorical(logits=logits_flat)
        logp_f = dist.log_prob(acts_flat)  # [N*num_agvs]
        ent_f = dist.entropy()  # [N*num_agvs]

        logp = logp_f.view(N, self.num_agvs).sum(dim=1)  # [N]
        ent = ent_f.view(N, self.num_agvs).mean(dim=1)  # [N]

        values = self.critic(xn, mask=m).squeeze(-1)  # [N]

        return logp, ent, values

    # ---------- save / load ----------
    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "rms_mean": self.state_rms.mean.cpu(),
            "rms_var": self.state_rms.var.cpu(),
            "rms_count": self.state_rms.count.cpu(),
            "use_transformer": self.use_transformer,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "num_layers": self.num_layers,
        }, path)

    def load(self, path: str, map_location: Optional[str] = None):
        data = torch.load(path, map_location=map_location or (self.device.type))
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        try:
            self.state_rms.mean = data.get("rms_mean", self.state_rms.mean).to(self.device)
            self.state_rms.var = data.get("rms_var", self.state_rms.var).to(self.device)
            self.state_rms.count = data.get("rms_count", self.state_rms.count).to(self.device)
        except Exception:
            pass
