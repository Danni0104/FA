# model/core.py
import torch
import torch.nn as nn

class RunningMeanStd:
    """
    维护逐特征的均值与方差：
      - shape: (F,) 或与特征维度一致
      - eps:   初始计数的稳定项
      - device: 放在哪个设备上
    提供：update(x), normalize(x), denormalize(x)
    """
    def __init__(self, shape=(1,), eps: float = 1e-4, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self.mean   = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var    = torch.ones(shape,  dtype=torch.float32, device=self.device)
        # 用 tensor 存计数，首批要能正确判断
        self.count  = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.eps    = float(eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """
        x: [N, F] 或 [*, F]，只在 dim=0 上做 batch 统计
        """
        if x is None or x.numel() == 0:
            return

        x = x.to(self.device, dtype=torch.float32)
        # 把除了最后一维以外都展平
        x = x.view(-1, x.shape[-1])  # [N, F]

        batch_mean  = x.mean(dim=0)
        batch_var   = x.var(dim=0, unbiased=False)  # 与运行时一致
        batch_count = torch.tensor(float(x.size(0)), device=self.device)

        if float(self.count.item()) == 0.0:
            # 首批：直接赋值（加上一点 eps 稳定项）
            self.mean  = batch_mean
            self.var   = batch_var.clamp_min(self.eps)
            self.count = batch_count
            return

        delta     = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a      = self.var * self.count
        m_b      = batch_var * batch_count
        # 合并方差（并行 Welford）
        M2       = m_a + m_b + (delta**2) * (self.count * batch_count / tot_count)
        new_var  = (M2 / tot_count).clamp_min(self.eps)

        self.mean  = new_mean
        self.var   = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.view(*([1] * (x.dim() - 1)), -1)
        var  = self.var.view(*([1] * (x.dim() - 1)), -1)
        return (x - mean) / torch.sqrt(var + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.view(*([1] * (x.dim() - 1)), -1)
        var  = self.var.view(*([1] * (x.dim() - 1)), -1)
        return x * torch.sqrt(var + 1e-8) + mean


def orthogonal_init(module: nn.Module, gain: float = 1.0):
    """
    对 Linear / Conv1d/2d/3d 做正交初始化，bias=0。
    用法：
        model.apply(lambda m: orthogonal_init(m, gain=1.0))
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LayerNorm):
        # 常见做法：权重=1，偏置=0
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)
