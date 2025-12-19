import torch.nn as nn
import torch.nn.functional as F
import torch

class RMSNorm(nn.Module):
    def __init__(self, ndim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x) * self.weight
