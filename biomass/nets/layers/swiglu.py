import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim, bias: bool = True) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, out_dim, bias=bias)
        self.w3 = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return F.silu(self.w1(x)) * self.w3(x)
