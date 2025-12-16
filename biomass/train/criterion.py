import torch
import torch.nn as nn
biomass.train.metrics import target_weights


class WeightedHuberLoss(torch.nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, x, y):
        loss = torch.where(
            torch.abs(y - x) < self.delta,  # | y - x | < delta
            0.5 * (y - x) ** 2,
            self.delta * (torch.abs(y - x) - 0.5 * self.delta),
        )

        return (loss * weights).mean()
