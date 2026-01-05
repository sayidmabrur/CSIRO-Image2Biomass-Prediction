"""
Complete Image2Biomass model extracted from notebook.
"""

import torch
import torch.nn as nn
from biomass.nets.backbone import DINOV3Backbone


class WeightedHuberLoss(nn.Module):
    """Weighted Huber Loss for biomass prediction."""

    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
        # Weights: [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
        self.register_buffer(
            "weights", torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], dtype=torch.float32)
        )

    def forward(self, x, y):
        # Only use first 3 weights if predicting 3 targets
        w = self.weights[: x.shape[-1]]
        loss = torch.where(
            torch.abs(y - x) < self.delta,
            0.5 * (y - x) ** 2,
            self.delta * (torch.abs(y - x) - 0.5 * self.delta),
        )
        return (loss * w).mean()


class Image2BiomassModel(nn.Module):
    """
    Image to Biomass prediction model using DINOv3 ConvNeXt Large backbone.

    This model predicts five biomass targets:
    - Dry_Green_g
    - Dry_Dead_g
    - Dry_Clover_g
    - GDM_g (Green Dry Matter: should equal Dry_Green_g + Dry_Clover_g)
    - Dry_Total_g (Total Dry Matter: should equal Dry_Green_g + Dry_Dead_g + Dry_Clover_g)

    The model learns to predict all 5 targets directly, allowing it to learn the relationships
    between component and aggregate measures in a more general way.
    """

    def __init__(self):
        """
        Initialize the model.

        Args:
            pretrained_model_path: Path to pretrained DINOv3 ConvNeXt Large model weights
        """
        super().__init__()
        # Load DINOv3 ConvNeXt Large backbone
        self.backbone = DINOV3Backbone()
        self.eps = 1e-8
        self.fc1 = nn.Sequential(
            nn.Linear(1536, 1536, bias=False),  # dinov3 convnext large
            nn.BatchNorm1d(1536),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(1536, 1024, bias=False),  # dinov3 convnext large
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Dropout(0.1),
        )

        self.out = nn.Linear(1024, 3, bias=True)

        # self.criterion = WeightedHuberLoss(delta=1.0)
        self.criterion = nn.SmoothL1Loss()

    def forward(self, x, y=None):
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W)
            y: Optional target values (B, 5) for loss calculation
               [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]

        Returns:
            Tuple of (predictions, loss)
            - predictions: (B, 5) tensor of biomass predictions
            - loss: Scalar loss value (None if y is None)
        """

        x = self.backbone(x)
        x = self.fc1(x)
        preds = self.out(x)
        preds = torch.nn.functional.softplus(preds)

        loss = None
        if y is not None:
            y = y[:, :3]
            loss = self.criterion(preds, y)

        return preds, loss
