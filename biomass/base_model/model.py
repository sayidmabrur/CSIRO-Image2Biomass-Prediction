"""
Complete Image2Biomass model extracted from notebook.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


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

    This model predicts three biomass targets:
    - Dry_Green_g
    - Dry_Dead_g
    - Dry_Clover_g
    """

    def __init__(self, pretrained_model_path: str = None):
        """
        Initialize the model.

        Args:
            pretrained_model_path: Path to pretrained DINOv3 ConvNeXt Large model weights
        """
        super().__init__()
        # Load DINOv3 ConvNeXt Large backbone
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_path, trust_remote_code=False
        )

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # DINOv3 ConvNeXt outputs features from pooler
        # Tiny model = 768 dims, Large model = 1536 dims
        # The actual dimension depends on which model is loaded
        self.fc1 = nn.Sequential(
            nn.Linear(1536, 1024),  # dinov3 convnext large
            # nn.Linear(768, 512), # dinov3 convenext tiny
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Dropout(0.1),
        )


        self.out = nn.Linear(64, 5)

        # self.criterion = nn.SmoothL1Loss(reduction="mean")
        self.criterion = WeightedHuberLoss(delta=1.0)

    def forward(self, x, y=None):
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W)
            y: Optional target values (B, 3) for loss calculation

        Returns:
            Tuple of (predictions, loss)
            - predictions: (B, 3) tensor of biomass predictions
            - loss: Scalar loss value (None if y is None)
        """
        # DINOv3 ConvNeXt expects normalized images and outputs pooled features
        outputs = self.backbone(x)
        # Use the pooler_output which is the global representation (1536-dim for ConvNeXt Large)
        x = outputs.pooler_output

        x = self.fc1(x)

        preds = self.out(x)

        loss = None
        if y is not None:
            loss = self.criterion(preds, y)

        return preds, loss
