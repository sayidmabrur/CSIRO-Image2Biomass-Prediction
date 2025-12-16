"""
Complete Image2Biomass model extracted from notebook.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


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

        # Default path if not provided
        if pretrained_model_path is None:
            pretrained_model_path = "/mnt/d/Sayid/Projects/Image2Biomass/CSIRO-Image2Biomass-Prediction/pretrained/dinov3-convnext-large/weights"

        # Load DINOv3 ConvNeXt Large backbone
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_path, trust_remote_code=False
        )

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # DINOv3 ConvNeXt Large outputs 1536-dim features (from the pooler)
        self.fc1 = nn.Sequential(
            nn.Linear(1536, 1024),
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

        self.out = nn.Linear(64, 3)

        self.criterion = nn.SmoothL1Loss(reduction="mean")

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
