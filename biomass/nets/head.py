import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional
from .backbone import BackBone, BackboneConfig


@dataclass
class HeadConfig:
    """
    Configuration class for the MLP head.

    Args:
        input_dim (int): Dimension of input features from backbone (default: 768)
        hidden_dims (List[int]): List of hidden layer dimensions
        output_dim (int): Dimension of output (default: 3 for Biomass: Green, Dead, Clover)
        dropout_rate (float): Dropout probability
        use_batch_norm (bool): Whether to use BatchNorm instead of LayerNorm
    """

    input_dim: int = 768
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    output_dim: int = 3
    dropout_rate: float = 0.3
    use_batch_norm: bool = False


class MLPHead(nn.Module):
    """
    MLP Head with Mish activation, LayerNorm/BatchNorm, and Dropout.
    """

    def __init__(self, config: HeadConfig):
        super().__init__()

        layers = []
        current_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))

            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(nn.Mish())
            layers.append(nn.Dropout(config.dropout_rate))
            current_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(current_dim, config.output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BiomassPredictor(nn.Module):
    """
    Combined model with DINOv3 backbone and MLP head for biomass prediction.
    """

    def __init__(
        self,
        backbone_config: Optional[BackboneConfig] = None,
        head_config: Optional[HeadConfig] = None,
    ):
        super().__init__()

        if backbone_config is None:
            backbone_config = BackboneConfig()
        if head_config is None:
            head_config = HeadConfig(input_dim=backbone_config.feature_dim)

        self.backbone = BackBone(backbone_config)
        self.head = MLPHead(head_config)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        features = self.backbone(x)  # (B, feature_dim)
        output = self.head(features)  # (B, output_dim)
        return output
