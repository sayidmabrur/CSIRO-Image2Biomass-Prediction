import torch
import torch.nn as nn
from transformers import AutoModel
from dataclasses import dataclass


@dataclass
class BackboneConfig:
    """
    Configuration class for DINOv3 ConvNeXt backbone.

    Args:
        pretrained_model_path (str): Path to the pretrained model directory
        feature_dim (int): Dimension of output features (768 for tiny, 768 for small)
        freeze_backbone (bool): Whether to freeze the backbone parameters during training
        image_size (int): Input image size (default: 224)
        num_channels (int): Number of input channels (default: 3 for RGB)
    """

    pretrained_model_path: str = "pretrained/dinov3-convnext-large-pretrain-lvd1689m"
    feature_dim: int = 1536
    freeze_backbone: bool = True


class DINOV3Backbone(nn.Module):
    """
    DINOv3 ConvNeXt backbone for feature extraction.

    Architecture:
    - Input: (B, 3, 224, 224) - RGB images (224x224 is the standard input size)
    - Output: (B, 768) - Pooled features for downstream tasks

    The ConvNeXt model outputs 768-dimensional pooled features (for tiny model) that can be
    used for various downstream tasks like regression, classification, or ensemble learning.
    """

    def __init__(self, config: BackboneConfig = None):
        """
        Args:
            config (BackboneConfig): Configuration object for the backbone.
                                    If None, uses default configuration.
        """
        super().__init__()

        # Use default config if none provided
        if config is None:
            config = BackboneConfig()

        self.config = config

        # Load pretrained DINOv3 ConvNeXt model
        self.dinov3 = AutoModel.from_pretrained(
            config.pretrained_model_path, trust_remote_code=False
        )

        # Optionally freeze backbone parameters
        if config.freeze_backbone:
            for param in self.dinov3.parameters():
                param.requires_grad = False

    def forward(self, x):
        """num_channels, image_size, image_size
        Forward pass through the backbone.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, 224, 224)

        Returns:
            torch.Tensor: Pooled features of shape (B, feature_dim)
        """
        outputs = self.dinov3(x)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state.mean(dim=1)

        return features


class EfficientNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        pass

    def forward(self):
        pass
