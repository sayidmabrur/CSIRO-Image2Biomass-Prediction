"""
Data augmentation and transformation functions for the CSIRO Biomass dataset.
"""

import torch
from torchvision.transforms import v2
from .config import DataConfig


# Initialize configuration
config = DataConfig()

# Aggressive training transform for small dataset
train_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(config.dtype, scale=True),
        v2.Resize(config.img_size),  # Larger for cropping
        # Geometric augmentations
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        # v2.RandomRotation(180, interpolation=v2.InterpolationMode.BILINEAR),
        # v2.RandomAffine(
        #     degrees=0,
        #     translate=(0.1, 0.1),
        #     scale=(0.85, 1.15),
        #     shear=10,
        #     interpolation=v2.InterpolationMode.BILINEAR,
        # ),
        # v2.RandomResizedCrop(
        #     size=config.img_size,
        #     scale=config.train_crop_scale,
        #     ratio=config.train_crop_ratio,
        #     interpolation=v2.InterpolationMode.BILINEAR,
        # ),
        # # Color augmentations
        # v2.ColorJitter(
        #     brightness=config.brightness,
        #     contrast=config.contrast,
        #     saturation=config.saturation,
        #     hue=config.hue,
        # ),
        # v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        # v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        # v2.RandomAutocontrast(p=0.2),
        # v2.RandomGrayscale(p=0.05),
        v2.Normalize(
            mean=config.imagenet_mean,
            std=config.imagenet_std,
        ),
    ]
)

# Clean validation transform
val_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(config.dtype, scale=True),
        v2.Resize(config.img_size),
        v2.Normalize(
            mean=config.imagenet_mean,
            std=config.imagenet_std,
        ),
    ]
)


def numeric_transform(X, X_max, X_min) -> torch.Tensor:
    """
    Min-max normalization for numeric features.

    Args:
        X: Input value(s)
        X_max: Maximum value for normalization
        X_min: Minimum value for normalization

    Returns:
        Normalized value(s) in range [0, 1]
    """
    X_normalized = (X - X_min) / (X_max - X_min)
    return X_normalized


def target_transform(targets) -> torch.Tensor:
    """
    Apply log1p transformation to target values.
    This helps with the skewed distribution of biomass values.

    Args:
        targets: Raw target values

    Returns:
        Log-transformed targets
    """
    return torch.log1p(targets)


def target_untransform(targets) -> torch.Tensor:
    """
    Inverse of target_transform using expm1.
    Use this to convert predictions back to original scale.

    Args:
        targets: Log-transformed targets

    Returns:
        Original scale targets
    """
    return torch.expm1(targets)


def categorical_transform(row) -> torch.Tensor:
    """
    Transform categorical features.
    Currently a passthrough (identity function).

    Args:
        row: Categorical feature values

    Returns:
        Unchanged categorical features
    """
    return row
