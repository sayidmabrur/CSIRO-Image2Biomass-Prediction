"""
Configuration class for dataset and dataloader hyperparameters.
"""

import torch


class DataConfig:
    """Configuration for dataset and dataloader parameters."""

    # Image settings
    img_size = (224, 224)
    dtype = torch.float32

    # ImageNet normalization statistics
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    # Dataloader settings
    batch_size = 16
    num_workers = 4
    pin_memory = True

    # Cross-validation settings
    n_folds = 5
    random_seed = 42

    # Augmentation settings (for reference, actual transforms in augmentations.py)
    train_resize = (256, 256)  # Larger size for random crop
    train_crop_scale = (0.75, 1.0)
    train_crop_ratio = (0.9, 1.1)

    # Color augmentation parameters
    brightness = 0.35
    contrast = 0.35
    saturation = 0.35
    hue = 0.1

    # Feature dimensions
    num_categorical_features = 3  # Sampling_Date, State, Species
    num_numeric_features = 2  # Pre_GSHH_NDVI, Height_Ave_cm
    num_targets = 3  # Dry_Green_g, Dry_Dead_g, Dry_Clover_g

    @classmethod
    def get_total_features(cls):
        """Get total number of combined features."""
        return cls.num_categorical_features + cls.num_numeric_features
