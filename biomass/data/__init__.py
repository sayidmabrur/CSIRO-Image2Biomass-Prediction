"""
CSIRO Biomass Training Data Module

This module provides dataset classes, augmentation transforms, and configuration
for training models to predict biomass from plant images.
"""

from .config import DataConfig
from .augmentations import (
    train_transform,
    val_transform,
    numeric_transform,
    target_transform,
    target_untransform,
    categorical_transform,
)
from .datasets import (
    Image2BioMassTrainValDataset,
    Image2BioMassTestDataset,
    create_kfold_datasets,
    create_dataloaders,
)

__all__ = [
    # Config
    "DataConfig",
    # Transforms
    "train_transform",
    "val_transform",
    "numeric_transform",
    "target_transform",
    "target_untransform",
    "categorical_transform",
    # Datasets
    "Image2BioMassTrainValDataset",
    "Image2BioMassTestDataset",
    # Helper functions
    "create_kfold_datasets",
    "create_dataloaders",
]
