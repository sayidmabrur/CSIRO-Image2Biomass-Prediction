"""
CSIRO Biomass Training Package

This package contains all the modules needed for training the Image2Biomass model.
"""

from .base_model import Image2BiomassModel
from .config import TrainingConfig
from .data import (
    DataConfig,
    train_transform,
    val_transform,
    Image2BioMassTrainValDataset,
    Image2BioMassTestDataset,
)
from .eval.metrics import weighted_r2, weighted_r2_single
from .train import run_cross_validation

__all__ = [
    # Model
    "Image2BiomassModel",
    # Config
    "TrainingConfig",
    "DataConfig",
    # Data
    "train_transform",
    "val_transform",
    "Image2BioMassTrainValDataset",
    "Image2BioMassTestDataset",
    # Metrics
    "weighted_r2",
    "weighted_r2_single",
    # Training
    "run_cross_validation",
]
