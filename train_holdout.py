"""
Main entry point for holdout validation training of the Image2Biomass model.

This script uses a simple 80:20 stratified train/validation split by Species
instead of K-fold cross-validation. This is useful when you want to train
on most of the data while still having a validation set for monitoring.

Usage:
    # Train with default settings (90:10 split)
    python train_holdout.py

    # Custom hyperparameters
    python train_holdout.py --epochs 500 --batch-size 16 --learning-rate 5e-5

    # See all options
    python train_holdout.py --help
"""

import tyro
from biomass.config import TrainingConfig
from biomass.train.train import run_holdout_validation


def main(config: TrainingConfig):
    """
    Train Image2Biomass model with holdout validation (90:10 stratified split).

    Args:
        config: Training configuration with all hyperparameters
    """
    print("=" * 80)
    print("Image2Biomass Holdout Validation Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset path: {config.dataset_path}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Train/Val Split: 80:20 (stratified by Species)")
    print(f"  Device: {config.device}")
    print(f"  W&B enabled: {config.use_wandb}")
    print(f"  LR Scheduler: {config.scheduler}")
    print("=" * 80 + "\n")

    # Run holdout validation training
    results = run_holdout_validation(config)

    print("\nTraining completed successfully!")
    return results


if __name__ == "__main__":
    # Parse configuration from command line using tyro
    config = tyro.cli(TrainingConfig)
    main(config)
