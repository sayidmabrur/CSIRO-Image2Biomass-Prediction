"""
Main entry point for training the Image2Biomass model.

This script provides a CLI interface using tyro for easy configuration and training.

Usage:
    # Train with default settings
    python main.py

    # Custom hyperparameters
    python main.py --epochs 500 --batch-size 16 --learning-rate 5e-5

    # Specify paths
    python main.py --dataset-path /path/to/data --output-dir my_results

    # See all options
    python main.py --help
"""

import tyro
from biomass.config import TrainingConfig
from biomass.train.train import run_cross_validation


def main(config: TrainingConfig):
    """
    Train Image2Biomass model with K-fold cross-validation.

    Args:
        config: Training configuration with all hyperparameters
    """
    print("=" * 80)
    print("Image2Biomass Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset path: {config.dataset_path}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  K-Folds: {config.n_folds}")
    print(f"  Device: {config.device}")
    print(f"  W&B enabled: {config.use_wandb}")
    print("=" * 80 + "\n")

    # Run cross-validation training
    results = run_cross_validation(config)

    print("\nTraining completed successfully!")
    return results


if __name__ == "__main__":
    # Parse configuration from command line using tyro
    config = tyro.cli(TrainingConfig)
    main(config)
