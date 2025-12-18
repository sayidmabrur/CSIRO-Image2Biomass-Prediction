"""
Training configuration dataclass for Image2Biomass model.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training the Image2Biomass model."""

    # ========== Data Paths ==========
    dataset_path: str = "datasets"
    """Path to the dataset directory containing train.csv and images"""

    output_dir: str = "train_results"
    """Directory to save training outputs (models, logs, etc.)"""

    # architecture: str = "dinov3-convnext-tiny-pretrain-lvd1689m"
    architecture: str = "dinov3-convnext-large-pretrain-lvd1689m"
    """Pretrained model architecture"""

    pretrained_model_path: str = "pretrained/dinov3-convnext-large-pretrain-lvd1689m"
    """Path to pretrained DINOv3 ConvNeXt Large model weights"""

    # ========== Training Hyperparameters ==========
    epochs: int = 100
    """Number of training epochs"""

    batch_size: int = 16
    """Batch size for training and validation"""

    learning_rate: float = 1e-4
    """Learning rate for AdamW optimizer"""

    weight_decay: float = 1e-2
    """Weight decay for AdamW optimizer"""

    l1_lambda: float = 1e-7
    """L1 regularization lambda"""

    grad_clip_norm: float = 1.0
    """Maximum norm for gradient clipping"""

    # ========== Cross-Validation ==========
    n_folds: int = 5
    """Number of folds for K-fold cross-validation"""

    seed: int = 42
    """Random seed for reproducibility"""

    # ========== Device and Performance ==========
    device: str = "cuda"
    """Device to use for training ('cuda' or 'cpu')"""

    num_workers: int = 2
    """Number of worker processes for data loading"""

    pin_memory: bool = True
    """Whether to pin memory for faster GPU transfer"""

    drop_last: bool = True
    """Whether to drop last incomplete batch in training"""

    # ========== Weights & Biases ==========
    use_wandb: bool = True
    """Whether to use Weights & Biases for logging"""

    wandb_project: str = "IMAGE2BIOMASSPREDICTION"
    """W&B project name"""

    wandb_group: str = "5fold-cv-dinov3"
    """W&B group name for organizing runs"""

    wandb_api_key: Optional[str] = None
    """W&B API key (optional, can also use environment variable)"""

    # ========== Logging ==========
    print_every: int = 10
    """Print training progress every N epochs"""

    log_freq: int = 100
    """Frequency for W&B gradient logging"""

    scheduler: str = "cosine"
