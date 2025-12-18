"""
Training functions for the Image2Biomass model.

This module contains all the training logic extracted from the Jupyter notebook,
organized into modular, reusable functions.
"""

import os
import time
import random
import numpy as np
import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold

from ..config.train_config import TrainingConfig
from ..data import (
    Image2BioMassTrainValDataset,
    train_transform,
    val_transform,
    numeric_transform,
    target_transform,
)
from biomass.eval.metrics import weighted_r2, weighted_r2_single
from biomass.base_model import Image2BiomassModel


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    weights,
    config: TrainingConfig,
    epoch: int,
    fold_idx: int,
):
    """
    Train the model for one epoch.

    Args:
        model: The neural network model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        weights: Weights for R2 calculation
        config: Training configuration
        epoch: Current epoch number
        fold_idx: Current fold index

    Returns:
        Tuple of (avg_loss, avg_r2, r2_per_target_dict)
    """
    model.train()
    train_loss = 0
    train_r2_scores = []
    train_r2_individual = {
        "Dry_Green_g": [],
        "Dry_Dead_g": [],
        "Dry_Clover_g": [],
        "GDM_g": [],
        "Dry_Total_g": [],
    }

    for imgs, _, y in tqdm(
        dataloader, desc=f"[Fold {fold_idx + 1}] Train Epoch {epoch}", leave=False
    ):
        imgs, y = imgs.to(device), y.to(device)
        # print("y:", y)

        preds, loss = model(imgs, y)

        # huber loss already applied L1 regularization
        # L1 regularization
        # reg_loss = sum(param.abs().sum() for param in model.parameters())
        # loss = loss + config.l1_lambda * reg_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()

        train_loss += loss.item()
        train_r2_scores.append(weighted_r2(y, preds, weights).item())

        # Track individual R2 scores
        r2_dict = weighted_r2_single(y, preds)
        for target_name, r2_value in r2_dict.items():
            train_r2_individual[target_name].append(r2_value)

    avg_train_loss = train_loss / len(dataloader)
    avg_train_r2 = sum(train_r2_scores) / len(train_r2_scores)
    avg_train_r2_individual = {
        k: sum(v) / len(v) for k, v in train_r2_individual.items()
    }

    return avg_train_loss, avg_train_r2, avg_train_r2_individual


def validate_one_epoch(model, dataloader, device, weights, epoch: int, fold_idx: int):
    """
    Validate the model for one epoch.

    Args:
        model: The neural network model
        dataloader: Validation data loader
        device: Device to validate on
        weights: Weights for R2 calculation
        epoch: Current epoch number
        fold_idx: Current fold index

    Returns:
        Tuple of (avg_loss, avg_r2, r2_per_target_dict)
    """
    model.eval()
    val_loss = 0
    val_r2_scores = []
    val_r2_individual = {
        "Dry_Green_g": [],
        "Dry_Dead_g": [],
        "Dry_Clover_g": [],
        "GDM_g": [],
        "Dry_Total_g": [],
    }

    with torch.no_grad():
        for imgs, _, y in tqdm(
            dataloader, desc=f"[Fold {fold_idx + 1}] Val Epoch {epoch}", leave=False
        ):
            imgs, y = imgs.to(device), y.to(device)
            preds, loss = model(imgs, y)
            val_loss += loss.item()
            val_r2_scores.append(weighted_r2(y, preds, weights).item())

            # Track individual R2 scores
            r2_dict = weighted_r2_single(y, preds)
            for target_name, r2_value in r2_dict.items():
                val_r2_individual[target_name].append(r2_value)

    avg_val_loss = val_loss / len(dataloader)
    avg_val_r2 = sum(val_r2_scores) / len(val_r2_scores)
    avg_val_r2_individual = {k: sum(v) / len(v) for k, v in val_r2_individual.items()}

    return avg_val_loss, avg_val_r2, avg_val_r2_individual


def train_fold(
    fold_idx: int,
    train_indices,
    val_indices,
    base_dataset,
    config: TrainingConfig,
    device,
    weights,
):
    """
    Train the model for one fold in K-fold cross-validation.

    Args:
        fold_idx: Index of the current fold
        train_indices: Indices for training data
        val_indices: Indices for validation data
        base_dataset: Base dataset object
        config: Training configuration
        device: Device to train on
        weights: Weights for R2 calculation

    Returns:
        Dictionary containing fold results
    """
    print(f"\n{'=' * 80}")
    print(f"FOLD {fold_idx + 1}/{config.n_folds}")
    print(f"{'=' * 80}")

    # Create fold directory
    fold_dir = f"{config.output_dir}/fold{fold_idx + 1}"
    os.makedirs(fold_dir, exist_ok=True)

    # Reset seed for each fold
    set_seed(config.seed + fold_idx)

    # Create datasets for this fold with proper transforms
    print("Creating train dataset for this fold...")
    train_dataset_fold = Image2BioMassTrainValDataset(
        dataset_path=config.dataset_path,
        indices=train_indices,
        img_transform=train_transform,
        numeric_transform=numeric_transform,
        target_transform=target_transform,
        df=base_dataset.df.copy(),
        label_encoders=base_dataset.get_label_encoders(),
        numeric_stats=base_dataset.numeric_stats,
    )

    print("Creating validation dataset for this fold...")
    val_dataset_fold = Image2BioMassTrainValDataset(
        dataset_path=config.dataset_path,
        indices=val_indices,
        img_transform=val_transform,
        numeric_transform=numeric_transform,
        target_transform=target_transform,
        df=base_dataset.df.copy(),
        label_encoders=base_dataset.get_label_encoders(),
        numeric_stats=base_dataset.numeric_stats,
    )

    # Create generator for reproducibility
    g = torch.Generator()
    g.manual_seed(config.seed)

    # Create dataloaders for this fold
    train_dataloader = DataLoader(
        train_dataset_fold,
        batch_size=config.batch_size,
        shuffle=True,
        generator=g,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )

    val_dataloader = DataLoader(
        val_dataset_fold,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")

    # Initialize NEW model for this fold (critical for K-Fold!)
    print("Loading backbone model (DINOv3 ConvNeXt Large)...")
    model = Image2BiomassModel(config.pretrained_model_path).to(device)
    print("Model loaded and moved to device")
    print("Initializing optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    print("Initializing scheduler...")
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.epochs // 2, gamma=0.1
        )
    else:
        scheduler = None

    # Initialize W&B for this fold if enabled
    print(f"W&B logging: {'enabled' if config.use_wandb else 'disabled'}")
    if config.use_wandb:
        _ = wandb.init(
            project=config.wandb_project,
            name=f"fold{fold_idx + 1}_{config.architecture}",
            group=config.wandb_group,
            job_type=f"fold{fold_idx + 1}",
            tags=[
                "dinov3",
                "convnext-large",
                f"{config.n_folds}fold-cv",
                f"fold{fold_idx + 1}",
            ],
            config={
                "architecture": "dinov3-convnext-large",
                "backbone": "DINOv3 ConvNeXt Large (198M params)",
                "dataset": "Image2Biomass",
                "epochs": config.epochs,
                "fold": fold_idx + 1,
                "n_folds": config.n_folds,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "optimizer": "AdamW",
                "loss_function": "SmoothL1Loss",
                "l1_regularization": config.l1_lambda,
                "gradient_clip": config.grad_clip_norm,
                "train_samples": len(train_dataset_fold),
                "val_samples": len(val_dataset_fold),
            },
            reinit=True,
        )

        # Log model architecture
        wandb.watch(model, log="all", log_freq=config.log_freq, log_graph=True)

    # Training tracking
    best_val_r2 = -float("inf")
    best_epoch = 0
    best_val_r2_individual = None
    train_losses, val_losses = [], []
    train_r2_history, val_r2_history = [], []
    fold_start_time = time.time()

    # Training loop for this fold
    print(f"\nStarting training loop for {config.epochs} epochs...\n")
    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()

        # Train
        avg_train_loss, avg_train_r2, avg_train_r2_individual = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            weights,
            config,
            epoch,
            fold_idx,
        )

        # Validate
        avg_val_loss, avg_val_r2, avg_val_r2_individual = validate_one_epoch(
            model, val_dataloader, device, weights, epoch, fold_idx
        )

        val_losses.append(avg_val_loss)
        train_losses.append(avg_train_loss)
        val_r2_history.append(avg_val_r2)
        train_r2_history.append(avg_train_r2)

        epoch_time = time.time() - epoch_start_time

        # Save best model for this fold
        is_best = False
        if avg_val_r2 > best_val_r2:
            best_val_r2 = avg_val_r2
            best_epoch = epoch
            best_val_r2_individual = avg_val_r2_individual.copy()
            best_model_path = f"{fold_dir}/best.pth"
            torch.save(model.state_dict(), best_model_path)
            is_best = True
            print(f"New best model saved! Val R2: {best_val_r2:.4f} at epoch {epoch}")

        # Log to W&B if enabled
        if config.use_wandb:
            log_dict = {
                "fold": fold_idx + 1,
                "epoch": epoch,
                "epoch_time": epoch_time,
                "train/loss": avg_train_loss,
                "train/r2_weighted": avg_train_r2,
                "val/loss": avg_val_loss,
                "val/r2_weighted": avg_val_r2,
                "val/best_r2": best_val_r2,
                "train/r2_dry_green": avg_train_r2_individual["Dry_Green_g"],
                "train/r2_dry_dead": avg_train_r2_individual["Dry_Dead_g"],
                "train/r2_dry_clover": avg_train_r2_individual["Dry_Clover_g"],
                "train/r2_gdm": avg_train_r2_individual["GDM_g"],
                "train/r2_dry_total": avg_train_r2_individual["Dry_Total_g"],
                "val/r2_dry_green": avg_val_r2_individual["Dry_Green_g"],
                "val/r2_dry_dead": avg_val_r2_individual["Dry_Dead_g"],
                "val/r2_dry_clover": avg_val_r2_individual["Dry_Clover_g"],
                "val/r2_gdm": avg_val_r2_individual["GDM_g"],
                "val/r2_dry_total": avg_val_r2_individual["Dry_Total_g"],
                "metrics/train_val_loss_diff": avg_train_loss - avg_val_loss,
                "metrics/train_val_r2_diff": avg_train_r2 - avg_val_r2,
                "metrics/is_best_epoch": int(is_best),
                "optimizer/learning_rate": optimizer.param_groups[0]["lr"],
            }
            wandb.log(log_dict)

        # Print progress
        if epoch % config.print_every == 0 or epoch == 1:
            print(f"\nEpoch {epoch}/{config.epochs} | Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train R2: {avg_train_r2:.4f}")
            print(
                f"  Val Loss: {avg_val_loss:.4f} | Val R2: {avg_val_r2:.4f} | Best: {best_val_r2:.4f}"
            )
            print(
                f"  Train R2 -> Green: {avg_train_r2_individual['Dry_Green_g']:.4f}, "
                f"Dead: {avg_train_r2_individual['Dry_Dead_g']:.4f}, "
                f"Clover: {avg_train_r2_individual['Dry_Clover_g']:.4f}, "
                f"GDM: {avg_train_r2_individual['GDM_g']:.4f}, "
                f"Total: {avg_train_r2_individual['Dry_Total_g']:.4f}"
            )
            print(
                f"  Val R2   -> Green: {avg_val_r2_individual['Dry_Green_g']:.4f}, "
                f"Dead: {avg_val_r2_individual['Dry_Dead_g']:.4f}, "
                f"Clover: {avg_val_r2_individual['Dry_Clover_g']:.4f}, "
                f"GDM: {avg_val_r2_individual['GDM_g']:.4f}, "
                f"Total: {avg_val_r2_individual['Dry_Total_g']:.4f}"
            )

    # Save last model for this fold
    last_model_path = f"{fold_dir}/last.pth"
    torch.save(model.state_dict(), last_model_path)

    fold_time = time.time() - fold_start_time

    # Store fold results
    fold_results = {
        "fold": fold_idx + 1,
        "best_val_r2": best_val_r2,
        "best_epoch": best_epoch,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "final_train_r2": avg_train_r2,
        "final_val_r2": avg_val_r2,
        "fold_time": fold_time,
        "best_val_r2_per_target": best_val_r2_individual,
    }

    # Log fold summary to W&B
    if config.use_wandb:
        wandb.run.summary["fold"] = fold_idx + 1
        wandb.run.summary["best_val_r2"] = best_val_r2
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["fold_time_hours"] = fold_time / 3600
        wandb.run.summary["final_train_r2"] = avg_train_r2
        wandb.run.summary["final_val_r2"] = avg_val_r2

        for target_name, r2_value in best_val_r2_individual.items():
            wandb.run.summary[f"best_val_r2_{target_name}"] = r2_value

        wandb.finish()

    print(f"\n{'=' * 80}")
    print(f"Fold {fold_idx + 1} completed! Time: {fold_time / 3600:.2f} hours")
    print(f"Best validation R2: {best_val_r2:.4f} achieved at epoch {best_epoch}")
    print("Best per-target R2:")
    for target_name, r2_value in best_val_r2_individual.items():
        print(f"  {target_name}: {r2_value:.4f}")
    print(f"Models saved in: {fold_dir}/")
    print(f"  - best.pth (epoch {best_epoch})")
    print(f"  - last.pth (epoch {config.epochs})")
    print(f"{'=' * 80}\n")

    return fold_results


def run_cross_validation(config: TrainingConfig):
    """
    Run K-fold cross-validation training.

    Args:
        config: Training configuration

    Returns:
        List of fold results dictionaries
    """
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create results directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Setup wandb if enabled
    if config.use_wandb and config.wandb_api_key:
        os.environ["WANDB_API_KEY"] = config.wandb_api_key
        wandb.login(key=config.wandb_api_key)

    # Set seed
    set_seed(config.seed)

    # Weights for R2 calculation
    weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], device=device)

    # Create base dataset to get full DataFrame and encoders
    print("\nLoading base dataset...")
    base_dataset = Image2BioMassTrainValDataset(
        dataset_path=config.dataset_path,
        img_transform=None,
        numeric_transform=None,
        target_transform=None,
    )

    print("Base dataset loaded")
    print(f"\nDataset Info:")
    print(f"  Total samples: {len(base_dataset)}")
    print(f"  K-Fold Cross Validation: {config.n_folds} folds")
    print(f"  ~{len(base_dataset) // config.n_folds} validation samples per fold")

    # Setup K-Fold Cross Validation
    print("\nCreating K-fold splits...")
    kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    fold_splits = list(kfold.split(range(len(base_dataset))))

    # Verify no data leakage
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        assert len(set(train_idx) & set(val_idx)) == 0, (
            f"Data leakage in fold {fold_idx}!"
        )
        print(f"Fold {fold_idx + 1}: Train={len(train_idx)}, Val={len(val_idx)}")

    # Train each fold
    all_folds_results = []
    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        fold_results = train_fold(
            fold_idx, train_indices, val_indices, base_dataset, config, device, weights
        )
        all_folds_results.append(fold_results)

    # Print cross-validation summary
    print(f"\n{'=' * 80}")
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'=' * 80}")

    avg_best_r2 = sum([r["best_val_r2"] for r in all_folds_results]) / config.n_folds
    std_best_r2 = np.std([r["best_val_r2"] for r in all_folds_results])
    avg_final_val_r2 = (
        sum([r["final_val_r2"] for r in all_folds_results]) / config.n_folds
    )

    print("\nOverall Performance:")
    print(f"  Average Best Val R2: {avg_best_r2:.4f} ± {std_best_r2:.4f}")
    print(f"  Average Final Val R2: {avg_final_val_r2:.4f}")

    print("\nPer-Fold Results:")
    for result in all_folds_results:
        print(
            f"  Fold {result['fold']}: Best R2 = {result['best_val_r2']:.4f} "
            f"(epoch {result['best_epoch']}) | Time: {result['fold_time'] / 3600:.2f}h"
        )

    print("\nPer-Target Average R2 (at best epochs across all folds):")
    for target in ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]:
        avg_r2 = (
            sum([r["best_val_r2_per_target"][target] for r in all_folds_results])
            / config.n_folds
        )
        std_r2 = np.std(
            [r["best_val_r2_per_target"][target] for r in all_folds_results]
        )
        print(f"  {target}: {avg_r2:.4f} ± {std_r2:.4f}")

    print(f"\n{'=' * 80}")
    print(f"All {config.n_folds} folds completed!")
    print(f"Models saved in {config.output_dir}/ directory")
    print(f"Best model for each fold is at fold*/best.pth")
    print(f"{'=' * 80}")

    return all_folds_results
