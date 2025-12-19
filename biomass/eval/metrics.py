import torch


DTYPE = torch.float32

defined_weights = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}
target_weights = torch.tensor(list(defined_weights.values()), dtype=DTYPE)


def target_untransform(targets: torch.Tensor) -> torch.Tensor:
    """
    Inverse transform for the targets (log1p -> expm1).
    """
    return torch.expm1(targets)


def weighted_r2(
    y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor = None, enforce_rules: bool = True
) -> torch.Tensor:
    """
    Compute weighted R2 score across all targets.

    Args:
        y_true: Ground truth targets (batch_size, 5) -> [Green, Dead, Clover, GDM, Total]
        y_pred: Predicted targets (batch_size, 5) -> [Green, Dead, Clover, GDM, Total]
        weights: Weights for each component.
                 Default corresponds to [Green, Dead, Clover, GDM, Total] with values [0.1, 0.1, 0.1, 0.2, 0.5].
        enforce_rules: If True, recalculate GDM and Total from components for validation.
                      If False, use model's direct predictions for training.

    Returns:
        Weighted mean R2 score.
    """
    if weights is None:
        # Default weights from the notebook
        weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], device=y_true.device)

    # Apply inverse log transform
    y_true = target_untransform(y_true)
    y_pred = target_untransform(y_pred)

    # Model predicts all 5 targets:
    # [0] = Dry_Green_g, [1] = Dry_Dead_g, [2] = Dry_Clover_g
    # [3] = GDM_g (should be [0] + [2]), [4] = Dry_Total_g (should be [0] + [1] + [2])

    if enforce_rules:
        # For validation: enforce calculation rules
        # Recalculate GDM and Total from first 3 components
        gdm_pred = (y_pred[:, 0] + y_pred[:, 2]).unsqueeze(1)
        tot_pred = (y_pred[:, 0] + y_pred[:, 1] + y_pred[:, 2]).unsqueeze(1)
        y_pred = torch.cat([y_pred[:, :3], gdm_pred, tot_pred], dim=1)

    # compute weighted R2
    mean = y_true.mean(dim=0)
    SSE = ((y_true - y_pred) ** 2).sum(dim=0)
    TSS = ((y_true - mean) ** 2).sum(dim=0)
    TSS = torch.clamp(TSS, min=1e-8)
    R2 = 1 - SSE / TSS
    R2 = torch.clamp(R2, min=-10, max=1)

    return (R2 * weights).sum() / weights.sum()


def weighted_r2_single(y_true: torch.Tensor, y_pred: torch.Tensor, enforce_rules: bool = True):
    """
    Compute R2 for each individual target separately.
    Returns dict with R2 for each target:
    - Dry_Green_g (y[0])
    - Dry_Dead_g (y[1])
    - Dry_Clover_g (y[2])
    - GDM_g (y[3]) - model directly predicts this (or calculated if enforce_rules=True)
    - Dry_Total_g (y[4]) - model directly predicts this (or calculated if enforce_rules=True)
    
    Args:
        enforce_rules: If True, recalculate GDM and Total from components for validation.
                      If False, use model's direct predictions for training.
    """
    # Apply inverse log transform
    y_true = target_untransform(y_true)
    y_pred = target_untransform(y_pred)

    # Model predicts all 5 targets:
    # [0] = Dry_Green_g, [1] = Dry_Dead_g, [2] = Dry_Clover_g
    # [3] = GDM_g (should be [0] + [2]), [4] = Dry_Total_g (should be [0] + [1] + [2])

    if enforce_rules:
        # For validation: enforce calculation rules
        # Recalculate GDM and Total from first 3 components
        gdm_pred = (y_pred[:, 0] + y_pred[:, 2]).unsqueeze(1)
        tot_pred = (y_pred[:, 0] + y_pred[:, 1] + y_pred[:, 2]).unsqueeze(1)
        y_pred = torch.cat([y_pred[:, :3], gdm_pred, tot_pred], dim=1)

    # compute R2 for each target separately
    mean = y_true.mean(dim=0)  # (5,)
    SSE = ((y_true - y_pred) ** 2).sum(dim=0)  # (5,)
    TSS = ((y_true - mean) ** 2).sum(dim=0)  # (5,)
    TSS = torch.clamp(TSS, min=1e-8)
    R2 = 1 - SSE / TSS  # (5,)
    R2 = torch.clamp(R2, min=-10, max=1)

    target_labels = [
        "Dry_Green_g",
        "Dry_Dead_g",
        "Dry_Clover_g",
        "GDM_g",
        "Dry_Total_g",
    ]

    return {label: r2_val.item() for label, r2_val in zip(target_labels, R2)}
