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
    y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute weighted R2 score across all targets.

    Args:
        y_true: Ground truth targets (batch_size, 3) -> [Green, Dead, Clover]
        y_pred: Predicted targets (batch_size, 3)
        weights: Weights for each component.
                 Default corresponds to [Green, Dead, Clover, GDM, Total] with values [0.1, 0.1, 0.1, 0.2, 0.5].

    Returns:
        Weighted mean R2 score.
    """
    if weights is None:
        # Default weights from the notebook
        weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], device=y_true.device)

    y_true = target_untransform(y_true)
    y_pred = target_untransform(y_pred)

    # compute weighted R2
    mean = y_true.mean(dim=0)
    SSE = ((y_true - y_pred) ** 2).sum(dim=0)
    TSS = ((y_true - mean) ** 2).sum(dim=0)
    TSS = torch.clamp(TSS, min=1e-8)
    R2 = 1 - SSE / TSS
    R2 = torch.clamp(R2, min=-10, max=1)

    return (R2 * weights).sum() / weights.sum()


def weighted_r2_single(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Compute R2 for each individual target separately.
    Returns dict with R2 for each target:
    - Dry_Green_g (y[0])
    - Dry_Dead_g (y[1])
    - Dry_Clover_g (y[2])
    - GDM_g (y[0] + y[2])
    - Dry_Total_g (y[0] + y[1] + y[2])
    """
    y_true = target_untransform(y_true)
    y_pred = target_untransform(y_pred)

    # create new columns for GDM and Total
    gdm_true = (y_true[:, 0] + y_true[:, 2]).unsqueeze(1)  # (batch, 1)
    tot_true = (y_true[:, 0] + y_true[:, 1] + y_true[:, 2]).unsqueeze(1)

    gdm_pred = (y_pred[:, 0] + y_pred[:, 2]).unsqueeze(1)  # (batch, 1)
    tot_pred = (y_pred[:, 0] + y_pred[:, 1] + y_pred[:, 2]).unsqueeze(1)

    # append columns
    y_true_full = torch.cat([y_true, gdm_true, tot_true], dim=1)
    y_pred_full = torch.cat([y_pred, gdm_pred, tot_pred], dim=1)

    # compute R2 for each target separately
    mean = y_true_full.mean(dim=0)  # (5,)
    SSE = ((y_true_full - y_pred_full) ** 2).sum(dim=0)  # (5,)
    TSS = ((y_true_full - mean) ** 2).sum(dim=0)  # (5,)
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
