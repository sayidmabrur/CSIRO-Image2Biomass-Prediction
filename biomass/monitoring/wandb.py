import wandb
import os
from typing import Optional, Dict, Any

# Credentials extracted from csiro-competitionv2.ipynb logs
WANDB_ENTITY = "sayid-10121012-universitas-komputer-indonesia"
WANDB_PROJECT = "IMAGE2BIOMASSPREDICTION"


def setup_wandb(
    config: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
    fold_idx: Optional[int] = None,
    model_name: Optional[str] = None,
    resume: str = "allow",
    notes: Optional[str] = None,
    tags: Optional[list] = None,
) -> Any:
    """
    Initialize WandB with the extracted project settings.

    Args:
        config: Dictionary of hyperparameters to log
        run_name: Specific name for the run. If None, generated from fold and model.
        fold_idx: Fold number (used for naming if run_name is None)
        model_name: Model architecture name (used for naming if run_name is None)
        resume: WandB resume mode ("allow", "must", "never", "auto")
        notes: Notes for the run
        tags: List of tags for the run

    Returns:
        wandb run object
    """

    # Construct run name if not provided
    if run_name is None and fold_idx is not None and model_name is not None:
        run_name = f"fold{fold_idx}_{model_name}"

    # Ensure API key is available
    # User mentioned "credentials etc, take from notebook".
    # Since specific keys are usually in Kaggle secrets which are not visible in code export,
    # we assume the environment might have it or the user will allow interactive login.
    # If a key was explicitly found in the notebook code, it would be set here.

    try:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config=config,
            resume=resume,
            notes=notes,
            tags=tags,
            reinit=True,
        )
        return run
    except Exception as e:
        print(f"Failed to initialize WandB: {e}")
        print(
            "Please ensure you are logged in using `wandb login` or set WANDB_API_KEY environment variable."
        )
        return None


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """
    Safely log metrics to WandB.
    """
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_wandb():
    """
    Finish the current WandB run.
    """
    if wandb.run is not None:
        wandb.finish()
