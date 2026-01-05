import torch
import torch.nn as nn
from typing import List, Optional


class LoRALayer(nn.Module):
    """
    A single LoRA layer that produces low-rank adjustments.
    
    LoRA decomposes weight updates into two low-rank matrices:
    ΔW = B @ A, where A ∈ R^(in×r) and B ∈ R^(r×out)
    
    This reduces trainable parameters from (in × out) to (in × r + r × out)
    """

    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r

        # Low-rank matrices with proper initialization
        # A uses Kaiming/He initialization, B starts at zero for stable training
        self.A = nn.Parameter(torch.zeros(in_features, r))
        self.B = nn.Parameter(torch.zeros(r, out_features))
        
        # Initialize A with small random values
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        # B starts at zero so LoRA has no effect at initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LoRA adjustment: x @ A @ B * scaling
        return (x @ self.A @ self.B) * self.scaling


class LoRALinear(nn.Module):
    """
    A linear layer with LoRA adaptation built-in.
    Wraps an existing nn.Linear and adds low-rank adaptation.
    
    Output = Original(x) + LoRA(x)
           = W @ x + b + (A @ B) @ x * scaling
    """

    def __init__(self, original_linear: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original_linear
        self.lora = LoRALayer(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            r=r,
            alpha=alpha,
        )
        
        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output + LoRA adjustment
        return self.original(x) + self.lora(x)


def inject_lora_into_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    r: int = 8,
    alpha: float = 16.0,
) -> int:
    """
    Inject LoRA into specific linear layers of a model.
    
    Args:
        model: The model to modify
        target_modules: List of module name patterns to target
                       For ConvNeXt: ["pwconv1", "pwconv2"] (pointwise convolutions implemented as Linear)
                       For ViT: ["query", "value", "dense"]
        r: LoRA rank (lower = fewer params, higher = more capacity)
        alpha: LoRA scaling factor (typically alpha = 2*r)
    
    Returns:
        Number of LoRA layers injected
    """
    if target_modules is None:
        # Default targets for ConvNeXt architecture
        target_modules = ["pointwise_conv1", "pointwise_conv2"]
    
    injected_count = 0
    modules_dict = dict(model.named_modules())
    
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if any(target in name for target in target_modules):
                # Get parent module and attribute name
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = modules_dict[parent_name]
                else:
                    parent = model
                    attr_name = name
                
                # Replace with LoRA-enhanced linear
                lora_linear = LoRALinear(module, r=r, alpha=alpha)
                setattr(parent, attr_name, lora_linear)
                injected_count += 1
    
    return injected_count


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from a model for optimizer.
    
    Args:
        model: Model with injected LoRA layers
        
    Returns:
        List of LoRA parameters (A and B matrices)
    """
    lora_params = []
    for name, param in model.named_parameters():
        if ".lora." in name:
            lora_params.append(param)
    return lora_params


def print_lora_info(model: nn.Module) -> None:
    """
    Print information about LoRA layers in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for name, p in model.named_parameters() if ".lora." in name)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"LoRA percentage: {100 * lora_params / total_params:.2f}%")