import torch
import torch.nn as nn
from typing import Dict

class EMA:
    """Exponential Moving Average of model weights to stabilize training.
    
    Maintains a shadow copy of all model parameters and updates them via an
    exponential moving average scheme. During inference, these averaged weights
    can be applied to potentially improve generalization.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """Initialize the EMA wrapper.
        
        Args:
            model (nn.Module): The standard PyTorch neural network module.
            decay (float, optional): The exponential decay factor. Defaults to 0.999.
        """
        self.model: nn.Module = model
        self.decay: float = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self.register()
    
    def register(self) -> None:
        """Register the initial shadow parameters from the base model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self) -> None:
        """Update the shadow parameters using the moving average formula."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self) -> None:
        """Swap the models actual weights with the smoothed shadow weights for inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self) -> None:
        """Restore the models actual backprop weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class BiasTempCalibrator(nn.Module):
    """Per-class temperature scaling and bias calibration layer.
    
    Learns specific temperature scaling factors and biases for each output logit
    to improve probability calibration.
    """
    
    def __init__(self, num_classes: int):
        """Initialize the calibrator.
        
        Args:
            num_classes (int): The total number of output classes.
        """
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.log_temp = nn.Parameter(torch.zeros(num_classes))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply scaling and bias to uncalibrated network logits.
        
        Args:
            logits (torch.Tensor): Unscaled network outputs.
            
        Returns:
            torch.Tensor: Calibrated scaled outputs.
        """
        T = torch.exp(self.log_temp).unsqueeze(0)
        return (logits + self.bias) / T
