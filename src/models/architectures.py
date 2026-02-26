import torch
import torch.nn as nn
from typing import Dict, Optional

try:
    import segmentation_models_pytorch as smp
    _HAS_SMP = True
except ImportError:
    _HAS_SMP = False

from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

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


class UNetSegmentationModel(nn.Module):
    """Deep learning model adapted from segmentation_models_pytorch for precise Lung localization."""
    
    def __init__(self, num_classes: int = 1, encoder_weights: str = 'imagenet'):
        super().__init__()
        if not _HAS_SMP:
            raise ImportError("segmentation_models_pytorch is required for UNetSegmentationModel")
            
        self.model = smp.Unet(
            encoder_name="efficientnet-b1",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DomainRobustClassifier(nn.Module):
    """
    Feature-extraction enhanced EfficientNet-B1 targeted for complex cross-domain representation alignment.
    
    Constructs a customized classification head with stronger regularization (Dropout) 
    designed to maintain consistent features spanning clinical boundaries.
    """
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout_rate: float = 0.4):
        super().__init__()
        if pretrained:
            self.backbone = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        else:
            self.backbone = efficientnet_b1(weights=None)
            
        # Extract features just before the final classification head
        in_features = self.backbone.classifier[1].in_features
        
        # Remove the standard classification head
        self.backbone.classifier = nn.Identity()
        
        # Build the robust classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        if not pretrained:
            self._initialize_weights()
            
    def _initialize_weights(self) -> None:
        """Initialize weights appropriately for random initialization."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass returning pre-classification pooled feature maps."""
        # features are the output of the pooling layer (1280 dimension for b1)
        features = self.backbone(x) 
        return features
        
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Standard classification forward pass optionally returning features for contrastive/MMD logic."""
        features = self.forward_features(x)
        logits = self.head(features)
        
        if return_features:
            return logits, features
        return logits
