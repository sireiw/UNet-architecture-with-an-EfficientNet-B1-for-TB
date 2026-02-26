import torch
import torch.nn as nn
import torch.nn.functional as F

class EMA:
    """Exponential Moving Average of model weights"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class BiasTempCalibrator(nn.Module):
    """Per-class temperature scaling and bias"""
    
    def __init__(self, num_classes):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.log_temp = nn.Parameter(torch.zeros(num_classes))
    
    def forward(self, logits):
        T = torch.exp(self.log_temp).unsqueeze(0)
        return (logits + self.bias) / T
