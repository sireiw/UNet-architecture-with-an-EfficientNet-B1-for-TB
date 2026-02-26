import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast
from typing import Tuple, List, Union
from .config import Config

def adaptive_bn_update(model: nn.Module, real_loader: DataLoader, device: torch.device, num_passes: int = 2) -> None:
    """Update BatchNorm statistics exclusively on real-world data target domain.
    
    Disables gradients and explicitly forces forward passes to allow Batch Normalization
    modules to recalibrate their running mean and variance against the new target distribution.
    
    Args:
        model (nn.Module): The classification neural network.
        real_loader (DataLoader): An iterable providing real-world target domain samples.
        device (torch.device): Compute hardware accelerator constraint.
        num_passes (int, optional): Number of sweeps over the distribution. Defaults to 2.
    """
    print(f"\n🔧 Running Adaptive BatchNorm update ({num_passes} passes)...")
    model.train()
    
    for p in model.parameters():
        p.requires_grad_(False)
    
    with torch.no_grad():
        for pass_idx in range(num_passes):
            for batch in tqdm(real_loader, desc=f'AdaBN Pass {pass_idx+1}/{num_passes}'):
                if len(batch) == 3:
                    xb, _, _ = batch
                else:
                    xb, _ = batch
                xb = xb.to(device, non_blocking=True)
                _ = model(xb)
    
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    
    print("  ✅ BatchNorm statistics updated on real-world data")


def tta_predict(model: nn.Module, dataloader: DataLoader, config: Config, n_tta: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Execute prediction using Test-Time Augmentation (TTA).
    
    Ensembles uncalibrated predictions across multiple geometric variations of the identical
    underlying input to yield a statistically stable output classification prediction.
    
    Args:
        model (nn.Module): Primary classification architecture.
        dataloader (DataLoader): Inference or testing iterable dataset.
        config (Config): Configuration holding hardware device pointers and hyperparameters.
        n_tta (int, optional): The number of concurrent augmentations to accumulate. Defaults to 4.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: A parallel structured tuple of predicted class outputs and actual labels.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='TTA Inference'):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(config.DEVICE)
            
            # ONLY geometric transforms (no photometric on normalized tensors)
            aug_imgs = [
                images,                                      # original
                torch.flip(images, dims=[-1]),              # horizontal flip
                torch.rot90(images, 1, dims=[-2, -1]),      # 90° rotate
                torch.rot90(images, 3, dims=[-2, -1])       # 270° rotate
            ]
            
            logits_agg = torch.zeros(images.size(0), config.NUM_CLASSES, device=config.DEVICE)
            
            for aug_img in aug_imgs[:n_tta]:
                with autocast(enabled=config.USE_AMP):
                    logits = model(aug_img)
                logits_agg += logits
            
            logits_agg /= n_tta
            preds = logits_agg.argmax(1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

