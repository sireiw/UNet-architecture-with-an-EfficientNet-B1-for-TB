import torch
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast

def adaptive_bn_update(model, real_loader, device, num_passes=2):
    """Update BatchNorm statistics on real-world data"""
    
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


def tta_predict(model, dataloader, config, n_tta=4):
    """Test-Time Augmentation - FIXED to use only geometric transforms"""
    model.eval()
    all_preds = []
    all_labels = []
    
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

