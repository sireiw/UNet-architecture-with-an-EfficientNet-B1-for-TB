import os
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from src.models.losses import kd_loss, coral_loss, mmd_loss, BCEDiceLoss, LabelSmoothingLoss, BalancedFocalLoss, SupConLoss
from src.models.architectures import EMA
from src.utils import quality_stats

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
            
            # ✅ FIXED: Only geometric transforms (no photometric on normalized tensors)
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


def train_segmentation_model(model, train_loader, val_loader, config):
    """Train segmentation model"""
    print("\n" + "="*70)
    print("PHASE 1: TRAINING SEGMENTATION MODEL")
    print("="*70)
    
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    criterion = BCEDiceLoss()
    scaler = GradScaler(enabled=config.USE_AMP)
    early_stopping = EarlyStopping(patience=config.EARLY_STOP_PATIENCE, mode='min')
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}
    
    for epoch in range(config.SEG_EPOCHS):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.SEG_EPOCHS}'):
            images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            with autocast(enabled=config.USE_AMP):
                logits = model(images)
                loss = criterion(logits, masks)
            
            scaler.scale(loss).backward()
            
            if config.GRADIENT_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                dice = (2 * (preds * masks).sum() / (preds.sum() + masks.sum() + 1e-7))
                train_dice += dice.item()
        
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
                
                with autocast(enabled=config.USE_AMP):
                    logits = model(images)
                    loss = criterion(logits, masks)
                
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                dice = (2 * (preds * masks).sum() / (preds.sum() + masks.sum() + 1e-7))
                val_dice += dice.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_dice /= len(train_loader)
        val_dice /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
              f'Train Dice={train_dice:.4f}, Val Dice={val_dice:.4f}')
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, 'best_segmentation_model.pth'))
            print('  ✓ Best model saved!')
        
        if early_stopping.step(val_loss):
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return history

def apply_segmentation_to_all_images(segmentation_model, cls_data, config):
    """Apply segmentation to all classification images"""
    print("\n" + "="*70)
    print("APPLYING SEGMENTATION TO ALL IMAGES")
    print("="*70)
    
    segmentation_model.eval()
    
    seg_transform = A.Compose([
        A.Resize(config.SEG_IMAGE_SIZE, config.SEG_IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    segmented_paths = {
        'train': [],
        'val': [],
        'test': []
    }
    
    failed_images = []
    
    with torch.no_grad():
        for split in ['train', 'val', 'test']:
            split_folder = 'train' if split == 'train' else ('validation' if split == 'val' else 'test')
            output_split_dir = os.path.join(config.SEGMENTED_IMAGES_DIR, split_folder)
            
            for idx, img_path in enumerate(tqdm(cls_data[split]['images'], 
                                              desc=f'Segmenting {split}')):
                try:
                    image = read_image_safe(img_path, grayscale=False)
                    if image is None:
                        failed_images.append(img_path)
                        segmented_paths[split].append(img_path)
                        continue
                    
                    if len(image.shape) != 3:
                        failed_images.append(img_path)
                        segmented_paths[split].append(img_path)
                        continue
                    
                    original_shape = image.shape[:2]
                    
                    augmented = seg_transform(image=image)
                    img_tensor = augmented['image'].unsqueeze(0).to(config.DEVICE)
                    
                    with autocast(enabled=config.USE_AMP):
                        mask_pred = segmentation_model(img_tensor)
                    
                    mask_pred = torch.sigmoid(mask_pred).cpu().numpy()[0, 0]
                    mask_pred = np.clip(mask_pred, 0.0, 1.0)
                    
                    mask_resized = cv2.resize(
                        mask_pred.astype(np.float32), 
                        (original_shape[1], original_shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                    
                    mask_binary = (mask_resized > config.MASK_THRESHOLD).astype(np.uint8) * 255
                    
                    if config.MORPH_CLEANUP:
                        kernel = np.ones((config.MORPH_KERNEL, config.MORPH_KERNEL), np.uint8)
                        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
                        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
                    
                    mask_3channel = np.stack([mask_binary, mask_binary, mask_binary], axis=2)
                    segmented_image = np.where(mask_3channel > 0, image, 0)
                    
                    label = cls_data[split]['labels'][idx]
                    class_dir = os.path.join(output_split_dir, label)
                    os.makedirs(class_dir, exist_ok=True)
                    
                    filename = os.path.basename(img_path)
                    output_path = os.path.join(class_dir, f"seg_{filename}")
                    
                    pil_image = Image.fromarray(segmented_image.astype(np.uint8))
                    pil_image.save(output_path)
                    
                    segmented_paths[split].append(output_path)
                    
                except Exception:
                    failed_images.append(img_path)
                    segmented_paths[split].append(img_path)
                    continue
    
    print(f"\nSegmentation complete!")
    print(f"Successfully processed: {sum(len(paths) for paths in segmented_paths.values()) - len(failed_images)}")
    if failed_images:
        print(f"Failed: {len(failed_images)} images")
    
    return segmented_paths


def train_classification_model(train_loader, val_loader, train_dataset, config):
    """Train classification model with EMA and improved settings"""
    
    print("\n" + "="*70)
    print("PHASE 2: TRAINING CLASSIFICATION MODEL")
    print("="*70)
    
    model = DomainRobustClassifier(
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    )
    model = model.to(config.DEVICE)
    
    # ✅ FIXED: Use CosineAnnealingWarmRestarts instead of ReduceLROnPlateau
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1
    )
    
    class_weights = calculate_class_weights(train_dataset, config)
    
    if config.USE_FOCAL_LOSS:
        criterion = BalancedFocalLoss(alpha=class_weights, gamma=config.FOCAL_LOSS_GAMMA)
    else:
        criterion = LabelSmoothingLoss(config.NUM_CLASSES, smoothing=config.LABEL_SMOOTHING)
    
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOP_PATIENCE,
        mode='max',
        min_delta=config.MIN_DELTA
    )
    
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # ✅ NEW: EMA
    ema = EMA(model, decay=config.EMA_DECAY) if config.USE_EMA else None
    
    best_val_f1 = 0.0
    history = {'train_loss': [], 'val_acc': [], 'val_macro_f1': []}
    
    for epoch in range(config.CLS_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.CLS_EPOCHS}'):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            # ✅ Apply Mixup
            if np.random.rand() < config.MIXUP_PROB:
                lam = np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA)
                idx = torch.randperm(images.size(0))
                
                mixed_images = lam * images + (1 - lam) * images[idx]
                
                # Soft labels
                labels_one_hot = F.one_hot(labels, config.NUM_CLASSES).float()
                mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot[idx]
                
                optimizer.zero_grad()
                with autocast(enabled=config.USE_AMP):
                    outputs = model(mixed_images)
                    # Use standard CE for soft labels
                    loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * mixed_labels, dim=1))
            
            else:
                optimizer.zero_grad()
                with autocast(enabled=config.USE_AMP):
                    outputs = model(images)
                    loss = criterion(outputs, labels) # Use Focal/LS for non-mixed
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            
            scaler.step(optimizer)
            scaler.update()
            
            # ✅ Update EMA
            if ema is not None:
                ema.update()
            
            train_loss += loss.item()
        
        # ✅ Validate with EMA weights
        if ema is not None:
            ema.apply_shadow()
        
        overall_acc, macro_f1, per_class_acc, per_class_f1 = validate_per_class(model, val_loader, config)
        
        if ema is not None:
            ema.restore()
        
        avg_loss = train_loss / len(train_loader)
        
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(overall_acc)
        history['val_macro_f1'].append(macro_f1)
        
        print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={overall_acc:.4f}, Macro F1={macro_f1:.4f}")
        
        scheduler.step()
        
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            # Save EMA weights if available
            if ema is not None:
                ema.apply_shadow()
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, 'best_classifier_model.pth'))
            if ema is not None:
                ema.restore()
            print('  ✓ Best model saved!')
        
        if early_stopping.step(macro_f1):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(config.OUTPUT_DIR, 'best_classifier_model.pth')))
    
    return model, history


def _run_finetune_phase(mixed_dataset, student, teacher, optimizer, scheduler, scaler,  # ✅ ADD mixed_dataset parameter
                        base_syn_indices, base_real_indices, 
                        val_loader_real, val_loader_syn, 
                        baseline_syn_f1, config, 
                        start_epoch, num_epochs, phase_name="Training"):

    """Internal loop for fine-tuning (used for pre-adaptation and main phase)"""

    best_combined_metric = 0.0 # ✅ TB RESCUE: Use combined metric
    best_model_state = None
    patience_counter = 0
    patience = 10  # ✅ Increased patience
    
    history = {
        'train_loss': [], 'real_f1': [], 'syn_f1': [], 
        'syn_ratio': [], 'tb_recall_real': [],
        'coral_loss': [], 'mmd_loss': [], 'contrastive_loss': []  # ✅ Track domain losses
    }
    
    # Define Focal Loss criterion for real samples (Mixup-compatible)
    focal_criterion_real = BalancedFocalLoss(
        alpha=None,  # No class weighting
        gamma=2.0,   # Standard gamma
        reduction='none' # Per-sample loss
    )
    
    # ✅ NEW: Contrastive loss
    contrastive_criterion = SupConLoss(temperature=config.CONTRASTIVE_TEMP) if config.USE_CONTRASTIVE else None
    
    for epoch_idx in range(num_epochs):
        epoch = start_epoch + epoch_idx # Absolute epoch number
        
        # ✅ Slower, more conservative curriculum
        if epoch < config.RATIO_TRANSITION_EPOCH:
            progress = epoch / config.RATIO_TRANSITION_EPOCH
            syn_ratio = config.INITIAL_SYN_RATIO - \
                       (config.INITIAL_SYN_RATIO - config.FINAL_SYN_RATIO) * progress
        else:
            syn_ratio = config.FINAL_SYN_RATIO
        
        # Override for pre-adaptation phase
        if phase_name == "Preadapt":
            syn_ratio = 1.0 - config.PREADAPT_REAL_RATIO
        
        print(f"\n📊 {phase_name} Epoch {epoch_idx+1}/{num_epochs} (Abs. {epoch+1})")
        print(f"  Synthetic ratio: {syn_ratio:.1%} → Real ratio: {1-syn_ratio:.1%}")
        
        # ✅ Rebuild sampler with current ratio
        sampler = TwoStreamBatchSampler(
            base_syn_indices, base_real_indices,
            batch_size=config.FINETUNE_BATCH_SIZE,
            ratio_syn=syn_ratio,
            drop_last=True # Drop last to stabilize batch sizes
        )
        train_loader = DataLoader(
            mixed_dataset, 
            batch_sampler=sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=True  # Keep workers alive between epochs
        )
        
        student.train()
        train_loss = 0.0
        loss_real_total = 0.0
        loss_syn_total = 0.0
        loss_coral_total = 0.0
        loss_mmd_total = 0.0
        loss_contrastive_total = 0.0
        num_batches = 0
        num_real_batches = 0
        num_syn_batches = 0
        
        # ✅ Calculate absolute epoch for proper CORAL warmup
        absolute_epoch = start_epoch + epoch_idx
        
        optimizer.zero_grad()
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f'{phase_name} Epoch {epoch_idx+1}')):
            if batch_data is None: continue
            images, labels, is_synthetic = batch_data
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            is_synthetic = is_synthetic.to(config.DEVICE)
            
            with autocast(enabled=config.USE_AMP):
                logits_student, feats = student(images, return_features=True)
            
            mask_real = ~is_synthetic
            mask_syn = is_synthetic
            
            loss = 0.0
            
            # ✅ REAL samples: Standard Focal Loss + Mixup
            if mask_real.any():
                real_logits = logits_student[mask_real]
                real_labels = labels[mask_real]
                
                loss_real = 0.0
                # Apply Mixup (on loss)
                if np.random.rand() < config.MIXUP_PROB_FINETUNE and real_labels.size(0) > 1:
                    lam = np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA)
                    idx = torch.randperm(real_labels.size(0))
                    
                    # loss = lam * loss(output, label_a) + (1-lam) * loss(output, label_b)
                    loss_real_a = focal_criterion_real(real_logits, real_labels)
                    loss_real_b = focal_criterion_real(real_logits, real_labels[idx])
                    loss_real = (lam * loss_real_a + (1 - lam) * loss_real_b).mean()
                else:
                    # Standard Focal Loss
                    loss_real = focal_criterion_real(real_logits, real_labels).mean()

                loss += config.KD_REAL_LOSS_WEIGHT * loss_real
                loss_real_total += loss_real.item()
                num_real_batches += 1
            
            # SYNTHETIC samples: Light KD + CE
            if mask_syn.any():
                with torch.no_grad():
                    with autocast(enabled=config.USE_AMP):
                        logits_teacher = teacher(images[mask_syn])
                
                ce_syn = F.cross_entropy(logits_student[mask_syn], labels[mask_syn])
                
                kd_syn = kd_loss(
                    logits_student[mask_syn],
                    logits_teacher,
                    labels=None,
                    T=config.KD_TEMPERATURE,
                    alpha=1.0
                )
                
                loss_syn = ce_syn + config.KD_ALPHA_SYN * kd_syn
                loss += loss_syn
                loss_syn_total += loss_syn.item()
                num_syn_batches += 1
            
            # ✅ DOMAIN ALIGNMENT LOSSES (CORAL + MMD + Contrastive)
            if mask_real.any() and mask_syn.any():
                feats_real = feats[mask_real]
                feats_syn = feats[mask_syn]
                
                if feats_real.size(0) > 1 and feats_syn.size(0) > 1:
                    # ✅ CORAL loss with FIXED warmup (uses absolute epoch)
                    if config.USE_CORAL_LOSS:
                        coral_weight = get_coral_weight(
                            epoch_idx, num_epochs, config.CORAL_WARMUP_EPOCHS, config,
                            absolute_epoch=absolute_epoch  # ✅ FIX: Use absolute epoch
                        )
                        coral = coral_loss(feats_syn, feats_real)
                        loss += coral_weight * coral
                        loss_coral_total += (coral_weight * coral).item()
                    
                    # ✅ NEW: MMD loss for additional domain alignment
                    if config.USE_MMD_LOSS:
                        mmd = mmd_loss(feats_syn, feats_real)
                        loss += config.MMD_LAMBDA * mmd
                        loss_mmd_total += (config.MMD_LAMBDA * mmd).item()
            
            # ✅ NEW: Supervised contrastive loss on real samples
            if config.USE_CONTRASTIVE and contrastive_criterion is not None and mask_real.any():
                feats_real = feats[mask_real]
                labels_real = labels[mask_real]
                if feats_real.size(0) > 2:  # Need at least 3 samples
                    con_loss = contrastive_criterion(feats_real, labels_real)
                    loss += config.CONTRASTIVE_LAMBDA * con_loss
                    loss_contrastive_total += (config.CONTRASTIVE_LAMBDA * con_loss).item()
            
            # ✅ Gradient accumulation (4 steps)
            loss = loss / config.ACCUM_STEPS
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, student.parameters()), 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * config.ACCUM_STEPS
            num_batches += 1
        
        # Final step for leftover gradients
        if num_batches % config.ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, student.parameters()), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        if phase_name != "Preadapt":
            scheduler.step()
        
        # Evaluate
        _, real_f1, _, per_class_f1_real = validate_per_class(student, val_loader_real, config)
        _, syn_f1, _, _ = validate_per_class(student, val_loader_syn, config)
        
        # ✅ TB RESCUE: Extract TB recall
        tb_recall = per_class_f1_real.get('tuberculosis', 0.0) if per_class_f1_real else 0.0
        
        avg_loss = train_loss / num_batches if num_batches > 0 else 0
        avg_loss_real = loss_real_total / num_real_batches if num_real_batches > 0 else 0
        avg_loss_syn = loss_syn_total / num_syn_batches if num_syn_batches > 0 else 0
        avg_loss_coral = loss_coral_total / num_batches if config.USE_CORAL_LOSS and num_batches > 0 else 0
        avg_loss_mmd = loss_mmd_total / num_batches if config.USE_MMD_LOSS and num_batches > 0 else 0
        avg_loss_con = loss_contrastive_total / num_batches if config.USE_CONTRASTIVE and num_batches > 0 else 0
        
        history['train_loss'].append(avg_loss)
        history['real_f1'].append(real_f1)
        history['syn_f1'].append(syn_f1)
        history['syn_ratio'].append(syn_ratio)
        history['tb_recall_real'].append(tb_recall)
        history['coral_loss'].append(avg_loss_coral)
        history['mmd_loss'].append(avg_loss_mmd)
        history['contrastive_loss'].append(avg_loss_con)
        
        print(f"  Loss: {avg_loss:.4f} (Real: {avg_loss_real:.4f}, Syn: {avg_loss_syn:.4f})")
        print(f"  Domain: CORAL={avg_loss_coral:.4f}, MMD={avg_loss_mmd:.4f}, Con={avg_loss_con:.4f}")
        print(f"  Real F1: {real_f1:.4f}")
        print(f"  Syn F1: {syn_f1:.4f} (baseline: {baseline_syn_f1:.4f}, drop: {baseline_syn_f1-syn_f1:.4f})")
        print(f"  🎯 TB Recall (real): {tb_recall:.4f}")
        
        # ✅ TB RESCUE: Modified Collapse Guardrail
        real_preds_unique = len(np.unique([p for p in range(config.NUM_CLASSES) 
                                          if per_class_f1_real.get(config.CLASS_NAMES[p], 0) > 0]))
        
        # Check after epoch 5
        abs_epoch_check = epoch > 5 
        
        if (real_preds_unique <= 1 or tb_recall == 0.0) and abs_epoch_check:
            print("\n⚠️ COLLAPSE DETECTED!")
            if tb_recall == 0.0:
                print("  Reason: TB recall is zero!")
            if real_preds_unique <= 1:
                print(f"  Reason: Predicting only {real_preds_unique} class(es)!")
            print("  Reverting to best checkpoint and stopping...")
            if best_model_state is not None:
                student.load_state_dict(best_model_state)
            return student, history, best_model_state, True # Return 'collapsed' flag
        
        # ✅ TB RESCUE: TB-specific recall guardrail
        if tb_recall < config.MIN_TB_RECALL and abs_epoch_check:
            print(f"\n⚠️ TB RECALL TOO LOW: {tb_recall:.4f} < {config.MIN_TB_RECALL}")
            print("  Reverting to previous checkpoint and stopping...")
            if best_model_state is not None:
                student.load_state_dict(best_model_state)
            return student, history, best_model_state, True # Return 'collapsed' flag
        
        # ✅ TB RESCUE: Save best model based on combined (F1 + TB Recall)
        combined_metric = 0.5 * real_f1 + 0.5 * tb_recall
        
        if combined_metric > best_combined_metric:
            best_combined_metric = combined_metric
            best_model_state = deepcopy(student.state_dict())
            patience_counter = 0
            print(f'  ✅ Best model saved! (Combined Metric: {combined_metric:.4f})')
        else:
            patience_counter += 1
        
        # Early stopping
        syn_drop = baseline_syn_f1 - syn_f1
        if syn_drop > config.MAX_SYNTHETIC_F1_DROP:
            print(f"  ⚠️ Large synthetic drop ({syn_drop:.1%}), but continuing...")
        
        if patience_counter >= patience and phase_name != "Preadapt":
            print(f"\nℹ️ Early stopping (no improvement for {patience} epochs)")
            break
            
    # Load best model at the end of the phase
    if best_model_state is not None:
        student.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model for phase (Combined Metric: {best_combined_metric:.4f})")
    
    return student, history, best_model_state, False # Not collapsed

def improved_finetune_with_kd(student, teacher, mixed_dataset, base_syn_indices, base_real_indices,
                               val_loader_real, val_loader_syn, baseline_syn_f1, config):
    """
    ✅ TB RESCUE Fine-tuning with ALL 10 corrections:
    """
    
    print("\n" + "="*70)
    print("PHASE 3: TB RESCUE FINE-TUNING (ALL CORRECTIONS)")
    print("="*70)
    
    print("\n🔧 Keeping BatchNorm for stability...")
    
    # Freeze teacher
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Unfreeze strategy
    total_params = list(student.backbone.features.parameters())
    unfreeze_from = int(len(total_params) * config.FREEZE_BACKBONE_RATIO)
    
    for idx, param in enumerate(total_params):
        param.requires_grad = (idx >= unfreeze_from)
    
    for param in student.backbone.classifier.parameters():
        param.requires_grad = True
    
    backbone_params = []
    classifier_params = []
    
    for name, param in student.named_parameters():
        if param.requires_grad:
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
    
    print(f"  Trainable: {len(backbone_params)} backbone + {len(classifier_params)} classifier params")
    
    # ✅ Gentler LRs
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.FINETUNE_LR_BACKBONE},
        {'params': classifier_params, 'lr': config.FINETUNE_LR_CLASSIFIER}
    ], weight_decay=config.FINETUNE_WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2 # Adjusted T_0
    )
    
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # ✅ PREADAPTATION PHASE
    if config.PREADAPT_EPOCHS > 0:
        print(f"\n🔧 Pre-adaptation phase ({config.PREADAPT_EPOCHS} epochs)...")
        student, pre_history, best_state, collapsed = _run_finetune_phase(
            mixed_dataset=mixed_dataset,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            base_syn_indices=base_syn_indices,
            base_real_indices=base_real_indices,
            val_loader_real=val_loader_real,
            val_loader_syn=val_loader_syn,
            baseline_syn_f1=baseline_syn_f1,
            config=config,
            start_epoch=0, # Starts from 0
            num_epochs=config.PREADAPT_EPOCHS,
            phase_name="Preadapt"
        )
        if collapsed:
            print("⚠️ Collapse detected during pre-adaptation. Stopping fine-tuning.")
            return student, pre_history
            
    # ✅ MAIN FINE-TUNING PHASE
    print(f"\n🔧 Main fine-tuning phase ({config.FINETUNE_EPOCHS} epochs)...")
    student, main_history, best_state, collapsed = _run_finetune_phase(
        mixed_dataset=mixed_dataset,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        base_syn_indices=base_syn_indices,
        base_real_indices=base_real_indices,
        val_loader_real=val_loader_real,
        val_loader_syn=val_loader_syn,
        baseline_syn_f1=baseline_syn_f1,
        config=config,
        start_epoch=config.PREADAPT_EPOCHS, # Starts after pre-adaptation
        num_epochs=config.FINETUNE_EPOCHS,
        phase_name="Finetune"
    )
    
    # Combine histories (if pre-adaptation was run)
    if config.PREADAPT_EPOCHS > 0:
        for key in main_history:
            main_history[key] = pre_history[key] + main_history[key]
            
    return student, main_history



