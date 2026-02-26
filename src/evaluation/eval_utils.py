import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def calculate_class_weights(train_dataset, config):
    """Calculate class weights"""
    label_counts = Counter(train_dataset.labels)
    total_samples = len(train_dataset)
    
    weights = []
    for class_name in config.CLASS_NAMES:
        count = label_counts.get(class_name, 1)
        weight = total_samples / (len(config.CLASS_NAMES) * count)
        weights.append(weight)
    
    weights = np.array(weights)
    weights = weights / weights.sum() * len(weights)
    weights = np.power(weights, 0.75)
    
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    
    print(f"\nCalculated class weights: {weight_tensor}")
    
    return weight_tensor.to(config.DEVICE)


def validate_per_class(model, val_loader, config, calibrator=None):
    """Validate with per-class metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(config.DEVICE)
            
            with autocast(enabled=config.USE_AMP):
                outputs = model(images)
            
            if calibrator is not None:
                outputs = calibrator(outputs)
            
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    per_class_acc = {}
    per_class_f1 = {}
    for i, class_name in enumerate(config.CLASS_NAMES):
        mask = np.array(all_labels) == i
        if mask.sum() > 0:
            class_acc = (np.array(all_preds)[mask] == i).mean()
            per_class_acc[class_name] = class_acc
            
            y_true_binary = (np.array(all_labels) == i).astype(int)
            y_pred_binary = (np.array(all_preds) == i).astype(int)
            class_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            per_class_f1[class_name] = class_f1
    
    overall_acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return overall_acc, macro_f1, per_class_acc, per_class_f1


def tune_class_thresholds(model, val_loader, config, target_metric='f1'):
    """Tune per-class probability thresholds for better precision-recall balance"""
    print("\n🎯 Tuning per-class thresholds...")
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(config.DEVICE)
            
            with autocast(enabled=config.USE_AMP):
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    
    # Find optimal threshold for each class
    thresholds = {}
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in np.arange(0.2, 0.8, 0.05):
            # Binary prediction for this class
            y_pred = (all_probs[:, class_idx] >= threshold).astype(int)
            y_true = (all_labels == class_idx).astype(int)
            
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        thresholds[class_name] = best_threshold
        print(f"  {class_name}: threshold={best_threshold:.2f} (F1={best_f1:.3f})")
    
    return thresholds


def predict_with_thresholds(probs, thresholds, class_names):
    """Apply per-class thresholds to make predictions"""
    n_samples = probs.shape[0]
    predictions = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Adjust probabilities by thresholds
        adjusted_probs = np.zeros(len(class_names))
        for j, class_name in enumerate(class_names):
            threshold = thresholds.get(class_name, 0.5)
            # Scale probability relative to threshold
            if probs[i, j] >= threshold:
                adjusted_probs[j] = probs[i, j]
            else:
                adjusted_probs[j] = probs[i, j] * (threshold / 0.5)
        
        predictions[i] = np.argmax(adjusted_probs)
    
    return predictions


def ensemble_predict(models, dataloader, config, weights=None):
    """Ensemble prediction from multiple models"""
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in tqdm(dataloader, desc='Ensemble Inference'):
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch
        
        images = images.to(config.DEVICE)
        
        # Average probabilities from all models
        ensemble_probs = torch.zeros(images.size(0), config.NUM_CLASSES, device=config.DEVICE)
        
        for model, weight in zip(models, weights):
            model.eval()
            with torch.no_grad():
                with autocast(enabled=config.USE_AMP):
                    outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                ensemble_probs += weight * probs
        
        preds = ensemble_probs.argmax(1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(ensemble_probs.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    return np.array(all_preds), np.array(all_labels), all_probs


def evaluate_model(model, test_loader, config, calibrator=None):
    """Evaluate model on test set"""
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    model.to(config.DEVICE)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(config.DEVICE)
            
            with autocast(enabled=config.USE_AMP):
                outputs = model(images)
            
            if calibrator is not None:
                outputs = calibrator(outputs)
            
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs)
    
    all_probs = np.vstack(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, 
                                   target_names=config.CLASS_NAMES, output_dict=True,
                                   zero_division=0) # ✅ Added zero_division
    cm = confusion_matrix(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) # ✅ Added zero_division
    
    return accuracy, report, cm, all_preds, all_labels, all_probs, macro_f1


