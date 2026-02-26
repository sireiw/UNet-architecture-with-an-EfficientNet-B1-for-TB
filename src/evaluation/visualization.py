import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import torch
import torch.nn.functional as F
from collections import Counter
import cv2
from PIL import Image
from src.models.architectures import _HAS_SMP

def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages, handle division by zero
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_percent = cm.astype('float') / cm_sum * 100
    cm_percent = np.nan_to_num(cm_percent) # Set NaNs (from 0/0) to 0
    
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(history, save_path, title="Training History"):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Metrics
    if 'val_macro_f1' in history:
        axes[1].plot(history['val_macro_f1'], label='Val Macro F1', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Validation F1')
        axes[1].legend()
        axes[1].grid(True)
    elif 'real_f1' in history and 'syn_f1' in history:
        axes[1].plot(history['real_f1'], label='Real F1', marker='o')
        axes[1].plot(history['syn_f1'], label='Synthetic F1', marker='s')
        if 'tb_recall_real' in history:
            axes[1].plot(history['tb_recall_real'], label='TB Recall (Real)', marker='^', linestyle='--')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('F1 Scores (Real vs Synthetic)')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_detailed_report(accuracy, report, cm, config, save_path, macro_f1=None, title_suffix=""):
    """Save comprehensive report"""
    
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"TB RESCUE PIPELINE EVALUATION REPORT {title_suffix}\n")
        f.write("All 10 corrections applied\n")
        f.write("="*70 + "\n\n")
        
        f.write("Pipeline:\n")
        f.write("  Phase 1: Lung Segmentation (U-Net + EfficientNet-B1)\n")
        f.write("  Phase 2: Disease Classification (EfficientNet-B1 + EMA)\n")
        f.write("  Phase 3: TB Rescue KD Fine-Tuning (K-Fold CV)\n\n")
        
        f.write("Key TB Rescue Fixes Applied:\n")
        f.write(f"  ✅ Slower Curriculum: {config.FINETUNE_EPOCHS} epochs, {config.INITIAL_SYN_RATIO*100}% -> {config.FINAL_SYN_RATIO*100}% over {config.RATIO_TRANSITION_EPOCH} epochs\n")
        f.write(f"  ✅ Gentler LRs: Backbone={config.FINETUNE_LR_BACKBONE}, Classifier={config.FINETUNE_LR_CLASSIFIER}\n")
        f.write(f"  ✅ TB-Aware Metric: 50% F1 + 50% TB Recall\n")
        f.write(f"  ✅ TB Guardrails: Stop if TB Recall < {config.MIN_TB_RECALL} or 0.0\n")
        f.write(f"  ✅ Stronger CORAL: λ={config.CORAL_LAMBDA} with {config.CORAL_WARMUP_EPOCHS} epoch warmup\n")
        f.write(f"  ✅ Mixup Re-enabled: p={config.MIXUP_PROB_FINETUNE} for real samples\n")
        f.write(f"  ✅ Standard Focal Loss: Replaced CB-Focal for real samples\n")
        f.write(f"  ✅ More Grad Accum: {config.ACCUM_STEPS} steps\n")
        f.write(f"  ✅ Preadaptation: {config.PREADAPT_EPOCHS} epochs with {config.PREADAPT_REAL_RATIO*100}% real data\n")
        f.write(f"  ✅ K-Fold CV: {config.FINETUNE_KFOLD_SPLITS} splits for robust validation\n\n")

        
        f.write(f"Test Set Accuracy: {accuracy:.4f}\n")
        if macro_f1:
            f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
        f.write("\n")
        
        f.write("Per-Class Metrics:\n")
        f.write("-"*70 + "\n")
        df_report = pd.DataFrame(report).transpose()
        f.write(df_report.round(4).to_string())
        f.write("\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("-"*70 + "\n")
        cm_df = pd.DataFrame(cm, index=config.CLASS_NAMES, columns=config.CLASS_NAMES)
        f.write(cm_df.to_string())




