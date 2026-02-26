import torch
import os

class Config:
    """Fixed configuration for few-shot learning (30 samples/class)"""
    
    # ===== PATHS =====
    DATASET_PATH = '/kaggle/working/chest_xray_4classes'
    SEGMENTATION_PATH_1 = '/kaggle/input/chest-x-ray-lungs-segmentation/Chest-X-Ray/Chest-X-Ray'
    SEGMENTATION_PATH_2 = '/kaggle/input/chest-xray-masks-and-labels/Lung Segmentation'
    OUTPUT_DIR = '/kaggle/working/output'
    SEGMENTED_IMAGES_DIR = '/kaggle/working/segmented_images'
    PREPROCESSED_DATASET_PATH = '/kaggle/working/preprocessed_dataset'
    SPLIT_DATASET_PATH = '/kaggle/working/split_dataset'
    MISCLASS_DIR = os.path.join(OUTPUT_DIR, 'misclassified')
    
    # ===== REAL-WORLD DATA PATHS =====
    REAL_WORLD_PATH = '/kaggle/input/realcxr2/chest'
    USE_REAL_WORLD_FINETUNING = True
    REAL_WORLD_SEGMENTED_DIR = '/kaggle/working/real_world_segmented'
    REAL_WORLD_SAMPLES_PER_CLASS = None  # Use ALL available real data (was 30)
    
    # ===== OFFLINE AUGMENTATION =====
    USE_OFFLINE_AUGMENTATION = True   # Enable offline augmentation
    OFFLINE_AUG_COPIES_PER_IMAGE = 10 # More copies for small dataset
    SAVE_AUGMENTED_IMAGES = True
    AUGMENTED_DATA_DIR = '/kaggle/working/augmented_real_world'
    USE_STRONGER_REAL_TRANSFORMS = True  # NEW: Use stronger transforms for real data
    
    # ===== SEGMENTATION OPTIONS =====
    USE_SEGMENTATION_1 = True
    USE_SEGMENTATION_2 = True
    COMBINE_SEGMENTATIONS = False
    COMBINE_METHOD = 'union'
    
    # ===== MODEL PARAMETERS =====
    SEG_IMAGE_SIZE = 256
    CLS_IMAGE_SIZE = 224
    SEG_BATCH_SIZE = 8
    CLS_BATCH_SIZE = 16
    SEG_EPOCHS = 10
    CLS_EPOCHS = 50
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # ===== FINE-TUNING PARAMETERS (TB RESCUE CHANGES) =====
    FINETUNE_EPOCHS = 30         # Up from 30 (Longer training)
    FINETUNE_LR_BACKBONE = 1e-5  # Down from 2e-5 (Gentler LR)
    FINETUNE_LR_CLASSIFIER = 2e-5 # Down from 5e-5 (Gentler LR)
    FINETUNE_BATCH_SIZE = 16
    FREEZE_BACKBONE_RATIO = 0.90
    FINETUNE_WEIGHT_DECAY = 1e-4
    
    # ===== TWO-STREAM MIXING (MORE REAL DATA) =====
    USE_TWO_STREAM_MIXING = True
    INITIAL_SYN_RATIO = 0.5      # Down from 0.7 (Start with more real)
    FINAL_SYN_RATIO = 0.2        # Down from 0.3 (End with even more real)
    RATIO_TRANSITION_EPOCH = 10  # Down from 20 (Faster transition to real)
    
    # ===== KNOWLEDGE DISTILLATION (FIXED) =====
    USE_KNOWLEDGE_DISTILLATION = True
    KD_TEMPERATURE = 5.0
    KD_ALPHA_REAL = 0.0
    KD_ALPHA_SYN = 0.7
    KD_REAL_LOSS_WEIGHT = 1.0  # FIXED: Reduced from 2.0
    
    # ===== TB RESCUE =====
    TB_OVERSAMPLE_FACTOR = 1.5  # FIXED: Reduced from 3 (no oversample)
    MIN_TB_RECALL = 0.40        # NEW: TB Guardrail
    
    # ===== ADAPTIVE BATCH NORM =====
    USE_ADAPTIVE_BN = True
    ADABN_EPOCHS = 2
    
    # ===== CLASS CALIBRATION =====
    USE_CLASS_CALIBRATION = True
    CALIBRATION_EPOCHS = 50
    CALIBRATION_LR = 0.01
    
    # ===== TTA PARAMETERS =====
    USE_TTA = True
    USE_TTA_REAL_WORLD = False  # NEW: Disable TTA for real-world (may hurt)
    TTA_AUGMENTATIONS = 4  # FIXED: Reduced from 5
    
    # ===== DEVICE =====
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RANDOM_SEED = 42
    
    # ===== SPLIT RATIOS =====
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    REAL_WORLD_VAL_RATIO = 0.20 # Smaller val = more train
    FINETUNE_KFOLD_SPLITS = 5   # More folds = more stable + ensemble
    
    # ===== CLASS CONFIGURATION =====
    CLASS_NAMES = ['normal', 'tuberculosis', 'pneumonia', 'lung_cancer']
    NUM_CLASSES = len(CLASS_NAMES)
    MINORITY_CLASSES = ['lung_cancer', 'pneumonia']
    
    # ===== PREPROCESSING =====
    PREPROCESSING_ENABLED = True
    PREPROCESS_OFFLINE = True
    REMOVE_BORDERS = True
    REMOVE_WATERMARKS = True
    APPLY_PREPROCESSING_CLAHE = True
    CLAHE_CLIP_LIMIT = 4.0
    MAINTAIN_ASPECT_RATIO = True
    
    # ===== TRAINING PARAMETERS =====
    USE_AMP = True
    GRADIENT_CLIP = 1.0
    EARLY_STOPPING = True
    EARLY_STOP_PATIENCE = 10
    MIN_DELTA = 1e-4
    
    # ===== REGULARIZATION =====
    DROPOUT_RATE = 0.4  # FIXED: Reduced from 0.6
    WEIGHT_DECAY = 1e-3
    
    # ===== AUGMENTATION =====
    EXTREME_AUGMENTATION = False
    CLASS_SPECIFIC_AUGMENTATION = True
    
    # ===== IMPROVEMENTS (TB RESCUE CHANGES) =====
    USE_CLASS_WEIGHTS = True
    LABEL_SMOOTHING = 0.1  # FIXED: Increased from 0.05
    MIXUP_ALPHA = 0.2
    MIXUP_PROB = 0.30
    MIXUP_PROB_FINETUNE = 0.20 # Up from 0.0 (Re-enabled Mixup)
    USE_FOCAL_LOSS = True
    FOCAL_LOSS_ALPHA = 0.75
    FOCAL_LOSS_GAMMA = 2.5
    
    # ===== TWO-PHASE PIPELINE =====
    APPLY_MASK_TO_CLASSIFICATION = True
    MASK_THRESHOLD = 0.5
    TUNE_MASK_THRESHOLD = True
    MORPH_CLEANUP = True
    MORPH_KERNEL = 3
    
    # ===== EMA =====
    USE_EMA = True  # FIXED: Enabled
    EMA_DECAY = 0.999
    
    # ===== SAMPLING =====
    USE_WEIGHTED_SAMPLER = True
    
    # ===== MONITORING AND CALIBRATION =====
    MONITOR_METRIC = 'macro_f1'
    USE_PER_CLASS_THRESHOLDS = True
    USE_TEMPERATURE_SCALING = True
    
    # ===== GUARDRAILS (FIXED) =====
    MAX_SYNTHETIC_F1_DROP = 0.15
    
    # ===== DOMAIN ALIGNMENT (ENHANCED) =====
    USE_CORAL_LOSS = True
    CORAL_LAMBDA = 0.3       # Up from 0.15 (stronger alignment)
    CORAL_WARMUP_EPOCHS = 3  # Faster warmup
    USE_MMD_LOSS = True      # NEW: Additional MMD loss
    MMD_LAMBDA = 0.2         # NEW: MMD weight
    USE_CONTRASTIVE = True   # NEW: Supervised contrastive
    CONTRASTIVE_LAMBDA = 0.1 # NEW: Contrastive weight
    CONTRASTIVE_TEMP = 0.1   # NEW: Temperature
    
    # ===== GRADIENT ACCUMULATION (TB RESCUE CHANGES) =====
    ACCUM_STEPS = 4  # Up from 2
    
    # ===== PREADAPTATION (MORE REAL DATA FROM START) =====
    PREADAPT_EPOCHS = 3          # Shorter pre-adaptation
    PREADAPT_REAL_RATIO = 0.3    # Up from 0.1 (30% real in preadapt)
    
    # ===== ANALYSIS =====
    SAVE_MISCLASSIFIED = True
    ANALYZE_ARTIFACTS = True
    ANALYSIS_SAMPLE_SIZE = 200

    @classmethod
    def create_dirs(cls):
        """Create necessary directories."""
        for dir_path in [cls.OUTPUT_DIR, cls.SEGMENTED_IMAGES_DIR, cls.PREPROCESSED_DATASET_PATH, 
                         cls.SPLIT_DATASET_PATH, cls.MISCLASS_DIR, cls.REAL_WORLD_SEGMENTED_DIR, 
                         cls.AUGMENTED_DATA_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        for split in ['train', 'validation', 'test']:
            os.makedirs(os.path.join(cls.SEGMENTED_IMAGES_DIR, split), exist_ok=True)
