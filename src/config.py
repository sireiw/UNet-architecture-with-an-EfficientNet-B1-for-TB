import os
import torch

class Config:
    """Fixed configuration for few-shot learning (30 samples/class)"""
    
    # ===== PATHS =====
    DATASET_PATH = 'chest_xray_4classes'
    SEGMENTATION_PATH_1 = 'input/chest-x-ray-lungs-segmentation/Chest-X-Ray/Chest-X-Ray'
    SEGMENTATION_PATH_2 = 'input/chest-xray-masks-and-labels/Lung Segmentation'
    OUTPUT_DIR = 'output'
    SEGMENTED_IMAGES_DIR = 'segmented_images'
    PREPROCESSED_DATASET_PATH = 'preprocessed_dataset'
    SPLIT_DATASET_PATH = 'split_dataset'
    MISCLASS_DIR = os.path.join(OUTPUT_DIR, 'misclassified')
    
    # ===== REAL-WORLD DATA PATHS =====
    REAL_WORLD_PATH = 'input/realcxr2/chest'
    USE_REAL_WORLD_FINETUNING = True
    REAL_WORLD_SEGMENTED_DIR = 'real_world_segmented'
    REAL_WORLD_SAMPLES_PER_CLASS = None  # Use ALL available real data
    
    # ===== OFFLINE AUGMENTATION =====
    USE_OFFLINE_AUGMENTATION = True   # Enable offline augmentation
    OFFLINE_AUG_COPIES_PER_IMAGE = 10 # More copies for small dataset
    SAVE_AUGMENTED_IMAGES = True
    AUGMENTED_DATA_DIR = 'augmented_real_world'
    USE_STRONGER_REAL_TRANSFORMS = True  # Use stronger transforms for real data
    
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
    FINETUNE_EPOCHS = 30         
    FINETUNE_LR_BACKBONE = 1e-5  
    FINETUNE_LR_CLASSIFIER = 2e-5 
    FINETUNE_BATCH_SIZE = 16
    FREEZE_BACKBONE_RATIO = 0.90
    FINETUNE_WEIGHT_DECAY = 1e-4
    
    # ===== TWO-STREAM MIXING =====
    USE_TWO_STREAM_MIXING = True
    INITIAL_SYN_RATIO = 0.5      
    FINAL_SYN_RATIO = 0.2        
    RATIO_TRANSITION_EPOCH = 10  
    
    # ===== KNOWLEDGE DISTILLATION =====
    USE_KNOWLEDGE_DISTILLATION = True
    KD_TEMPERATURE = 5.0
    KD_ALPHA_REAL = 0.0
    KD_ALPHA_SYN = 0.7
    KD_REAL_LOSS_WEIGHT = 1.0  
    
    # ===== TB RESCUE =====
    TB_OVERSAMPLE_FACTOR = 1.5  
    MIN_TB_RECALL = 0.40        
    
    # ===== ADAPTIVE BATCH NORM =====
    USE_ADAPTIVE_BN = True
    ADABN_EPOCHS = 2
    
    # ===== CLASS CALIBRATION =====
    USE_CLASS_CALIBRATION = True
    CALIBRATION_EPOCHS = 50
    CALIBRATION_LR = 0.01
    
    # ===== TTA PARAMETERS =====
    USE_TTA = True
    USE_TTA_REAL_WORLD = False  
    TTA_AUGMENTATIONS = 4  
    
    # ===== DEVICE =====
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    RANDOM_SEED = 42
    
    # ===== SPLIT RATIOS =====
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    REAL_WORLD_VAL_RATIO = 0.20 
    FINETUNE_KFOLD_SPLITS = 5   
    
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
    DROPOUT_RATE = 0.4  
    WEIGHT_DECAY = 1e-3
    
    # ===== AUGMENTATION =====
    EXTREME_AUGMENTATION = False
    CLASS_SPECIFIC_AUGMENTATION = True
    
    # ===== IMPROVEMENTS =====
    USE_CLASS_WEIGHTS = True
    LABEL_SMOOTHING = 0.1  
    MIXUP_ALPHA = 0.2
    MIXUP_PROB = 0.30
    MIXUP_PROB_FINETUNE = 0.20 
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
    USE_EMA = True  
    EMA_DECAY = 0.999
    
    # ===== SAMPLING =====
    USE_WEIGHTED_SAMPLER = True
    
    # ===== MONITORING AND CALIBRATION =====
    MONITOR_METRIC = 'macro_f1'
    USE_PER_CLASS_THRESHOLDS = True
    USE_TEMPERATURE_SCALING = True
    
    # ===== GUARDRAILS =====
    MAX_SYNTHETIC_F1_DROP = 0.15
    
    # ===== DOMAIN ALIGNMENT =====
    USE_CORAL_LOSS = True
    CORAL_LAMBDA = 0.3       
    CORAL_WARMUP_EPOCHS = 3  
    USE_MMD_LOSS = True      
    MMD_LAMBDA = 0.2         
    USE_CONTRASTIVE = True   
    CONTRASTIVE_LAMBDA = 0.1 
    CONTRASTIVE_TEMP = 0.1   
    
    # ===== GRADIENT ACCUMULATION =====
    ACCUM_STEPS = 4  
    
    # ===== PREADAPTATION =====
    PREADAPT_EPOCHS = 3          
    PREADAPT_REAL_RATIO = 0.3    
    
    # ===== ANALYSIS =====
    SAVE_MISCLASSIFIED = True
    ANALYZE_ARTIFACTS = True
    ANALYSIS_SAMPLE_SIZE = 200

    @classmethod
    def create_dirs(cls, base_path: str = "."):
        """Create necessary directories relative to base_path."""
        cls.OUTPUT_DIR = os.path.join(base_path, cls.OUTPUT_DIR)
        cls.SEGMENTED_IMAGES_DIR = os.path.join(base_path, cls.SEGMENTED_IMAGES_DIR)
        cls.PREPROCESSED_DATASET_PATH = os.path.join(base_path, cls.PREPROCESSED_DATASET_PATH)
        cls.SPLIT_DATASET_PATH = os.path.join(base_path, cls.SPLIT_DATASET_PATH)
        cls.MISCLASS_DIR = os.path.join(base_path, cls.MISCLASS_DIR)
        cls.REAL_WORLD_SEGMENTED_DIR = os.path.join(base_path, cls.REAL_WORLD_SEGMENTED_DIR)
        cls.AUGMENTED_DATA_DIR = os.path.join(base_path, cls.AUGMENTED_DATA_DIR)
        
        for dir_path in [cls.OUTPUT_DIR, cls.SEGMENTED_IMAGES_DIR, cls.PREPROCESSED_DATASET_PATH, 
                         cls.SPLIT_DATASET_PATH, cls.MISCLASS_DIR, cls.REAL_WORLD_SEGMENTED_DIR, 
                         cls.AUGMENTED_DATA_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        for split in ['train', 'validation', 'test']:
            os.makedirs(os.path.join(cls.SEGMENTED_IMAGES_DIR, split), exist_ok=True)
