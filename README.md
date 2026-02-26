# Tuberculosis and Medical Image Neural Pipeline

## Abstract

**Objective:** To develop and evaluate the performance of an Artificial Intelligence (AI) system for classifying four lung conditions from chest X-rays: Normal, Tuberculosis (TB), Pneumonia, and Lung Cancer (Nodule/Mass). Performance was evaluated using statistical metrics including Accuracy, Precision, Specificity, and F1-score.

**Methods:** Chest X-ray images were collected from seven standard databases (NIH Chest X-ray, RSNA, VinBigData, TBX11K, NIAID, Shenzhen) and a specific dataset from Bhumibol Adulyadej Hospital. Data underwent quality control and balancing. The development process under the "TB-Rescue" framework consisted of three phases: 1) Lung Segmentation using U-Net architecture, 2) Learning from synthetic data, and 3) Fine-tuning with real-world data. Specifically, a local dataset of 286 images from Bhumibol Adulyadej Hospital (100 Normal, 73 TB, 86 Pneumonia, and 27 Lung Cancer) was utilized to adapt the model to the radiographic characteristics of Thai and Asian populations. The study integrated Knowledge Distillation techniques with advanced loss functions (CORAL Loss, Supervised Contrastive Learning) and employed 5-Fold Cross-Validation alongside mixed Data Augmentation.

**Results:** The baseline model achieved an accuracy of 81.00% when tested on public datasets. However, when applied to clinical real-world data, significant domain shift caused accuracy to drop to 34.97%. Following the fine-tuning process, the model demonstrated significantly improved adaptation to Thai radiographic characteristics, with accuracy on real-world data increasing from 34.97% to 63.00%. Regarding Tuberculosis classification, the fine-tuned model maintained high performance, achieving an F1-Score of 0.80, a Sensitivity of 80.00%, and a Specificity of 93.00%.

**Conclusion:** While the baseline model exhibited high potential on standard datasets, its practical application was limited by domain shift. The fine-tuning process proved critical in mitigating this issue, recovering model performance for clinical application in Thailand. Despite the occurrence of catastrophic forgetting regarding public data, the model demonstrated superior domain adaptation to real-world data, which is essential for the practical deployment of AI in clinical settings.

**Keywords:** Artificial Intelligence, Lung Disease Classification, Chest X-ray, Tuberculosis, U-Net, EfficientNet-B1, Domain Shift, Domain Adaptation

---

This project contains a professional and modular deep learning pipeline for processing medical imaging datasets, specialized for the classification and segmentation of Tuberculosis, Pneumonia, Lung Cancer, and "Normal" Chest X-Rays. It is designed around the principles of DRY (Don't Repeat Yourself), single responsibility, and readability.

## Features

- **Modular Architecture**: Code is split up cleanly into configuration, utilities, dataset processing, modeling, losses, and training logic.
- **Robust Pipeline**: Includes features like Adaptive Batch Norm, Exponential Moving Averages (EMA), Knowledge Distillation, and Two-Stream Class Balancing.
- **Specialized Losses**: Implements Class-Balanced Focal Loss, CORAL Loss, Maximum Mean Discrepancy (MMD) Loss, Supervised Contrastive Loss, Dice Loss, and Label Smoothing.
- **Data Engineering**: Contains specialized scripts for processing massive datasets such as TBX11K, NIH Chest X-rays, RSNA, and Shenzhen Hospital datasets, with robust dataset availability checks and duplicate prevention using MD5 hashing.

## File Structure

The core pipeline has been abstracted out of monolithic Jupyter notebooks into a professional Python package (`src/`):

```text
├── src/
│   ├── config.py           # Configuration parameters and path routing
│   ├── data_collection.py  # Dataset compilation, resizing, and preparation
│   ├── dataset.py          # PyTorch dataset logic and Two-Stream Samplers
│   ├── losses.py           # Advanced custom loss functions
│   ├── models.py           # EMA wrapping and Architecture calibrations
│   ├── train.py            # Logic for model training and TTA inference
│   └── utils.py            # IO functions, duplicate checks, hashing
├── 2phase-loadmodel-completedemo.ipynb  # Clean, high-level notebook
├── requirements.txt        # Package dependencies
└── README.md               # Project documentation
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup Directories:**
   You can initialize the required output and processing directory structure dynamically via the python module:
   ```python
   from src.config import Config
   Config.create_dirs()
   ```

## Running the Pipeline:

The `2phase-loadmodel-completedemo.ipynb` Jupyter notebook serves as the high-level demonstration for consuming the `src/` modules. Launch the notebook and execute the cells to begin the dual-phase fine-tuning procedure.

## Dataset Instructions

Because medical image datasets are massive in size, they are **not** included in this repository. You must download them manually to execute the training code.

By default, the pipeline expects the datasets to be located at `/kaggle/input/` (as configured in `src/config.py`). If you are running locally, either create this mock directory structure or edit the paths in `src/config.py`.

1. **TBX11K Dataset**
   - **Download:** Search for "TBX11K" or "TBX11K Simplified" on Kaggle.
   - **Placement:** `/kaggle/input/tbx11k-simplified/` or `/kaggle/input/tbx11k/`

2. **NIH Chest X-ray Dataset**
   - **Download:** Search for "NIH Chest X-rays" on Kaggle.
   - **Placement:** Ensure the raw DICOM/PNG files and `Data_Entry_2017.csv` metadata file are placed under the accessible dataset path.

3. **RSNA / Shenzhen Hospital / Real World Data**
   - **Download:** Search for "Real CXR" or the respective source on Kaggle.
   - **Placement:** `/kaggle/input/realcxr2/chest/` (as pointed to by `REAL_WORLD_PATH` in the configuration).

## Kaggle Environment Compatibility
This pipeline was initially targeted towards Kaggle's `/kaggle/working/` directory architectures. Make sure you either run this within Kaggle, or edit the `src/config.py` paths to map appropriately to your localized `data/` setup.

## Code Review
The pipeline enforces:
- Meaningful variable names.
- Clean module structures avoiding long script files.
- Avoiding redundancy via functional encapsulation.
