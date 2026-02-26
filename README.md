# Tuberculosis and Medical Image Neural Pipeline

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

3. **Running the Pipeline:**
   The `2phase-loadmodel-completedemo.ipynb` Jupyter notebook serves as the high-level demonstration for consuming the `src/` modules. Launch the notebook and execute the cells to begin the dual-phase fine-tuning procedure.

## Kaggle Environment Compatibility
This pipeline was initially targeted towards Kaggle's `/kaggle/working/` directory architectures. Make sure you either run this within Kaggle, or edit the `src/config.py` paths to map appropriately to your localized `data/` setup.

## Code Review
The pipeline enforces:
- Meaningful variable names.
- Clean module structures avoiding long script files.
- Avoiding redundancy via functional encapsulation.
