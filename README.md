<div align="center">

# 🫁 TB-Rescue: Domain-Adaptive Tuberculosis Detection from Chest X-Rays

**A 3-phase deep learning pipeline for robust, cross-domain lung disease classification**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

[Getting Started](#-getting-started) •
[Architecture](#-architecture) •
[Usage](#-usage) •
[Results](#-key-results) •
[Citation](#-citation)

</div>

---

## 📋 Abstract

> **Objective:** To develop and evaluate an AI system for classifying four lung conditions from chest X-rays — **Normal**, **Tuberculosis (TB)**, **Pneumonia**, and **Lung Cancer (Nodule/Mass)** — with a focus on domain adaptation for clinical deployment in Thailand.

**Methods:** Images were collected from seven standard databases (NIH Chest X-ray, RSNA, VinBigData, TBX11K, NIAID, Shenzhen) plus a local dataset of **286 images from Bhumibol Adulyadej Hospital** (100 Normal, 73 TB, 86 Pneumonia, 27 Lung Cancer). The "TB-Rescue" framework consists of three phases:

1. 🔬 **Lung Segmentation** — U-Net with EfficientNet-B1 encoder
2. 📚 **Source-Domain Learning** — Train on large-scale public datasets
3. 🏥 **Domain Adaptation** — Fine-tune with Knowledge Distillation on real clinical data

The pipeline integrates CORAL Loss, Supervised Contrastive Learning, and 5-Fold Cross-Validation with mixed Data Augmentation.

**Results:** Baseline accuracy dropped from **81.00% → 34.97%** due to domain shift. After fine-tuning, real-world accuracy recovered to **63.00%**, with TB classification achieving **F1=0.80**, **Sensitivity=80%**, and **Specificity=93%**.

**Keywords:** `Tuberculosis` · `Chest X-ray` · `Domain Adaptation` · `U-Net` · `EfficientNet-B1` · `Knowledge Distillation`

---

## 🏗 Architecture

```
tb-rescue/
├── main.py                         # CLI entrypoint
├── requirements.txt                # Dependencies
├── 2phase-loadmodel-completedemo.ipynb  # Interactive notebook
│
└── src/
    ├── config.py                   # Hyperparameters & paths
    ├── utils.py                    # Image validation & hashing
    │
    ├── data/
    │   ├── collection.py           # Dataset acquisition (TBX11K, NIH, etc.)
    │   └── dataset.py              # PyTorch Datasets & Samplers
    │
    ├── models/
    │   ├── architectures.py        # U-Net Segmentation & DomainRobustClassifier
    │   └── losses.py               # Focal, CORAL, MMD, SupCon Losses
    │
    ├── training/
    │   ├── engine.py               # Core training loops (Phase 1/2/3)
    │   └── pipeline.py             # End-to-end orchestrator
    │
    └── evaluation/
        ├── eval_utils.py           # Metrics & threshold tuning
        └── visualization.py        # Grad-CAM & confusion matrices
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or run on [Kaggle](https://kaggle.com) / [Google Colab](https://colab.research.google.com)

### Installation

```bash
# Clone the repository
git clone https://github.com/sireiw/UNet-architecture-with-an-EfficientNet-B1-for-TB.git
cd UNet-architecture-with-an-EfficientNet-B1-for-TB

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Command Line Interface

```bash
# Download & preprocess datasets
python main.py --data

# Run the full 3-phase training pipeline
python main.py --train

# Evaluate & generate Grad-CAM visualizations
python main.py --eval

# Run everything end-to-end
python main.py --all
```

### Jupyter Notebook

For interactive exploration and visualization:

```python
%load_ext autoreload
%autoreload 2

from src.config import Config
from src.training.pipeline import TBPipeline

config = Config()
pipeline = TBPipeline(config)
model = pipeline.execute()
```

---

## 📊 Key Results

| Metric | Baseline (Public) | After Domain Shift | After Fine-tuning |
|--------|:-----------------:|:------------------:|:-----------------:|
| **Accuracy** | 81.00% | 34.97% | **63.00%** |

### TB-Specific Performance (Post Fine-tuning)

| Metric | Score |
|--------|:-----:|
| **F1-Score** | 0.80 |
| **Sensitivity** | 80.00% |
| **Specificity** | 93.00% |

---

## ⭐ Core Features

| Feature | Description |
|---------|-------------|
| **Domain-Robust Classifier** | Custom EfficientNet-B1 head with Dropout + Kaiming init for cross-domain stability |
| **CORAL + MMD Alignment** | Feature distribution alignment across diverse hospital sources |
| **Knowledge Distillation** | Teacher-student architecture prevents catastrophic forgetting during fine-tuning |
| **Two-Stream Sampling** | `TwoStreamBatchSampler` enforces strict real/synthetic ratios per batch |
| **Grad-CAM Interpretability** | Visual ROI overlays showing what drives each prediction |
| **TB-Rescue Oversampling** | Targeted oversampling to recover minority-class (TB) performance |

---

## 📂 Datasets

This pipeline supports the following chest X-ray datasets:

| Dataset | Source | Classes |
|---------|--------|---------|
| [NIH Chest X-ray](https://www.kaggle.com/datasets/nih-chest-xrays/data) | Kaggle | Multi-label |
| [RSNA Pneumonia](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) | Kaggle | Normal / Pneumonia |
| [VinBigData](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) | Kaggle | Multi-label |
| [TBX11K](https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified) | Kaggle | TB / Healthy |
| [Shenzhen TB](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html) | NIAID/NIH | TB / Normal |
| **Bhumibol Adulyadej Hospital** | Local | 4-class (286 images) |

---

## 🔧 Technical Stack

<div align="center">

| Component | Technology |
|-----------|-----------|
| Segmentation | U-Net (segmentation_models_pytorch) |
| Classification | EfficientNet-B1 (torchvision) |
| Augmentation | Albumentations |
| Training | PyTorch + AMP (Mixed Precision) |
| Visualization | Matplotlib + Seaborn + Grad-CAM |

</div>

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{tb_rescue_2026,
  author    = {sireiw},
  title     = {TB-Rescue: Domain-Adaptive Tuberculosis Detection from Chest X-Rays},
  year      = {2026},
  url       = {https://github.com/sireiw/UNet-architecture-with-an-EfficientNet-B1-for-TB},
  license   = {MIT}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Made with ❤️ for advancing TB diagnosis in Thailand**

*If this project helps your research, please consider giving it a ⭐*

</div>
