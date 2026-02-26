# Tuberculosis Deep Learning Diagnosis Pipeline

This repository contains a professional, modular PyTorch pipeline designed to train a highly robust model for **Tuberculosis (TB)** detection from Chest X-Rays. It employs a 3-phase training architecture emphasizing domain robustness, knowledge distillation, and "TB rescue" oversampling logic.

## Abstract

**Objective:** To develop and evaluate the performance of an Artificial Intelligence (AI) system for classifying four lung conditions from chest X-rays: Normal, Tuberculosis (TB), Pneumonia, and Lung Cancer (Nodule/Mass). Performance was evaluated using statistical metrics including Accuracy, Precision, Specificity, and F1-score.

**Methods:** Chest X-ray images were collected from seven standard databases (NIH Chest X-ray, RSNA, VinBigData, TBX11K, NIAID, Shenzhen) and a specific dataset from Bhumibol Adulyadej Hospital. Data underwent quality control and balancing. The development process under the "TB-Rescue" framework consisted of three phases: 1) Lung Segmentation using U-Net architecture, 2) Learning from synthetic data, and 3) Fine-tuning with real-world data. Specifically, a local dataset of 286 images from Bhumibol Adulyadej Hospital (100 Normal, 73 TB, 86 Pneumonia, and 27 Lung Cancer) was utilized to adapt the model to the radiographic characteristics of Thai and Asian populations. The study integrated Knowledge Distillation techniques with advanced loss functions (CORAL Loss, Supervised Contrastive Learning) and employed 5-Fold Cross-Validation alongside mixed Data Augmentation.

**Results:** The baseline model achieved an accuracy of 81.00% when tested on public datasets. However, when applied to clinical real-world data, significant domain shift caused accuracy to drop to 34.97%. Following the fine-tuning process, the model demonstrated significantly improved adaptation to Thai radiographic characteristics, with accuracy on real-world data increasing from 34.97% to 63.00%. Regarding Tuberculosis classification, the fine-tuned model maintained high performance, achieving an F1-Score of 0.80, a Sensitivity of 80.00%, and a Specificity of 93.00%.

**Conclusion:** While the baseline model exhibited high potential on standard datasets, its practical application was limited by domain shift. The fine-tuning process proved critical in mitigating this issue, recovering model performance for clinical application in Thailand. Despite the occurrence of catastrophic forgetting regarding public data, the model demonstrated superior domain adaptation to real-world data, which is essential for the practical deployment of AI in clinical settings.

**Keywords:** Artificial Intelligence, Lung Disease Classification, Chest X-ray, Tuberculosis, U-Net, EfficientNet-B1, Domain Shift, Domain Adaptation

## Architecture

The previous monolithic Jupyter Notebook has been fully refactored into a scalable Python package under `src/`.

```
.
├── main.py                     # CLI entrypoint for the pipeline
├── 2phase-loadmodel-completedemo.ipynb # Interactive notebook (imports from src/)
├── requirements.txt            # Python dependencies
├── src/
│   ├── config.py               # Central hyperparameters and paths
│   ├── utils.py                # Image validation & hashing
│   ├── data/
│   │   ├── collection.py       # Kaggle dataset acquisition (TBX11K, NIH, etc.)
│   │   └── dataset.py          # PyTorch Datasets & Samplers
│   ├── models/
│   │   ├── architectures.py    # UNet Segmentation & EfficientNet-B1 Classifier
│   │   └── losses.py           # Focal, Coral, MMD, Supervised Contrastive Losses
│   ├── training/
│   │   ├── engine.py           # Core training loops for Phase 1/2/3
│   │   └── pipeline.py         # Main orchestrator
│   └── evaluation/
│       ├── eval_utils.py       # Metrics calculation
│       └── visualization.py    # Grad-CAM overlays and CM matrices
```

## Setup Instructions

### 1. Environment Requirements
Ensure you have Python 3.10+ and a CUDA-capable GPU (or run on Kaggle/Colab).

```bash
pip install -r requirements.txt
```

### 2. Prepare Data
The original pipeline leverages several Kaggle datasets (NIH, TBX11K, VinBigData, JSRT, Pakistan Hospital Dataset). You can trigger the automated data scraping and preprocessing by running:

```bash
python main.py --data
```

This will automatically assemble the datasets, check for file duplicates via MD5 hashing, validate minimum image dimensions, and organize them into `./chest_xray_4classes`.

## Usage

You can run the pipeline sequentially via the command line:

```bash
# 1. Run just the Full 3-Phase Training
python main.py --train

# 2. Run just Evaluation & Visualization (assuming a trained model)
python main.py --eval

# 3. Run EVERYTHING End-to-End
python main.py --all
```

### Interactive Jupyter Notebook

For interactive debugging, visualizations, and exploratory data analysis, use `2phase-loadmodel-completedemo.ipynb`. The notebook now cleanly imports from `src/` and benefits from `%autoreload 2`, allowing you to modify the underlying python package code and immediately see the results in your cells without restarting the kernel.

## Core Features (TB Rescue Plan)

- **Domain Robustness Extraction**: Removes the standard class head from EfficientNet-B1 and adds a customized Dropout+Kaiming initialization head.
- **Advanced Loss Alignment**: Uses `coral_loss` and `mmd_loss` to align feature distributions across diverse hospital source domains.
- **Knowledge Distillation (KD)**: Phase 3 incorporates a teacher-student KD architecture to perform gentle fine-tuning on real clinical data without catastrophic forgetting of the synthetic source.
- **Class-Balanced Sampling**: Features a custom `TwoStreamBatchSampler` that guarantees strict ratios of real-to-synthetic data per batch, preventing domain dominance.
- **Grad-CAM Interpretability**: Built-in support to highlight ROIs (Regions of Interest) that drove model predictions, overlaid transparently on the original X-ray images.

## License
MIT License
