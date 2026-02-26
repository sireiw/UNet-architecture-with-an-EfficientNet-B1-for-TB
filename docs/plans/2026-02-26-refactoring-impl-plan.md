# Refactoring Notebook to Proper Python Package Implementation Plan

**Goal:** Extract the 3,000-line monolithic codebase from `2phase-loadmodel-completedemo.ipynb` into a clean, properly structured Python package under `src/` and a standalone `main.py` entrypoint.
**Architecture:** The code will be structured into domain-specific modules: Config, Data, Models, Training, Evaluation, and Utils. The notebook will become a thin UI wrapper that imports `src`.
**Tech Stack:** Python, PyTorch, Scikit-learn, OpenCV, Albumentations.

---

### Task 1: Basic Structure & Configuration

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Modify: `src/config.py`

**Step 1: Write `.gitignore` and `requirements.txt`**
- Create `.gitignore` to ignore `__pycache__`, `/data`, `/output`, `*.dcm`, and standard Python ignores.
- Create `requirements.txt` based on the imports in Cell 4 (torch, torchvision, albumentations, opencv-python, pandas, scikit-learn, etc).

**Step 2: Wipe and rewrite `src/config.py`**
- Delete the old contents of `src/config.py`.
- Copy the `Config` class from Cell 4 containing all path variables, hyperparameters, and constants.

**Step 3: Commit**
```bash
git init
git add .gitignore requirements.txt src/config.py
git commit -m "chore: setup base configuration and ignores"
```

---

### Task 2: Utilities

**Files:**
- Create: `src/utils.py`

**Step 1: Implement `src/utils.py`**
- Extract utility functions from the notebook (hashing, checking duplicates, reading images safely).
- Ensure explicit type hints and well-formatted docstrings.

**Step 2: Test utility imports**
Run: `python3 -c "import src.utils"`
Expected: No errors.

**Step 3: Commit**
```bash
git add src/utils.py
git commit -m "feat: add utility functions"
```

---

### Task 3: Data Management

**Files:**
- Create: `src/data/__init__.py`
- Create: `src/data/collection.py`
- Create: `src/data/dataset.py`

**Step 1: Implement `src/data/collection.py`**
- Copy all data downloading/organization logic currently in Cell 1.
- Encapsulate the Kaggle-specific parsing logic into functions like `collect_data()`.

**Step 2: Implement `src/data/dataset.py`**
- Extract dataset classes from Notebook Cell 4 (`LungSegmentationDataset`, `EnhancedLungDataset`, `TwoStreamBatchSampler`, `create_balanced_real_indices`).
- Add offline augmentation / safe medical augmentation classes here (or in a separate file if needed, but `dataset.py` is fine for now).

**Step 3: Test imports**
Run: `python3 -c "from src.data.dataset import TwoStreamBatchSampler"`
Expected: No errors.

**Step 4: Commit**
```bash
git add src/data/
git commit -m "feat: add data collection and dataset classes"
```

---

### Task 4: Modeling & Loss Functions

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/architectures.py`
- Create: `src/models/losses.py`

**Step 1: Implement `src/models/architectures.py`**
- Extract `UNetSegmentationModel`, `DomainRobustClassifier`, `EMA`, and `BiasTempCalibrator`.

**Step 2: Implement `src/models/losses.py`**
- Extract `coral_loss`, `mmd_loss`, `SupConLoss`, `kd_loss`, `DiceLoss`, `BalancedFocalLoss`.

**Step 3: Commit**
```bash
git add src/models/
git commit -m "feat: add model architectures and loss functions"
```

---

### Task 5: Evaluation & Visualization

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/metrics.py`
- Create: `src/evaluation/visualization.py`

**Step 1: Extract Visualization and Evaluation**
- Extract `plot_confusion_matrix`, `plot_training_curves`, `save_detailed_report`, and Grad-CAM visualization classes into `visualization.py`.
- Extract metric calculation into `metrics.py`.

**Step 2: Commit**
```bash
git add src/evaluation/
git commit -m "feat: add evaluation and visualization tools"
```

---

### Task 6: Training Pipeline

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/engine.py`
- Create: `src/training/pipeline.py`

**Step 1: Extract the Training loops**
- Extract `train_segmentation_model`, `train_classification_model`, and `improved_finetune_with_kd` into `engine.py`.
- Implement `pipeline.py` that takes `Config` and strings together Phase 1, Phase 2, and Phase 3 (mirroring the main loop in the notebook).

**Step 2: Commit**
```bash
git add src/training/
git commit -m "feat: add training engine and orchestrator pipeline"
```

---

### Task 7: Entrypoint & Notebook Cleanup

**Files:**
- Create: `main.py`
- Modify: `2phase-loadmodel-completedemo.ipynb`
- Modify: `README.md`

**Step 1: Implement `main.py`**
- Use `argparse` to allow running specific phases (e.g. `python main.py --train-all`).
- Import all the logic from `src`.

**Step 2: Strip Notebook**
- Use the Jupyter notebook JSON format parser to safely empty the 3000-line cell.
- Replace with clean imports and simple execution cells pointing to `src`.

**Step 3: Verify execution**
- Run a minimal dummy run via `main.py`.

**Step 4: Commit**
```bash
git add main.py 2phase-loadmodel-completedemo.ipynb README.md
git commit -m "refactor: add main entrypoint and clean notebook"
```
