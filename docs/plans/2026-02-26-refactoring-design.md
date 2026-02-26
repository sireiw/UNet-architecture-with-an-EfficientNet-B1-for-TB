# Refactoring Design: TB Project Repository

## Goal
To transform the current monolithic Jupyter notebook (`2phase-loadmodel-completedemo.ipynb`) into a well-structured, professional Python package with a clear separation of concerns, executable via a standalone entry point (`main.py`), and to leave behind a thin notebook that imports these modules.

## Current State
- The notebook currently holds all configuration, dataset collection (Kaggle-specific), PyTorch model definitions (UNet + EfficientNet-B1), loss functions, custom sampling, 3-phase training pipeline, evaluation metrics, and Grad-CAM visualization.
- An initial attempt to refactor created partial modules in `src/` (`config.py`, `dataset.py`, `data_collection.py`, `train.py`, `models.py`, `losses.py`, `utils.py`), but these are completely un-used by the notebook, which contains thousands of lines of duplicated, inline code in Cell 4.
- Lack of a central `main.py` entrypoint.

## Proposed Design Approach

We will pursue **Option 3 (Both)**: completely rewrite the `src/` modules for consistency, robustness, and typing, and clean up the notebook to simply act as a UI client to the newly written core package.

### 1. Code Restructuring & `src/` Architecture
We will wipe and rebuild `src/` with a clean, fully typed architecture:

- **`src/config.py`**: The definitive Dataclass/Config object storing paths, hyperparameters, model constants, and Kaggle-specific constants.
- **`src/utils/`**: Helpers for image validation, hashing, deterministic logging, and augmentations.
- **`src/data/`**: 
  - `collection.py` (The Kaggle data-downloading logic from Cell 1).
  - `dataset.py` (Dataset classes like `EnhancedLungDataset` and `TwoStreamBatchSampler`).
- **`src/models/`**: 
  - `architectures.py` (UNet segmentation and EfficientNet-B1 logic).
  - `losses.py` (All custom loss functions: Focal, MMD, Coral, Dice, etc.).
- **`src/training/`**: 
  - `engine.py` (The core generic training loops).
  - `pipeline.py` (The 3-phase TB rescue logic orchestration).
- **`src/evaluation/`**: Checkpointing, testing metrics, Grad-CAM visualization, and prediction saving.

### 2. Standalone Entrypoint (`main.py`)
- We will craft a `main.py` script in the root directory that imports from `src/` and parses command-line arguments (using `argparse` or `click`) to trigger specific phases: e.g., `--collect-data`, `--train`, `--evaluate`, `--demo`.
- This ensures the project can be run natively on any GPU machine (or via automation), not just interactively in a notebook.

### 3. Notebook Cleanup
- `2phase-loadmodel-completedemo.ipynb` will be completely stripped of its function definitions.
- It will import exclusively from the robust `src/` package instead.
- We will include an `autoreload` magic so notebook development continues smoothly while code is edited in source.

### 4. Git & Project Professionalization
- A solid `.gitignore`.
- Explicit `requirements.txt` / `environment.yml`.
- Standardize the `README.md` to show instructions on how to use `main.py` and the notebook.

## Trade-offs
- **Pros:** Highly reusable, cleanly separated, testable, fully deployable out of Kaggle.
- **Cons:** This requires essentially rewriting the 3K+ line cell into roughly 8-10 python files. It will take diligent care to ensure no data-flow bindings are broken between Phase 1 (Segmentation), Phase 2 (Classification), and Phase 3 (Finetuning).

## Verification Strategy
- Compare the exact flow of the `main()` function previously defined in the notebook against our new `pipeline.py`.
- Run a dummy data collection block / partial training loop locally just to catch missing imports/variable scopes before finalizing.
