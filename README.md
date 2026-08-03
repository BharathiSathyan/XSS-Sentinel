# XSS-Sentinel
Beyond Detection: An Intelligent Ensemble & Deep Learning Approach for XSS Type Classification and Defense

XSS-Sentinel is a machine learning research project for classifying Cross-Site Scripting (XSS) attack types into four distinct classes:
1. Normal
2. Reflected XSS
3. Stored XSS
4. DOM-based XSS

It employs a custom feature extraction framework (**CAXF**) with an ensemble learning classifier (**LCCDE**).

---

## Installation & Setup

1. **Clone the repository** and navigate to the project directory:
   ```bash
   cd XSS-Sentinel
   ```

2. **Activate the virtual environment** and install requirements:
   ```bash
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   # Install dependencies
   pip install -r requirements.txt
   ```

---

## Running the Pipelines

All scripts must be run from the `src/` directory.

```bash
cd src/
```

### 1. Centralized Seed Configuration
Before running any baseline or ensemble scripts, you can configure the random seed for splitting and training in `src/config.py`:

```python
# Open src/config.py
SEED = 42      # Default / Original baseline run
# SEED = 100    # Extra validation seed 1
# SEED = 200    # Extra validation seed 2
```

To run with a different seed, simply uncomment that seed line and comment out the others. Ensure that only **one** seed is active at any time.

### 2. Feature Extractor Import Setting
The active feature extraction method is set by modifying the import lines at the top of the training scripts (`train_lgbm_baseline.py`, `train_xgb_baseline.py`, and `train_catboost_baseline.py`):

- **TF-IDF:** `from src.caxf.caxf_extractor_tfidf import CAXFExtractor`
- **Sentence Embedding:** `from src.caxf.caxf_extractor_sentence_embedding import CAXFExtractor`
- **CharCNN:** `from src.caxf.caxf_extractor_charcnn import CAXFExtractor`

*Note: In baseline scripts, ensure only one of these lines is uncommented at a time.*

### 3. Run Baselines
Run the model training baseline scripts. They will automatically detect the active extractor, redirect their command-line output logs to the appropriate folder, and save confusion matrix plots without blocking GUI popups.

```bash
python train_lgbm_baseline.py
python train_xgb_baseline.py
python train_catboost_baseline.py
```

### 4. Run LCCDE Ensemble Experiments
Run the main ensemble experiment script corresponding to your feature extractor:

```bash
# For TF-IDF
python experiments/run_lccde_caxf_tfidf.py

# For Sentence Embedding
python experiments/run_lccde_caxf_sentence_embedding.py

# For CharCNN
python experiments/run_lccde_caxf_charcnn.py
```

---

## Results and Caching Directories

Outputs are organized under the `results/` folder to prevent data overwrites and avoid pollution between seed runs:

- **Cached Files (`results/cache/<extractor>/`):** Caches `.npy` files for embeddings and `.pkl` for trained models. To prevent state collision, cache files are named using seed suffixes (e.g., `X_train_embed_seed_100.npy`).
- **Original Outputs (Seed 42):** Saved under `results/caxf_<extractor>_results/` with names like `lccde_results.txt` and `train_lgbm_unbalanced_smote.txt`.
- **Validation Run Outputs (e.g., Seed 100):** Saved in the same directories with seed suffixes to keep your original results intact:
  - `lccde_results_seed_100.txt` / `lccde_results_seed_100.png`
  - `train_lgbm_unbalanced_smote_seed_100.txt` / `train_lgbm_unbalanced_smote_seed_100.png`
  - `train_xgb_unbalanced_smote_seed_100.txt` / `train_xgb_unbalanced_smote_seed_100.png`
  - `train_catboost_unbalanced_smote_seed_100.txt` / `train_catboost_unbalanced_smote_seed_100.png`

commands to run:
# CWD must be src/
cd c:\Users\bhara\OneDrive\Desktop\XSS-Sentinel\src

# 1. Individual baselines (CatBoost, LGBM, XGBoost)
python train_catboost_baseline.py
python train_lgbm_baseline.py
python train_xgb_baseline.py

# 2. Full LCCDE ensemble for each extractor type
python experiments/run_lccde_caxf_tfidf.py
python experiments/run_lccde_caxf_sentence_embedding.py
python experiments/run_lccde_caxf_charcnn.py
