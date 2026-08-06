"""
Experiment: CCDS — Confidence-Calibrated Dynamic Selection — CAXF CharCNN, Seed 100
=====================================================================================
Novel ensemble using Jensen-Shannon Divergence and Shannon entropy to route
each sample through a disagreement spectrum.

Run from src/ directory:
    python experiments/run_ccds_caxf_charcnn.py
"""

import time
import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

_here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.caxf.caxf_extractor_charcnn import CAXFExtractor
from src.ensemble.ccds import CCDS
from config import SEED

# ===============================
# PATHS & CONFIG
# ===============================
suffix = f"_seed_{SEED}"

DATA_PATH = "data/processed/Final_XSS_4class_dataset.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join("..", DATA_PATH)

CACHE_DIR = "results/cache/charcnn"
OUTPUT_DIR = "results/caxf_char_cnn_results"
if not os.path.exists("results") and os.path.exists("../results"):
    CACHE_DIR = os.path.join("..", CACHE_DIR)
    OUTPUT_DIR = os.path.join("..", OUTPUT_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TXT_OUT = os.path.join(OUTPUT_DIR, f"ccds_results{suffix}.txt")
PNG_OUT = os.path.join(OUTPUT_DIR, f"ccds_results{suffix}.png")

CAT_RESULT_PATH = os.path.join(OUTPUT_DIR, f"catboost_model{suffix}.pkl")
CAT_CACHE_PATH  = os.path.join(CACHE_DIR, f"cat{suffix}.pkl")


class CatBoostIntAdapter:
    def __init__(self, model, label_encoder):
        self._model = model
        self._le = label_encoder

    def predict(self, X):
        str_preds = self._model.predict(X)
        flat = [p[0] if hasattr(p, "__len__") and not isinstance(p, str) else p
                for p in str_preds]
        return self._le.transform(flat)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


def main():
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    log("=" * 60)
    log("CCDS ENSEMBLE PIPELINE — CAXF CHAR-CNN")
    log("Confidence-Calibrated Dynamic Selection (Novel Algorithm)")
    log("Routing: JSD disagreement spectrum → entropy-weighted strategies")
    log("=" * 60)

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------
    log("Loading dataset...")
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    X = df["Sentence"].astype(str)
    y = df["Final_Label"]
    log(f"Total samples: {len(df)}")
    log("Class distribution:")
    log(str(y.value_counts()))
    log()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    log(f"Splitting dataset (70:30, seed={SEED})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.30, random_state=SEED, stratify=y_encoded
    )
    log(f"Train size: {len(X_train)}")
    log(f"Test  size: {len(X_test)}")
    log()

    # ---------------------------------------------------------------
    # Load cached CharCNN embeddings
    # ---------------------------------------------------------------
    cache_train_emb = f"{CACHE_DIR}/X_train_embed{suffix}.npy"
    cache_test_emb  = f"{CACHE_DIR}/X_test_embed{suffix}.npy"

    if os.path.exists(cache_train_emb) and os.path.exists(cache_test_emb):
        log("⚡ Loading cached CharCNN embeddings...")
        X_train_embed = np.load(cache_train_emb).astype(np.float32)
        X_test_embed  = np.load(cache_test_emb).astype(np.float32)
    else:
        log("Running CAXF CharCNN feature extraction...")
        t0 = time.time()
        caxf = CAXFExtractor()
        caxf.fit(X_train)
        X_train_embed = caxf.transform(X_train)
        X_test_embed  = caxf.transform(X_test)
        if hasattr(X_train_embed, "toarray"):
            X_train_embed = X_train_embed.toarray()
            X_test_embed  = X_test_embed.toarray()
        np.save(cache_train_emb, X_train_embed)
        np.save(cache_test_emb, X_test_embed)
        log(f"Saved. Time: {round(time.time()-t0, 2)}s")
        X_train_embed = X_train_embed.astype(np.float32)
        X_test_embed  = X_test_embed.astype(np.float32)

    log(f"Embedding shape (train): {X_train_embed.shape}")
    log(f"Embedding shape (test) : {X_test_embed.shape}")
    log()

    # ---------------------------------------------------------------
    # SMOTE
    # ---------------------------------------------------------------
    log("Applying SMOTE...")
    smote = SMOTE(random_state=SEED)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_embed, y_train)
    log(f"After SMOTE — class counts: {dict(pd.Series(y_train_smote).value_counts())}")
    log()

    X_train_orig = X_train_embed
    y_train_orig = y_train

    # ---------------------------------------------------------------
    # Load base models
    # ---------------------------------------------------------------
    cache_lgbm = os.path.join(CACHE_DIR, f"lgbm{suffix}.pkl")
    cache_xgb  = os.path.join(CACHE_DIR, f"xgb{suffix}.pkl")
    cat_path   = CAT_RESULT_PATH if os.path.exists(CAT_RESULT_PATH) else CAT_CACHE_PATH

    if os.path.exists(cache_lgbm) and os.path.exists(cache_xgb) and os.path.exists(cat_path):
        log("⚡ Loading trained base models...")
        lgbm     = joblib.load(cache_lgbm)
        xgb      = joblib.load(cache_xgb)
        _cat_raw = joblib.load(cat_path)
        cat      = CatBoostIntAdapter(_cat_raw, le)
    else:
        log("Training base models...")
        lgbm = LGBMClassifier(
            objective="multiclass", num_class=4, n_estimators=100,
            learning_rate=0.1, random_state=SEED, n_jobs=-1
        )
        lgbm.fit(X_train_smote, y_train_smote)
        joblib.dump(lgbm, cache_lgbm)

        xgb = XGBClassifier(
            objective="multi:softprob", num_class=4, n_estimators=100,
            learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=SEED, n_jobs=-1
        )
        xgb.fit(X_train_smote, y_train_smote)
        joblib.dump(xgb, cache_xgb)

        class_counts = pd.Series(y_train_orig).value_counts()
        total = len(y_train_orig)
        class_weights = {
            cls: total / (len(class_counts) * count)
            for cls, count in class_counts.items()
        }
        _cat_raw = CatBoostClassifier(
            loss_function="MultiClass", iterations=100, learning_rate=0.1,
            depth=4, class_weights=class_weights, random_seed=SEED,
            thread_count=4, task_type="CPU", verbose=False
        )
        _cat_raw.fit(X_train_orig, y_train_orig)
        joblib.dump(_cat_raw, CAT_RESULT_PATH)
        cat = CatBoostIntAdapter(_cat_raw, le)

    # ---------------------------------------------------------------
    # CCDS Ensemble
    # ---------------------------------------------------------------
    log("=" * 60)
    log("Evaluating CCDS Ensemble...")
    log("Thresholds: theta_low=0.05, theta_high=0.15, min_confidence=0.5")
    log("=" * 60)

    ccds = CCDS(
        models=[lgbm, xgb, cat],
        theta_low=0.05,
        theta_high=0.15,
        min_confidence=0.5
    )

    t0 = time.time()
    final_preds = ccds.predict(X_test_embed)
    inf_time = round(time.time() - t0, 4)
    log(f"CCDS Inference time: {inf_time} seconds\n")

    y_test_labels = le.inverse_transform(y_test)
    pred_labels   = le.inverse_transform(final_preds.astype(int))

    acc       = accuracy_score(y_test_labels, pred_labels)
    precision = precision_score(y_test_labels, pred_labels, average="macro")
    recall    = recall_score(y_test_labels, pred_labels, average="macro")
    f1        = f1_score(y_test_labels, pred_labels, average="macro")

    log(f"Accuracy : {round(acc, 4)}")
    log("\nClassification Report:")
    log(classification_report(y_test_labels, pred_labels))
    log("Paper Format Metrics")
    log(f"Precision   : {round(precision, 4)}")
    log(f"Recall      : {round(recall, 4)}")
    log(f"Average F1  : {round(f1, 4)}")

    with open(TXT_OUT, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nResults saved to: {TXT_OUT}")

    cm = confusion_matrix(y_test_labels, pred_labels, labels=le.classes_)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix — CCDS Ensemble (CAXF CharCNN)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PNG_OUT)
    plt.close()
    print(f"Confusion Matrix plot saved to: {PNG_OUT}")


if __name__ == "__main__":
    main()
