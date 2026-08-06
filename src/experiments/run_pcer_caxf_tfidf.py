"""
Experiment: Per-Class Expert Routing (PCER) — CAXF TF-IDF Features
=====================================================================
Novel algorithm: pre-computes per-class precision of each base model on training
data, identifies the expert model per class, and routes inference accordingly.

Run from src/ directory:
    python experiments/run_pcer_caxf_tfidf.py
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
sys.path.insert(0, os.path.abspath(os.path.join(_here, "../..")))  # project root
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))     # src/

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.ensemble.pcer import PCER
from cache_utils import resolve_cache_path, load_embedding
from config import SEED

# ===============================
# PATHS & CONFIG
# ===============================
suffix = f"_seed_{SEED}"

DATA_PATH = "data/processed/Final_XSS_4class_dataset.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join("..", DATA_PATH)

CACHE_DIR = "results/cache/tfidf"
OUTPUT_DIR = "results/caxf_tfidf_results"
if not os.path.exists("results") and os.path.exists("../results"):
    CACHE_DIR = os.path.join("..", CACHE_DIR)
    OUTPUT_DIR = os.path.join("..", OUTPUT_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TXT_OUT = os.path.join(OUTPUT_DIR, f"pcer_results{suffix}.txt")
PNG_OUT = os.path.join(OUTPUT_DIR, f"pcer_results{suffix}.png")


def main():
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    log("=" * 60)
    log("PCER ENSEMBLE PIPELINE — CAXF TF-IDF")
    log("Per-Class Expert Routing (Novel Algorithm)")
    log("Expert assignment via per-class precision on training data")
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
    # Load cached embeddings
    # ---------------------------------------------------------------
    cache_train_emb = resolve_cache_path(CACHE_DIR, "X_train_embed", suffix, "npy")
    cache_test_emb  = resolve_cache_path(CACHE_DIR, "X_test_embed",  suffix, "npy")

    if os.path.exists(cache_train_emb) and os.path.exists(cache_test_emb):
        log("[CACHE] Loading cached TF-IDF embeddings...")
        X_train_embed = load_embedding(cache_train_emb)
        X_test_embed  = load_embedding(cache_test_emb)
    else:
        log("ERROR: Cached embeddings not found. Run run_lccde_caxf_tfidf.py first.")
        return

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
    # Load / train base models
    # ---------------------------------------------------------------
    cache_lgbm = resolve_cache_path(CACHE_DIR, "lgbm", suffix, "pkl")
    cache_xgb  = resolve_cache_path(CACHE_DIR, "xgb",  suffix, "pkl")
    cache_cat  = resolve_cache_path(CACHE_DIR, "cat",  suffix, "pkl")

    if os.path.exists(cache_lgbm) and os.path.exists(cache_xgb) and os.path.exists(cache_cat):
        log("[CACHE] Loading trained base models...")
        lgbm = joblib.load(cache_lgbm)
        xgb  = joblib.load(cache_xgb)
        cat  = joblib.load(cache_cat)
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
        cat = CatBoostClassifier(
            loss_function="MultiClass", iterations=100, learning_rate=0.1,
            depth=4, class_weights=class_weights, random_seed=SEED,
            thread_count=4, task_type="CPU", verbose=False
        )
        cat.fit(X_train_orig, y_train_orig)
        joblib.dump(cat, cache_cat)

    # ---------------------------------------------------------------
    # PCER: compute expert table from training precision
    # ---------------------------------------------------------------
    log("=" * 60)
    log("Computing PCER expert table from training set...")
    log("=" * 60)

    pcer = PCER(
        models=[lgbm, xgb, cat],
        n_classes=4,
        confidence_threshold=0.5
    )
    pcer.fit(X_train_embed, y_train)

    class_names  = le.classes_
    model_names  = ["LightGBM", "XGBoost", "CatBoost"]

    log("Per-class precision matrix (Laplace-smoothed):")
    header = f"  {'Class':<20}" + "".join(f"{n:<12}" for n in model_names) + "  Expert"
    log(header)
    for c_idx, c_name in enumerate(class_names):
        row = f"  {c_name:<20}"
        for k in range(len(model_names)):
            row += f"{pcer.class_precision[c_idx, k]:.4f}      "
        expert_name = model_names[pcer.expert_table[c_idx]]
        row += f"  -> {expert_name}"
        log(row)
    log()

    # ---------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------
    log("=" * 60)
    log("Evaluating PCER Ensemble...")
    log("=" * 60)

    t0 = time.time()
    final_preds = pcer.predict(X_test_embed)
    inf_time = round(time.time() - t0, 4)
    log(f"PCER Inference time: {inf_time} seconds\n")

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
    plt.title("Confusion Matrix — PCER Ensemble (CAXF TF-IDF)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PNG_OUT)
    plt.close()
    print(f"Confusion Matrix plot saved to: {PNG_OUT}")


if __name__ == "__main__":
    main()
