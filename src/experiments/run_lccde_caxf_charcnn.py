import time
import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure both project root (for src.caxf.*) and src/ (for config) are in sys.path
_here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "../..")))  # project root
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))    # src/

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.caxf.caxf_extractor_charcnn import CAXFExtractor
from src.ensemble.lccde import LCCDE

# ===============================
# PATHS & CONFIG
# ===============================
from config import SEED

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

TXT_OUT = os.path.join(OUTPUT_DIR, f"lccde_results{suffix}.txt")
PNG_OUT = os.path.join(OUTPUT_DIR, f"lccde_results{suffix}.png")

def main():
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("="*60)
    log("LCCDE ENSEMBLE PIPELINE — CAXF CHAR-CNN")
    log("="*60)
    log("Loading dataset...")
    log("="*60)

    df = pd.read_csv(DATA_PATH).drop_duplicates()
    X = df["Sentence"].astype(str)
    y = df["Final_Label"]

    log(f"Total samples: {len(df)}")
    log("Class distribution:")
    log(str(y.value_counts()))
    log()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    class CatBoostIntAdapter:
        def __init__(self, model, label_encoder):
            self._model = model
            self._le = label_encoder

        def predict(self, X):
            str_preds = self._model.predict(X)
            flat = [p[0] if hasattr(p, "__len__") and not isinstance(p, str) else p
                    for p in str_preds]
            if len(flat) > 0 and isinstance(flat[0], (int, np.integer)):
                return np.array(flat, dtype=int)
            return self._le.transform(flat)

        def predict_proba(self, X):
            return self._model.predict_proba(X)

    log("="*60)
    log("Splitting dataset (70:30)...")
    log("="*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.30,
        random_state=SEED,
        stratify=y_encoded
    )

    log(f"Train size: {len(X_train)}")
    log(f"Test size : {len(X_test)}")
    overlap = len(set(X_train).intersection(set(X_test)))
    log(f"Overlap Size: {overlap}")
    log()

    # ===============================
    # FEATURE EXTRACTION
    # ===============================
    cache_train_emb = f"{CACHE_DIR}/X_train_embed{suffix}.npy"
    cache_test_emb = f"{CACHE_DIR}/X_test_embed{suffix}.npy"

    if os.path.exists(cache_train_emb) and os.path.exists(cache_test_emb):
        log("[CACHE] Loading cached CharCNN embeddings...")
        X_train_embed = np.load(cache_train_emb)
        X_test_embed = np.load(cache_test_emb)
        caxf_time = 0.0
    else:
        log("="*60)
        log("Running CAXF Feature Extraction (CharCNN)...")
        log("="*60)
        start = time.time()
        caxf = CAXFExtractor()
        caxf.fit(X_train)
        X_train_embed = caxf.transform(X_train)
        X_test_embed = caxf.transform(X_test)
        
        # Convert sparse to array if needed for caching
        if hasattr(X_train_embed, "toarray"):
            X_train_embed = X_train_embed.toarray()
            X_test_embed = X_test_embed.toarray()

        np.save(cache_train_emb, X_train_embed)
        np.save(cache_test_emb, X_test_embed)
        caxf_time = round(time.time() - start, 2)
        log("Saved embeddings checkpoint.")

    X_train_embed = X_train_embed.astype(np.float32)
    X_test_embed = X_test_embed.astype(np.float32)

    log(f"Embedding shape (train): {X_train_embed.shape}")
    log(f"Embedding shape (test) : {X_test_embed.shape}")
    log(f"CAXF time: {caxf_time} seconds\n")

    # ===============================
    # SMOTE FOR LGBM & XGB
    # ===============================
    log("="*60)
    log("Applying SMOTE...")
    log("="*60)
    log("Before SMOTE:")
    log(str(pd.Series(y_train).value_counts()))

    start_smote = time.time()
    smote = SMOTE(random_state=SEED)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_embed, y_train)
    smote_time = round(time.time() - start_smote, 2)

    log("\nAfter SMOTE:")
    log(str(pd.Series(y_train_smote).value_counts()))
    log(f"SMOTE time: {smote_time} seconds\n")

    X_train_orig = X_train_embed
    y_train_orig = y_train

    # ===============================
    # TRAIN BASE MODELS
    # ===============================
    cache_lgbm = os.path.join(CACHE_DIR, f"lgbm{suffix}.pkl")
    if not os.path.exists(cache_lgbm):
        cache_lgbm = os.path.join(CACHE_DIR, "lgbm.pkl")

    cache_xgb = os.path.join(CACHE_DIR, f"xgb{suffix}.pkl")
    if not os.path.exists(cache_xgb):
        cache_xgb = os.path.join(CACHE_DIR, "xgb.pkl")

    cache_cat = os.path.join(CACHE_DIR, f"cat{suffix}.pkl")
    if not os.path.exists(cache_cat):
        cache_cat = os.path.join(CACHE_DIR, "cat.pkl")

    if os.path.exists(cache_lgbm) and os.path.exists(cache_xgb) and os.path.exists(cache_cat):
        log("[CACHE] Loading trained base models...")
        lgbm = joblib.load(cache_lgbm)
        xgb = joblib.load(cache_xgb)
        _cat_raw = joblib.load(cache_cat)
        cat = CatBoostIntAdapter(_cat_raw, le)
        train_time_lgbm, train_time_xgb, train_time_cat = 0.0, 0.0, 0.0
    else:
        # LightGBM
        log("Training LightGBM...")
        lgbm = LGBMClassifier(
            objective="multiclass", num_class=4, n_estimators=100,
            learning_rate=0.1, random_state=SEED, n_jobs=-1
        )
        t0 = time.time()
        lgbm.fit(X_train_smote, y_train_smote)
        train_time_lgbm = round(time.time() - t0, 2)
        joblib.dump(lgbm, cache_lgbm)

        # XGBoost
        log("Training XGBoost...")
        xgb = XGBClassifier(
            objective="multi:softprob", num_class=4, n_estimators=100,
            learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=SEED, n_jobs=-1
        )
        t0 = time.time()
        xgb.fit(X_train_smote, y_train_smote)
        train_time_xgb = round(time.time() - t0, 2)
        joblib.dump(xgb, cache_xgb)

        # CatBoost
        log("Training CatBoost...")
        class_counts = pd.Series(y_train_orig).value_counts()
        total = len(y_train_orig)
        class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
        cat = CatBoostClassifier(
            loss_function="MultiClass", iterations=100, learning_rate=0.1,
            depth=4, class_weights=class_weights, random_seed=SEED, thread_count=4,
            task_type="CPU", verbose=False
        )
        t0 = time.time()
        cat.fit(X_train_orig, y_train_orig)
        train_time_cat = round(time.time() - t0, 2)
        joblib.dump(cat, cache_cat)

    # ===============================
    # LCCDE ENSEMBLE EVALUATION
    # ===============================
    log("="*60)
    log("Evaluating LCCDE Ensemble...")
    log("="*60)

    start_inf = time.time()
    ensemble = LCCDE(lgbm, xgb, cat)
    final_preds = ensemble.predict(X_test_embed)
    inf_time = round(time.time() - start_inf, 4)
    log(f"LCCDE Inference time: {inf_time} seconds\n")

    y_test_labels = le.inverse_transform(y_test)
    final_preds_labels = le.inverse_transform(final_preds.astype(int))

    acc = accuracy_score(y_test_labels, final_preds_labels)
    precision = precision_score(y_test_labels, final_preds_labels, average="macro")
    recall = recall_score(y_test_labels, final_preds_labels, average="macro")
    f1 = f1_score(y_test_labels, final_preds_labels, average="macro")

    log(f"Accuracy: {round(acc, 4)}")
    log("\nClassification Report:")
    log(classification_report(y_test_labels, final_preds_labels))

    log("Paper Format Metrics")
    log(f"Precision: {round(precision, 4)}")
    log(f"Recall: {round(recall, 4)}")
    log(f"Average F1: {round(f1, 4)}")

    # Save output text file
    with open(TXT_OUT, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nResults saved to: {TXT_OUT}")

    # Generate & Save Confusion Matrix Plot
    cm = confusion_matrix(y_test_labels, final_preds_labels, labels=le.classes_)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix - LCCDE (CAXF CharCNN)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PNG_OUT)
    plt.close()
    print(f"Confusion Matrix plot saved to: {PNG_OUT}")

if __name__ == "__main__":
    main()
