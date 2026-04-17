import time
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from caxf.caxf_extractor_tfidf import CAXFExtractor
from src.ensemble.lccde import LCCDE


# ===============================
# PATHS
# ===============================
DATA_PATH = "data/processed/Final_XSS_4class_dataset.csv"
CACHE_DIR = "results/cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ===============================
# LOAD DATA
# ===============================
print("="*60)
print("Loading dataset...")
print("="*60)

df = pd.read_csv(DATA_PATH).drop_duplicates()

X = df["Sentence"].astype(str)
y = df["Final_Label"]

print("Total samples:", len(df))
print("Class distribution:")
print(y.value_counts())
print()


# ===============================
# LABEL ENCODING
# ===============================
le = LabelEncoder()
y_encoded = le.fit_transform(y)


# ===============================
# TRAIN TEST SPLIT
# ===============================
print("="*60)
print("Splitting dataset (70:30)...")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded
)

print("Train size:", len(X_train))
print("Test size :", len(X_test))
print()


# ===============================
# LOAD / COMPUTE CAXF
# ===============================
if os.path.exists(f"{CACHE_DIR}/X_train_embed.npy"):
    print("⚡ Loading cached embeddings...")

    X_train_embed = np.load(f"{CACHE_DIR}/X_train_embed.npy")
    X_test_embed = np.load(f"{CACHE_DIR}/X_test_embed.npy")
    y_train = np.load(f"{CACHE_DIR}/y_train.npy")
    y_test = np.load(f"{CACHE_DIR}/y_test.npy")

else:
    print("="*60)
    print("Running CAXF Feature Extraction...")
    print("="*60)

    start = time.time()

    caxf = CAXFExtractor()
    caxf.fit(X_train)

    X_train_embed = caxf.transform(X_train)
    X_test_embed = caxf.transform(X_test)

    np.save(f"{CACHE_DIR}/X_train_embed.npy", X_train_embed)
    np.save(f"{CACHE_DIR}/X_test_embed.npy", X_test_embed)
    np.save(f"{CACHE_DIR}/y_train.npy", y_train)
    np.save(f"{CACHE_DIR}/y_test.npy", y_test)

    print(" Saved embeddings (checkpoint)")
    print("CAXF time:", round(time.time() - start, 2), "seconds\n")


# ===============================
# FIX DATA TYPE (IMPORTANT)
# ===============================
X_train_embed = X_train_embed.astype(np.float32)
X_test_embed = X_test_embed.astype(np.float32)


# ===============================
# SMOTE (ONLY FOR LGBM + XGB)
# ===============================
print("="*60)
print("Applying SMOTE...")
print("="*60)

print("Before SMOTE:")
print(pd.Series(y_train).value_counts())

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_embed, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())
print()

X_train_orig = X_train_embed
y_train_orig = y_train


# ===============================
# LOAD / TRAIN MODELS
# ===============================
if os.path.exists(f"{CACHE_DIR}/lgbm.pkl"):
    print("⚡ Loading trained models...")

    lgbm = joblib.load(f"{CACHE_DIR}/lgbm.pkl")
    xgb = joblib.load(f"{CACHE_DIR}/xgb.pkl")
    cat = joblib.load(f"{CACHE_DIR}/cat.pkl")

else:
    # -------- LightGBM --------
    print("="*60)
    print("Training LightGBM...")
    print("="*60)

    lgbm = LGBMClassifier(
        objective="multiclass",
        num_class=4,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )

    start = time.time()
    lgbm.fit(X_train_smote, y_train_smote)
    print("LightGBM time:", round(time.time()-start,2), "sec\n")

    joblib.dump(lgbm, f"{CACHE_DIR}/lgbm.pkl")

    # -------- XGBoost --------
    print("="*60)
    print("Training XGBoost...")
    print("="*60)

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    start = time.time()
    xgb.fit(X_train_smote, y_train_smote)
    print("XGBoost time:", round(time.time()-start,2), "sec\n")

    joblib.dump(xgb, f"{CACHE_DIR}/xgb.pkl")

    # -------- CatBoost --------
    print("="*60)
    print("Training CatBoost...")
    print("="*60)

    class_counts = pd.Series(y_train_orig).value_counts()
    total = len(y_train_orig)
    num_classes = len(class_counts)

    class_weights = {
        cls: total / (num_classes * count)
        for cls, count in class_counts.items()
    }

    cat = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=100,
        learning_rate=0.1,
        depth=4,
        class_weights=class_weights,
        random_seed=42,
        thread_count=4,
        task_type="CPU",
        verbose=False
    )

    start = time.time()
    cat.fit(X_train_orig, y_train_orig)
    print("CatBoost time:", round(time.time()-start,2), "sec\n")

    joblib.dump(cat, f"{CACHE_DIR}/cat.pkl")


# ===============================
# LCCDE ENSEMBLE
# ===============================
print("="*60)
print("Running LCCDE Ensemble...")
print("="*60)

start = time.time()
ensemble = LCCDE(lgbm, xgb, cat)
final_preds = ensemble.predict(X_test_embed)
print("LCCDE time:", round(time.time()-start,2), "sec\n")


# ===============================
# EVALUATION
# ===============================
y_test_labels = le.inverse_transform(y_test)
final_preds_labels = le.inverse_transform(final_preds.astype(int))

acc = accuracy_score(y_test_labels, final_preds_labels)
precision = precision_score(y_test_labels, final_preds_labels, average="macro")
recall = recall_score(y_test_labels, final_preds_labels, average="macro")
f1 = f1_score(y_test_labels, final_preds_labels, average="macro")

print("="*60)
print("FINAL RESULTS")
print("="*60)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Macro F1 : {f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test_labels, final_preds_labels))


# ===============================
# SAVE RESULTS
# ===============================
result_path = "results/caxf_tfidf_results/lccde_results.txt"

with open(result_path, "w") as f:
    f.write("LCCDE ENSEMBLE RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"Macro F1: {f1:.4f}\n\n")
    f.write(classification_report(y_test_labels, final_preds_labels))

print("\n Results saved to:", result_path)