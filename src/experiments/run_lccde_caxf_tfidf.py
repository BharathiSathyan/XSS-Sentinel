import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.caxf.caxf_extractor import CAXFExtractor
from src.ensemble.lccde import LCCDE


# ===============================
# DATASET PATH
# ===============================

DATA_PATH = "data/processed/Final_XSS_4class_dataset.csv"


# ===============================
# LOAD DATA
# ===============================

print("="*60)
print("Loading dataset...")
print("="*60)

df = pd.read_csv(DATA_PATH)

X = df["Sentence"].astype(str)
y = df["Final_Label"]

print("Total samples:", len(df))
print("Class distribution:")
print(y.value_counts())
print()


# ===============================
# TRAIN TEST SPLIT
# ===============================

print("="*60)
print("Splitting dataset (70:30)...")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Train size:", len(X_train))
print("Test size :", len(X_test))
print()


# ===============================
# FEATURE EXTRACTION
# ===============================

print("="*60)
print("Running CAXF Feature Extraction...")
print("="*60)

start = time.time()

caxf = CAXFExtractor()

X_train_embed = caxf.fit_transform(X_train)
X_test_embed = caxf.transform(X_test)

print("Embedding shape (train):", X_train_embed.shape)
print("Embedding shape (test) :", X_test_embed.shape)

print("CAXF time:", round(time.time() - start,2), "seconds")
print()


# ===============================
# APPLY SMOTE
# ===============================

print("="*60)
print("Applying SMOTE...")
print("="*60)

start = time.time()

print("Before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_embed, y_train = smote.fit_resample(X_train_embed, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train).value_counts())

print("SMOTE time:", round(time.time()-start,2), "seconds")
print()


# ===============================
# TRAIN LIGHTGBM
# ===============================

print("="*60)
print("Training LightGBM...")
print("="*60)

start = time.time()

lgbm = LGBMClassifier(
    objective="multiclass",
    num_class=4,
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

lgbm.fit(X_train_embed, y_train)

print("LightGBM training time:", round(time.time()-start,2), "seconds")
print()


# ===============================
# TRAIN XGBOOST
# ===============================

print("="*60)
print("Training XGBoost...")
print("="*60)

start = time.time()

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

xgb.fit(X_train_embed, y_train)

print("XGBoost training time:", round(time.time()-start,2), "seconds")
print()


# ===============================
# TRAIN CATBOOST
# ===============================

print("="*60)
print("Training CatBoost...")
print("="*60)

start = time.time()

cat = CatBoostClassifier(
    loss_function="MultiClass",
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=False
)

cat.fit(X_train_embed, y_train)

print("CatBoost training time:", round(time.time()-start,2), "seconds")
print()


# ===============================
# LCCDE ENSEMBLE
# ===============================

print("="*60)
print("Running LCCDE Ensemble...")
print("="*60)

start = time.time()

ensemble = LCCDE(lgbm, xgb, cat)

final_preds = ensemble.predict(X_test_embed)

print("LCCDE inference time:", round(time.time()-start,2), "seconds")
print()


# ===============================
# EVALUATION
# ===============================

print("="*60)
print("Evaluating Ensemble...")
print("="*60)

acc = accuracy_score(y_test, final_preds)

print("Accuracy:", acc)
print()

print("Classification Report:")
print(classification_report(y_test, final_preds))


# ===============================
# SAVE RESULTS
# ===============================

result_path = "results/caxf_tfidf_results/lccde_results.txt"

with open(result_path, "w") as f:

    f.write("LCCDE ENSEMBLE RESULTS\n")
    f.write("="*40 + "\n")
    f.write(f"Accuracy: {acc}\n\n")
    f.write(classification_report(y_test, final_preds))

print("\nResults saved to:", result_path)