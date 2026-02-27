import time
import pandas as pd
from caxf.caxf_extractor import CAXFExtractor
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


print("XSS Detection Pipeline Started")
print("=" * 60)

pipeline_start = time.time()

# -----------------------------
# 1. Load Dataset
# -----------------------------
stage_start = time.time()
print("Step 1: Loading dataset")

df = pd.read_csv("data/processed/Final_XSS_4class_dataset.csv")

print("Dataset loaded")
print("Dataset shape:", df.shape)
print("Class distribution:")
print(df["Final_Label"].value_counts())
print("Time taken:", round(time.time() - stage_start, 2), "seconds\n")

payloads = df["Sentence"].astype(str)
labels = df["Final_Label"]

# -----------------------------
# 2. CAXF Feature Extraction
# -----------------------------
stage_start = time.time()
print("Step 2: Initializing CAXF extractor")

caxf = CAXFExtractor()

print("Extracting CAXF features (this may take time)...")
X = caxf.fit_transform(payloads)

print("Feature extraction completed")
print("Feature matrix shape:", X.shape)
print("Time taken:", round(time.time() - stage_start, 2), "seconds\n")

# -----------------------------
# 3. Encode Labels
# -----------------------------
stage_start = time.time()
print("Step 3: Encoding class labels")

le = LabelEncoder()
y_encoded = le.fit_transform(labels)

print("Label mapping:")
for i, cls in enumerate(le.classes_):
    print(f"{cls} -> {i}")

print("Time taken:", round(time.time() - stage_start, 2), "seconds\n")

# -----------------------------
# 4. Train-Test Split
# -----------------------------
stage_start = time.time()
print("Step 4: Splitting dataset (70:30)")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.3,
    stratify=y_encoded,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Training class distribution:")
print(pd.Series(y_train).value_counts())
print("Time taken:", round(time.time() - stage_start, 2), "seconds\n")

# -----------------------------
# 5. Apply SMOTE
# -----------------------------
stage_start = time.time()
print("Step 5: Applying SMOTE (this may take time)")

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("SMOTE completed")
print("Resampled training shape:", X_train_sm.shape)
print("Class distribution after SMOTE:")
print(pd.Series(y_train_sm).value_counts())
print("Time taken:", round(time.time() - stage_start, 2), "seconds\n")

# -----------------------------
# 6. Train Base Models
# -----------------------------

# LightGBM
stage_start = time.time()
print("Step 6A: Training LightGBM")

lgb = LGBMClassifier(random_state=42)
lgb.fit(X_train_sm, y_train_sm)

print("LightGBM training completed")
print("Time taken:", round(time.time() - stage_start, 2), "seconds\n")

# XGBoost
stage_start = time.time()
print("Step 6B: Training XGBoost")

xgb = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss'
)

xgb.fit(X_train_sm, y_train_sm)

print("XGBoost training completed")
print("Time taken:", round(time.time() - stage_start, 2), "seconds\n")

# CatBoost
stage_start = time.time()
print("Step 6C: Training CatBoost")

cat = CatBoostClassifier(
    verbose=100,   # shows iteration progress
    random_state=42
)

cat.fit(X_train_sm, y_train_sm)

print("CatBoost training completed")
print("Time taken:", round(time.time() - stage_start, 2), "seconds\n")

# -----------------------------
# 7. Evaluation
# -----------------------------
print("Step 7: Evaluating models\n")

def evaluate_model(model, name):
    print(f"Evaluating {name}")
    eval_start = time.time()

    predictions = model.predict(X_test)

    print(classification_report(
        le.inverse_transform(y_test),
        le.inverse_transform(predictions)
    ))

    print("Evaluation time:", round(time.time() - eval_start, 2), "seconds\n")

evaluate_model(lgb, "LightGBM")
evaluate_model(xgb, "XGBoost")
evaluate_model(cat, "CatBoost")

# -----------------------------
# 8. Confusion Matrix
# -----------------------------
print("Step 8: Generating confusion matrix (LightGBM)")

cm = confusion_matrix(y_test, lgb.predict(X_test))

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title("Confusion Matrix - LightGBM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("Total pipeline time:", round(time.time() - pipeline_start, 2), "seconds")
print("Pipeline completed successfully")