import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt

from caxf.caxf_extractor import CAXFExtractor
from xgboost import XGBClassifier


def main():

    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    df = pd.read_csv("data/processed/Final_XSS_4class_dataset.csv")
    df = df.drop_duplicates()   # ✅ FIXED

    X = df["Sentence"].astype(str)
    y = df["Final_Label"]

    # ---------------------------------------------------
    # Label Encoding
    # ---------------------------------------------------
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Total samples:", len(df))
    print("Class distribution (original):")
    print(y.value_counts())
    print()

    # ---------------------------------------------------
    # Train-Test Split (70:30)
    # ---------------------------------------------------
    print("=" * 60)
    print("Splitting dataset (70:30)...")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,                 # ✅ Use encoded labels
        test_size=0.3,
        random_state=42,
        stratify=y_encoded         # ✅ Stratify properly
    )

    print("Train size:", len(X_train))
    print("Test size :", len(X_test))
    print()

    # Overlap check
    overlap = set(X_train).intersection(set(X_test))
    print("Overlap Size:", len(overlap))
    print()

    # ---------------------------------------------------
    # CAXF Feature Extraction
    # ---------------------------------------------------
    print("=" * 60)
    print("Running CAXF Feature Extraction...")
    print("=" * 60)

    caxf = CAXFExtractor()

    start = time.time()
    caxf.fit(X_train)

    X_train_embed = caxf.transform(X_train)
    X_test_embed = caxf.transform(X_test)
    end = time.time()

    print("CAXF completed.")
    print("Embedding shape (train):", X_train_embed.shape)
    print("Embedding shape (test) :", X_test_embed.shape)
    print("CAXF time:", round(end - start, 2), "seconds")
    print()

    # ---------------------------------------------------
    # SMOTE (ONLY on training set)
    # ---------------------------------------------------
    print("=" * 60)
    print("Applying SMOTE...")
    print("=" * 60)

    smote = SMOTE(random_state=42)

    start = time.time()
    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train_embed,
        y_train
    )
    end = time.time()

    print("SMOTE completed.")
    print("Before SMOTE:")
    print(pd.Series(y_train).value_counts())
    print()
    print("After SMOTE:")
    print(pd.Series(y_train_balanced).value_counts())
    print("SMOTE time:", round(end - start, 2), "seconds")
    print()

    # ---------------------------------------------------
    # XGBoost Model
    # ---------------------------------------------------
    print("=" * 60)
    print("Training XGBoost...")
    print("=" * 60)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_encoded)),
        eval_metric="mlogloss",
        random_state=42,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        tree_method="hist"
    )

    start = time.time()
    model.fit(X_train_balanced, y_train_balanced)  # ✅ Balanced data
    end = time.time()

    print("Training time:", round(end - start, 2), "seconds")
    print()

    # ---------------------------------------------------
    # Evaluation
    # ---------------------------------------------------
    print("=" * 60)
    print("Evaluating model on TEST set...")
    print("=" * 60)

    start = time.time()
    y_pred_encoded = model.predict(X_test_embed)
    end = time.time()

    print("Inference time:", round(end - start, 2), "seconds")
    print()

    # Decode labels for readable output
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    print("Accuracy:", round(accuracy_score(y_test_labels, y_pred_labels), 4))
    print()

    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred_labels))
    print()
    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\nPaper Format Metrics:")
    print("Precision:", round(precision,4))
    print("Recall:", round(recall,4))
    print("Average F1:", round(f1,4))

    # ---------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------
    cm = confusion_matrix(y_test_labels, y_pred_labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix - XGBoost (CAXF)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()