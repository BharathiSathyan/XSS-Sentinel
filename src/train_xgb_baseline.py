import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt

from src.caxf.caxf_extractor import CAXFExtractor
from xgboost import XGBClassifier


def main():

    print("="*60)
    print("Loading dataset...")
    print("="*60)

    df = pd.read_csv("data/processed/Final_XSS_4class_dataset.csv")
    df = df.drop_duplicates()

    X = df["Sentence"].astype(str)
    y = df["Final_Label"]

    print("Total samples:", len(df))
    print("Class distribution:")
    print(y.value_counts())
    print()

    # Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train Test Split
    print("="*60)
    print("Splitting dataset...")
    print("="*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    print("Train size:", len(X_train))
    print("Test size :", len(X_test))
    print()

    overlap = set(X_train).intersection(set(X_test))
    print("Overlap Size:", len(overlap))
    print()

    # Feature Extraction
    print("="*60)
    print("Running CAXF Feature Extraction......")
    print("="*60)

    caxf = CAXFExtractor()

    start = time.time()
    caxf.fit(X_train)

    X_train_embed = caxf.transform(X_train)
    X_test_embed = caxf.transform(X_test)

    print("Embedding shape (train):", X_train_embed.shape)
    print("Embedding shape (test):", X_test_embed.shape)
    print("CAXF time:", round(time.time()-start,2))
    print()

    # SMOTE
    print("=" * 60)
    print("Applying SMOTE...")
    print("=" * 60)

    smote = SMOTE(random_state=42)

    start = time.time()

    X_train_bal, y_train_bal = smote.fit_resample(
        X_train_embed,
        y_train
    )

    print("Before SMOTE:")
    print(pd.Series(y_train).value_counts())
    print()

    print("After SMOTE:")
    print(pd.Series(y_train_bal).value_counts())

    print("SMOTE time:", round(time.time() - start, 2), "seconds")
    print()

    # XGBoost Model
    print("=" * 60)
    print("Training CatBoost...")
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
        tree_method="hist",
        n_jobs=-1
    )

    start = time.time()
    model.fit(X_train_bal, y_train_bal)
    print("Training time:", round(time.time()-start,2))
    print()

    print("="*60)
    print("Evaluating model...")
    print("="*60)

    start = time.time()
    y_pred = model.predict(X_test_embed)
    print("Inference time:", round(time.time()-start,2))
    print()

    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)

    print("Accuracy:", accuracy_score(y_test_labels, y_pred_labels))
    print()

    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred_labels))
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\nPaper Format Metrics")
    print("Precision:", round(precision,4))
    print("Recall:", round(recall,4))
    print("Average F1:", round(f1,4))

    cm = confusion_matrix(y_test_labels, y_pred_labels)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title("Confusion Matrix - LightGBM")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()