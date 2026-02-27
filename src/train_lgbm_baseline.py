import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt

from src.caxf.caxf_extractor import CAXFExtractor


def main():

    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    df = pd.read_csv("data/processed/Final_XSS_4class_dataset.csv")
    
    X = df["Sentence"].astype(str)
    y = df["Final_Label"]

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
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    print("Train size:", len(X_train))
    print("Test size :", len(X_test))
    print()

    #Overlab test
    train_set=set(X_train)
    test_set=set(X_test)

    overlap = train_set.intersection(test_set)
    print("Overlap Size ", len(overlap))
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
    ""
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
    print(y_train.value_counts())
    print()
    print("After SMOTE:")
    print(pd.Series(y_train_balanced).value_counts())
    print("SMOTE time:", round(end - start, 2), "seconds")
    print()
    
    # ---------------------------------------------------
    # LightGBM Model (FAST BASELINE)
    # ---------------------------------------------------
    print("=" * 60)
    print("Training LightGBM...")
    print("=" * 60)

    model = LGBMClassifier(
        objective="multiclass",
        num_class=4,
        random_state=42,
        n_estimators=200,       # moderate
        learning_rate=0.1,
        n_jobs=-1               # use all CPU cores
    )

    start = time.time()
    model.fit(X_train_embed, y_train)
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
    y_pred = model.predict(X_test_embed)
    end = time.time()

    print("Inference time:", round(end - start, 2), "seconds")
    print()

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print()

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print()

    # ---------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - LightGBM (CAXF)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()