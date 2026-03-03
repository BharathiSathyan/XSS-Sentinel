import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt

from caxf.caxf_extractor import CAXFExtractor


def main():

    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    df = pd.read_csv("data/processed/Final_XSS_4class_dataset.csv")
    df = df.drop_duplicates()  #Proper duplicate removal

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

    print("Embedding shape (train):", X_train_embed.shape)
    print("Embedding shape (test) :", X_test_embed.shape)
    print("CAXF time:", round(end - start, 2), "seconds")
    print()

    # ---------------------------------------------------
    # Compute Class Weights (Research-Grade)
    # ---------------------------------------------------
    print("=" * 60)
    print("Computing Class Weights...")
    print("=" * 60)

    class_counts = y_train.value_counts()
    total_samples = len(y_train)
    num_classes = len(class_counts)

    class_weights = {
        cls: total_samples / (num_classes * count)
        for cls, count in class_counts.items()
    }

    print("Class Weights:")
    for k, v in class_weights.items():
        print(f"{k}: {round(v, 3)}")
    print()

    # ---------------------------------------------------
    # CatBoost Model (Recommended Configuration)
    # ---------------------------------------------------
    print("=" * 60)
    print("Training CatBoost (Research Mode)...")
    print("=" * 60)

    model = CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=42,
        iterations=300,
        learning_rate=0.05,
        depth=6,
        class_weights=class_weights,
        verbose=False
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
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_
    )
    plt.title("Confusion Matrix - CatBoost (CAXF)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()