import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Uncomment ONE extractor at a time:
# from caxf.caxf_extractor_tfidf import CAXFExtractor
from caxf.caxf_extractor_sentence_embedding import CAXFExtractor
# from caxf.caxf_extractor_charcnn import CAXFExtractor
import joblib
import sys
import os
from config import SEED

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    extractor_module = CAXFExtractor.__module__
    if "sentence_embedding" in extractor_module:
        results_dir = "results/caxf_sentence_embedding_results"
    elif "charcnn" in extractor_module:
        results_dir = "results/caxf_char_cnn_results"
    else:
        results_dir = "results/caxf_tfidf_results"

    if not os.path.exists("results") and os.path.exists("../results"):
        results_dir = os.path.join("..", results_dir)

    suffix = f"_seed_{SEED}"
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, f"train_catboost_unbalanced_smote{suffix}.txt")
    sys.stdout = Logger(log_file)

    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    dataset_path = "data/processed/Final_XSS_4class_dataset.csv"
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join("..", dataset_path)
    df = pd.read_csv(dataset_path)
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
        random_state=SEED,
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
        random_seed=SEED,
        iterations=300,
        learning_rate=0.05,
        depth=5,
        class_weights=class_weights,
        task_type="CPU",
        thread_count=4,
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

    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\nPaper Format Metrics")
    print("Precision:", round(precision,4))
    print("Recall:", round(recall,4))
    print("Average F1:", round(f1,4))

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
    
    plot_file = os.path.join(results_dir, f"train_catboost_unbalanced_smote{suffix}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Confusion Matrix plot saved to: {plot_file}")

    # ---------------------------------------------------
    # Save Trained Model
    # ---------------------------------------------------
    model_file = os.path.join(results_dir, f"catboost_model{suffix}.pkl")
    joblib.dump(model, model_file)
    print(f"Model saved to: {model_file}")


if __name__ == "__main__":
    main()