import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier
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
    log_file = os.path.join(results_dir, f"train_lgbm_unbalanced_smote{suffix}.txt")
    sys.stdout = Logger(log_file)

    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    dataset_path = "data/processed/Final_XSS_4class_dataset.csv"
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join("..", dataset_path)
    df = pd.read_csv(dataset_path)
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
    print("=" * 60)
    print("Splitting dataset (70:30)...")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.3,
        random_state=SEED,
        stratify=y_encoded
    )

    print("Train size:", len(X_train))
    print("Test size :", len(X_test))
    print()

    overlap = set(X_train).intersection(set(X_test))
    print("Overlap Size:", len(overlap))
    print()

    # Feature Extraction
    print("=" * 60)
    print("Running CAXF Feature Extraction...")
    print("=" * 60)

    caxf = CAXFExtractor()

    start = time.time()

    caxf.fit(X_train)
    X_train_embed = caxf.transform(X_train)
    X_test_embed = caxf.transform(X_test)

    print("Embedding shape (train):", X_train_embed.shape)
    print("Embedding shape (test):", X_test_embed.shape)
    print("CAXF time:", round(time.time() - start, 2), "seconds")
    print()

    # SMOTE
    print("=" * 60)
    print("Applying SMOTE...")
    print("=" * 60)

    smote = SMOTE(random_state=SEED)

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

    # LightGBM Model
    print("=" * 60)
    print("Training LightGBM...")
    print("=" * 60)

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(np.unique(y_encoded)),
        random_state=SEED,
        n_estimators=200,
        learning_rate=0.1,
        n_jobs=-1
    )

    start = time.time()
    model.fit(X_train_bal, y_train_bal)
    print("Training time:", round(time.time() - start, 2), "seconds")
    print()

    # Evaluation
    print("=" * 60)
    print("Evaluating model...")
    print("=" * 60)

    start = time.time()
    y_pred = model.predict(X_test_embed)
    print("Inference time:", round(time.time() - start, 2), "seconds")
    print()

    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)

    print("Accuracy:", round(accuracy_score(y_test_labels, y_pred_labels), 4))
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
    
    plot_file = os.path.join(results_dir, f"train_lgbm_unbalanced_smote{suffix}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Confusion Matrix plot saved to: {plot_file}")

    # ---------------------------------------------------
    # Save Trained Model
    # ---------------------------------------------------
    model_file = os.path.join(results_dir, f"lgbm_model{suffix}.pkl")
    joblib.dump(model, model_file)
    print(f"Model saved to: {model_file}")


if __name__ == "__main__":
    main()