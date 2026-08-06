import time
import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

from src.ensemble.lccde import LCCDE as ProposedMVECT
from src.ensemble.strict_lccde import StrictLCCDE

DATA_PATH = "data/processed/Final_XSS_4class_dataset.csv"
OUTPUT_DIR = "results"
TXT_OUT = os.path.join(OUTPUT_DIR, "ensemble_ablation_study.txt")

def main():
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("="*75)
    log("ENSEMBLE ABLATION STUDY: STRICT LCCDE (ALG 2) VS. PROPOSED MVE-CT")
    log("="*75)
    log()

    df = pd.read_csv(DATA_PATH).drop_duplicates()
    le = LabelEncoder()
    y_encoded = le.fit_transform(df["Final_Label"])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["Sentence"], y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
    )

    combos = [
        ("CAXF + TF-IDF", "results/cache/tfidf"),
        ("CAXF + CharCNN", "results/cache/charcnn"),
        ("CAXF + Sentence Embedding", "results/cache/sentence_embedding")
    ]

    summary_rows = []

    for name, cache_dir in combos:
        log("="*75)
        log(f"EVALUATING FEATURE SET: {name}")
        log("="*75)

        lgbm = joblib.load(f"{cache_dir}/lgbm.pkl")
        xgb = joblib.load(f"{cache_dir}/xgb.pkl")
        cat = joblib.load(f"{cache_dir}/cat.pkl")

        X_train = np.load(f"{cache_dir}/X_train_embed.npy").astype(np.float32)
        X_test = np.load(f"{cache_dir}/X_test_embed.npy").astype(np.float32)
        y_test_labels = le.inverse_transform(y_test)

        # 1. Proposed MVE-CT
        start_mve = time.time()
        mve_ens = ProposedMVECT(lgbm, xgb, cat)
        mve_preds = mve_ens.predict(X_test)
        mve_time = round(time.time() - start_mve, 4)
        mve_labels = le.inverse_transform(mve_preds.astype(int))

        mve_acc = accuracy_score(y_test_labels, mve_labels)
        mve_prec = precision_score(y_test_labels, mve_labels, average="macro")
        mve_rec = recall_score(y_test_labels, mve_labels, average="macro")
        mve_f1 = f1_score(y_test_labels, mve_labels, average="macro")

        # DOM F1 for MVE
        f1s_mve = f1_score(y_test_labels, mve_labels, average=None, labels=le.classes_)
        dom_idx = list(le.classes_).index("DOM-based XSS")
        mve_dom_f1 = f1s_mve[dom_idx]

        # 2. Strict LCCDE (Algorithm 2)
        log("Computing Out-Of-Fold Class Leaders for Strict LCCDE...")
        strict_ens = StrictLCCDE(lgbm, xgb, cat)
        leader_map, f1_matrix = strict_ens.fit_leaders_cv(X_train, np.array(y_train))
        log(f"Leader Map (0=LGBM, 1=XGB, 2=Cat): {leader_map}")

        start_strict = time.time()
        strict_preds = strict_ens.predict(X_test)
        strict_time = round(time.time() - start_strict, 4)
        strict_labels = le.inverse_transform(strict_preds.astype(int))

        strict_acc = accuracy_score(y_test_labels, strict_labels)
        strict_prec = precision_score(y_test_labels, strict_labels, average="macro")
        strict_rec = recall_score(y_test_labels, strict_labels, average="macro")
        strict_f1 = f1_score(y_test_labels, strict_labels, average="macro")

        f1s_strict = f1_score(y_test_labels, strict_labels, average=None, labels=le.classes_)
        strict_dom_f1 = f1s_strict[dom_idx]

        delta_f1 = (mve_f1 - strict_f1) * 100

        winner = "Proposed MVE-CT" if mve_f1 > strict_f1 else ("Strict LCCDE" if strict_f1 > mve_f1 else "Tie")

        summary_rows.append({
            "Feature Set": name,
            "Strict LCCDE F1": round(strict_f1, 4),
            "Strict DOM F1": round(strict_dom_f1, 4),
            "Proposed MVE-CT F1": round(mve_f1, 4),
            "Proposed DOM F1": round(mve_dom_f1, 4),
            "Delta F1 (%)": f"{delta_f1:+.2f}%",
            "Winner": winner
        })

        log(f"\nResults for {name}:")
        log(f"  Strict LCCDE F1 : {strict_f1:.4f} (DOM F1: {strict_dom_f1:.4f})")
        log(f"  Proposed MVE-CT F1: {mve_f1:.4f} (DOM F1: {mve_dom_f1:.4f})")
        log(f"  Performance Delta : {delta_f1:+.2f}%")
        log(f"  Winner            : {winner}\n")

    log("="*75)
    log("SUMMARY ABLATION COMPARISON TABLE")
    log("="*75)
    df_summary = pd.DataFrame(summary_rows)
    log(str(df_summary.to_string(index=False)))
    log()

    with open(TXT_OUT, "w") as f:
        f.write("\n".join(log_lines))

    print(f"\nAblation study results saved to: {TXT_OUT}")

if __name__ == "__main__":
    main()
