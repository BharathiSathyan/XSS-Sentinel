"""
label_rule_audit.py
===================
Context for interpreting the SHAP results: how much of the 4-class label set in
data/processed/Final_XSS_4class_dataset.csv is reproduced by the keyword rules in
src/data_preprocessing/recategorize_dataset_rulebased.py ?

If the rules reproduce (almost) all labels, then a model whose top SHAP features are the
same keywords/patterns has largely recovered the labelling rule, which is relevant when
claiming that it learned XSS *semantics*.

Run from the project root:
    venv\\Scripts\\python.exe src\\explainability\\label_rule_audit.py
"""
import os
import sys

import pandas as pd

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(_root, "src", "data_preprocessing"))
from recategorize_dataset_rulebased import categorize_xss  # noqa: E402

df = pd.read_csv(os.path.join(_root, "data/processed/Final_XSS_4class_dataset.csv")).drop_duplicates()
binary = (df["Final_Label"] != "Normal").astype(int)        # 0 = benign, 1 = any XSS
rule = [categorize_xss(s, b) for s, b in zip(df["Sentence"].astype(str), binary)]
df["rule_label"] = rule

lines = [f"n = {len(df)}",
         f"overall agreement with rule-based relabeller: {(df.rule_label == df.Final_Label).mean():.4f}",
         "", "confusion (rows = Final_Label, cols = rule-based label):",
         pd.crosstab(df.Final_Label, df.rule_label).to_string(), "",
         "per-class agreement:"]
for c, g in df.groupby("Final_Label"):
    lines.append(f"  {c:14s}: {(g.rule_label == c).mean():.4f}  (n={len(g)})")
out = os.path.join(_root, "results", "explainability", "label_rule_audit.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
