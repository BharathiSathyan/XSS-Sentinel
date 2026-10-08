"""
feature_names.py
================
Recovers human-readable names (and CAXF component group) for every column of the
cached CAXF+TF-IDF feature matrix, so SHAP values can be reported as
"which payload property drove the decision" instead of "feature #4172".

Column order is dictated by ``CAXFExtractor.transform`` (src/caxf/caxf_extractor_tfidf.py):

    [ HTML tags | JS events | URL | special chars | keywords | char TF-IDF n-grams ]

The extractors are re-fitted on the *training* payloads only (identical to the
original experiments), which is deterministic, so the vocabularies are identical
to the ones used to build the cached matrices.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

_here = os.path.abspath(os.path.dirname(__file__))
_src = os.path.abspath(os.path.join(_here, ".."))
_root = os.path.abspath(os.path.join(_here, "../.."))
for _p in (_root, _src):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.caxf.caxf_extractor_tfidf import CAXFExtractor  # noqa: E402
from cache_utils import load_embedding  # noqa: E402

GROUPS = ["HTML tags", "JS events", "URL patterns",
          "Special chars", "XSS keywords", "Char n-gram TF-IDF"]

DATA_PATH = os.path.join(_root, "data/processed/Final_XSS_4class_dataset.csv")
CACHE_DIR = os.path.join(_root, "results/cache/tfidf")


def load_split(seed=42):
    """Reproduces the exact 70:30 stratified split used by every experiment."""
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    X = df["Sentence"].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(df["Final_Label"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y)
    return X_tr, X_te, y_tr, y_te, le


def _vocab(vectorizer):
    return list(vectorizer.get_feature_names_out())


def build_feature_names(X_train_payloads):
    """Fit CAXF on the train payloads; return (names, groups) per column."""
    ext = CAXFExtractor().fit(X_train_payloads)
    parts = [
        ("HTML tags",          [f"<{t}>" for t in _vocab(ext.html_extractor.vectorizer)]),
        ("JS events",          [f"{e}=" for e in _vocab(ext.js_extractor.vectorizer)]),
        ("URL patterns",       ["url_pattern_count"]),
        ("Special chars",      [f"char[{c}]" for c in _vocab(ext.sc_extractor.vectorizer)]),
        ("XSS keywords",       [f"kw:{k}" for k in _vocab(ext.kw_extractor.vectorizer)]),
        ("Char n-gram TF-IDF", [f"ngram:'{g}'" for g in _vocab(ext.tfidf_extractor.vectorizer)]),
    ]
    names, groups = [], []
    for grp, cols in parts:
        names += cols
        groups += [grp] * len(cols)
    return np.array(names, dtype=object), np.array(groups, dtype=object), ext


def load_cached_features(seed=42):
    """Cached train/test CAXF+TF-IDF matrices (dense float32)."""
    tr = os.path.join(CACHE_DIR, "X_train_embed.npy")
    te = os.path.join(CACHE_DIR, "X_test_embed.npy")
    return load_embedding(tr), load_embedding(te)


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, le = load_split()
    names, groups, ext = build_feature_names(X_tr)
    Xtr_c, Xte_c = load_cached_features()
    print("classes        :", list(le.classes_))
    print("names          :", len(names))
    print("cached train   :", Xtr_c.shape, " cached test:", Xte_c.shape)
    print("group sizes    :", {g: int((groups == g).sum()) for g in GROUPS})

    # Alignment check: re-transform the TRAIN payloads and compare to the cache.
    Xtr_new = ext.transform(X_tr).toarray().astype(np.float32)
    print("train re-extract identical to cache:",
          Xtr_new.shape == Xtr_c.shape and bool(np.allclose(Xtr_new, Xtr_c, atol=1e-5)))

    # Special-char transform() re-fits its DictVectorizer on the test payloads
    # (see src/caxf/special_chars.py): check the test column set equals the train set.
    sc_te = ext.sc_extractor.transform(X_te)
    sc_tr_vocab = _vocab(ext.sc_extractor.vectorizer)
    print("special-char test cols:", sc_te.shape[1], "vs train vocab:", len(sc_tr_vocab))
