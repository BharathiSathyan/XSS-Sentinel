"""
Stacking (Stacked Generalisation) Ensemble
============================================
Theory — Wolpert 1992
----------------------
Stacking introduces a *meta-learner* that learns how to optimally combine the
predictions of a set of base learners.  The key insight is that the base models'
*probability outputs* are treated as features for a second-level model, which can
learn non-trivial, input-dependent combination rules.

Formally, let f_k(x) = P_k(y | x) be the probability vector from base model k.
Define the *meta-feature vector*:

    z(x) = concat[f_1(x), f_2(x), ..., f_K(x)]   shape: (K * C,)

A meta-learner g is trained on { (z(x_i), y_i) } to produce the final prediction.

Out-of-Fold (OOF) Design for Leakage Prevention
-------------------------------------------------
If we simply generate z(x_train) using the same models trained on all of x_train,
the meta-learner sees predictions that had zero generalisation error on training
points, creating severe overfitting (data leakage).

The solution is Out-of-Fold prediction:
  1. Divide x_train into K folds (here K=5 by default).
  2. For each fold f:
       a. Train a *fresh copy* of each base model on the other K-1 folds.
       b. Predict probabilities on fold f using this fresh copy.
  3. Collect all OOF predictions to form z_meta_train.
  4. Train the meta-learner on z_meta_train with true labels y_train.
  5. At test time, use the *original* full-dataset-trained base models to build
     z_meta_test, then apply the meta-learner.

This ensures the meta-learner always trains on hold-out predictions, matching the
generalisation regime it will see at test time.

Meta-Learner Choice
-------------------
Logistic Regression with L2 penalty is the canonical and theoretically principled
choice for stacking:
  - It is a log-linear model, which is the maximum-entropy model given linear
    constraints on the expected class probabilities (no unnecessary assumptions).
  - The L2 penalty provides closed-form Bayesian regularisation (Gaussian prior).
  - It is inherently calibrated, producing valid probabilities.
  - It is interpretable: weights reveal which base model is trusted per class.

Comparison with LCCDE
---------------------
LCCDE uses fixed hand-crafted rules (majority vote, highest-confidence fallback).
Stacking learns the combination rule from data, adapting to per-class model
expertise automatically.  It can, for example, learn that CatBoost is more reliable
for DOM-based XSS while LGBM dominates for Reflected XSS.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import copy


class StackingEnsemble:
    """
    Stacked Generalisation ensemble using Out-of-Fold meta-feature generation.

    Parameters
    ----------
    base_models : list of (name, fitted_clf) tuples
        Each clf must implement fit(X, y), predict(X), predict_proba(X).
    meta_learner : sklearn-compatible classifier, optional
        Defaults to LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial').
    n_folds : int
        Number of cross-validation folds for OOF generation (default 5).
    random_state : int
        Seed for StratifiedKFold reproducibility.
    n_classes : int
        Number of output classes (must be set before fit).
    """

    def __init__(self, base_models, meta_learner=None, n_folds=5,
                 random_state=42, n_classes=4):
        self.base_models = base_models          # list of (name, clf)
        self.meta_learner = meta_learner or LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            random_state=random_state
        )
        self.n_folds = n_folds
        self.random_state = random_state
        self.n_classes = n_classes
        self._fitted_base_models = None         # full-data trained clfs for inference

    # ------------------------------------------------------------------
    # Helper: get proba from a model (handles CatBoost 2D edge case)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_proba(model, X):
        proba = model.predict_proba(X)
        return np.array(proba, dtype=float)

    # ------------------------------------------------------------------
    # Fit: OOF meta-feature generation + meta-learner training
    # ------------------------------------------------------------------
    def fit(self, X_train, y_train):
        """
        1. Generate OOF meta-features via StratifiedKFold.
        2. Train meta-learner on OOF features.
        3. Retain the already-trained full-dataset base models for inference.

        Parameters
        ----------
        X_train : ndarray, shape (n_train, n_features)
        y_train : ndarray, shape (n_train,)  — integer labels
        """
        n_samples = X_train.shape[0]
        n_base = len(self.base_models)
        # Meta-feature matrix: one block of n_classes columns per base model
        Z_oof = np.zeros((n_samples, n_base * self.n_classes), dtype=float)

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                              random_state=self.random_state)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val   = X_train[val_idx]

            for model_idx, (name, clf) in enumerate(self.base_models):
                # Deep-copy so we don't mutate the caller's trained model
                fold_clf = copy.deepcopy(clf)
                fold_clf.fit(X_fold_train, y_fold_train)
                proba = self._get_proba(fold_clf, X_fold_val)  # (|val|, C)
                col_start = model_idx * self.n_classes
                col_end   = col_start + self.n_classes
                Z_oof[val_idx, col_start:col_end] = proba

        # Train meta-learner on OOF meta-features
        self.meta_learner.fit(Z_oof, y_train)

        # Store the original (full-data) fitted models for test-time inference
        self._fitted_base_models = [clf for _, clf in self.base_models]
        return self

    # ------------------------------------------------------------------
    # Build meta-features from the full-data base models (test time)
    # ------------------------------------------------------------------
    def _build_meta_features(self, X):
        """Concatenate predict_proba from each full-data base model."""
        blocks = []
        for model in self._fitted_base_models:
            proba = self._get_proba(model, X)
            blocks.append(proba)
        return np.hstack(blocks)   # (n_samples, n_base * n_classes)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """Return meta-learner probability distribution. Shape: (n, n_classes)."""
        if self._fitted_base_models is None:
            raise RuntimeError("Call fit() before predict_proba().")
        Z = self._build_meta_features(X)
        return self.meta_learner.predict_proba(Z)

    def predict(self, X):
        """Return final class predictions."""
        return np.argmax(self.predict_proba(X), axis=1)
