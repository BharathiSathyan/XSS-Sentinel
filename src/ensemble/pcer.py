"""
Per-Class Expert Routing (PCER)
=================================
Novel algorithm designed for this project.

Motivation
----------
Standard ensemble methods (majority vote, soft averaging) assign the same trust to
every model regardless of which class is being predicted.  Yet tree-based ensemble
models are known to develop *differential class expertise*: a model trained with
SMOTE may learn better decision boundaries for rare classes, while a model trained
with class weights may generalise differently on the majority class.

PCER exploits this by pre-computing a "class-expert table" — a mapping from each
class to the model that achieves the highest precision on that class in training.
At inference time, after obtaining a consensus class prediction (via soft vote),
PCER routes the final decision to that class's expert, provided the expert is
confident enough.

Algorithm
---------
Offline (training / fit phase):
  1. For each model k and each class c, compute per-class precision on X_train:
         precision(k, c) = TP_kc / (TP_kc + FP_kc)
  2. Assign expert: expert[c] = argmax_k precision(k, c)
  3. Store fallback weights: inverse-precision weights across models per class.

Online (inference / predict phase):
  For each sample x:
  1. Compute soft-vote consensus:
         consensus_class = argmax sum_k P_k(c | x)
  2. Identify the expert model for consensus_class:
         E = expert[consensus_class]
  3. If P_E(consensus_class | x) >= confidence_threshold:
         final = consensus_class   (expert confirms the soft-vote)
     Else:
         final = argmax over classes of:
             max_c [ sum_k precision(k, c) * P_k(c | x) ]
         i.e., precision-weighted soft vote as the fallback.

Theoretical Justification
--------------------------
- Expert selection from per-class precision is grounded in the Neyman-Pearson
  classification paradigm: precision directly measures the model's ability to
  distinguish class c from all others (positive predictive value).
- The confidence gate prevents over-trust of the expert when it is uncertain,
  reverting to a full ensemble view.
- The fallback precision-weighted soft vote is a generalisation of BMA's per-class
  weighting applied at decision time.
- For severely imbalanced classes (e.g., DOM-based XSS with 30 samples), having a
  dedicated expert model that has been pushed toward that class (by SMOTE or class
  weights) is particularly beneficial.

Comparison with LCCDE
---------------------
LCCDE has no concept of class expertise.  Its majority-vote step will almost always
defer to the two models that were trained with SMOTE (LGBM, XGB) and may ignore
CatBoost's class-weighted training, which is specifically designed to handle
imbalanced classes.  PCER explicitly identifies which model is best at each class
and routes the decision accordingly.
"""

import numpy as np


class PCER:
    """
    Per-Class Expert Routing ensemble.

    Parameters
    ----------
    models : list of fitted classifiers
        Each must implement predict_proba(X) and predict(X).
    n_classes : int
        Number of target classes (default 4).
    confidence_threshold : float
        Minimum probability the expert must assign to the consensus class to
        confirm it.  If below threshold, precision-weighted fallback is used.
        Default 0.5.
    """

    def __init__(self, models, n_classes=4, confidence_threshold=0.5):
        self.models = models
        self.n_classes = n_classes
        self.confidence_threshold = confidence_threshold
        self.expert_table = None          # shape (n_classes,)  — model index per class
        self.class_precision = None       # shape (n_classes, n_models)

    # ------------------------------------------------------------------
    # Training phase: compute per-class precision and expert table
    # ------------------------------------------------------------------
    def fit(self, X_train, y_train):
        """
        Pre-compute per-class expert table from training data.

        Parameters
        ----------
        X_train : ndarray, shape (n_train, n_features)
        y_train : ndarray, shape (n_train,)  — integer labels
        """
        n_models = len(self.models)
        # precision_matrix[c, k] = precision of model k on class c
        precision_matrix = np.zeros((self.n_classes, n_models), dtype=float)

        for k, model in enumerate(self.models):
            preds = np.array(model.predict(X_train)).flatten().astype(int)
            for c in range(self.n_classes):
                y_true_bin = (y_train == c).astype(int)
                y_pred_bin = (preds == c).astype(int)
                tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
                fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
                # Laplace-smoothed precision
                precision_matrix[c, k] = (tp + 1.0) / (tp + fp + n_models)

        self.class_precision = precision_matrix            # (n_classes, n_models)
        # Expert for class c = model with highest precision on class c
        self.expert_table = np.argmax(precision_matrix, axis=1)  # (n_classes,)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X):
        """
        Route each sample through the per-class expert.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        ndarray of int, shape (n_samples,)
        """
        if self.expert_table is None:
            raise RuntimeError("Call fit() before predict().")

        # Collect all probability matrices upfront (n_samples, n_classes) per model
        all_probas = [np.array(m.predict_proba(X), dtype=float)
                      for m in self.models]

        n = X.shape[0]
        preds = np.empty(n, dtype=int)

        for i in range(n):
            # Step 1: Uniform soft-vote to find consensus class
            soft_proba = np.mean(
                np.stack([all_probas[k][i] for k in range(len(self.models))], axis=0),
                axis=0
            )  # (n_classes,)
            consensus_class = int(np.argmax(soft_proba))

            # Step 2: Query the expert for the consensus class
            expert_k = int(self.expert_table[consensus_class])
            expert_confidence = float(all_probas[expert_k][i][consensus_class])

            # Step 3: Expert confirms or fallback
            if expert_confidence >= self.confidence_threshold:
                preds[i] = consensus_class
            else:
                # Precision-weighted soft vote as fallback
                # For each class c: score(c) = sum_k precision(c, k) * P_k(c|x)
                score = np.zeros(self.n_classes, dtype=float)
                for k in range(len(self.models)):
                    for c in range(self.n_classes):
                        score[c] += self.class_precision[c, k] * all_probas[k][i][c]
                preds[i] = int(np.argmax(score))

        return preds

    def predict_proba(self, X):
        """
        Return precision-weighted soft-vote probabilities (for inspection).
        Shape: (n_samples, n_classes)
        """
        all_probas = [np.array(m.predict_proba(X), dtype=float)
                      for m in self.models]
        n = X.shape[0]
        result = np.zeros((n, self.n_classes), dtype=float)

        for k in range(len(self.models)):
            for c in range(self.n_classes):
                result[:, c] += self.class_precision[c, k] * all_probas[k][:, c]

        # Normalise to valid distribution
        row_sums = result.sum(axis=1, keepdims=True)
        result /= np.where(row_sums == 0, 1.0, row_sums)
        return result
