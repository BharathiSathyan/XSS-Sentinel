"""
Weighted Soft Voting (WSV) Ensemble
====================================
Theory
------
Each base classifier outputs a probability distribution P_k(y | x) over C classes.
WSV combines them as a weighted convex mixture:

    P_wsv(y | x) = sum_k [ w_k * P_k(y | x) ]    where sum_k(w_k) = 1, w_k >= 0

A weighted mixture of valid probability distributions is itself a valid probability
distribution (the probability simplex is convex), making argmax P_wsv a principled
Bayes-optimal predictor under the mixture model.

Weight Selection
----------------
Weights are estimated from each model's macro-F1 on a held-out validation set,
then L1-normalised so they sum to 1:

    w_k = F1_k / sum_j(F1_j)

Macro-F1 is chosen over accuracy because our dataset is heavily imbalanced
(DOM-based XSS: only 30 samples total). Macro-F1 gives equal weight to each class,
penalising a model that ignores rare classes.

Comparison with LCCDE
---------------------
LCCDE uses hard labels and majority voting.  In its "all-disagree" case it picks
the single most confident model and discards the other two models' full probability
vectors entirely.  WSV always aggregates the entire distribution from every model,
losing no probability signal.  It is strictly more information-preserving than LCCDE.
"""

import numpy as np


class WeightedSoftVoting:
    """
    Weighted Soft Voting ensemble for multi-class probabilistic classifiers.

    Parameters
    ----------
    models : list of fitted classifiers
        Each must implement predict_proba(X) -> (n_samples, n_classes).
    weights : array-like of shape (len(models),), optional
        Non-negative weights.  If None, call fit() to compute from validation F1.
    """

    def __init__(self, models, weights=None):
        self.models = models
        if weights is not None:
            w = np.array(weights, dtype=float)
            if w.sum() == 0:
                raise ValueError("Weights must not all be zero.")
            self.weights = w / w.sum()
        else:
            self.weights = None

    # ------------------------------------------------------------------
    # Weight estimation from validation macro-F1
    # ------------------------------------------------------------------
    def fit(self, X_val, y_val):
        """
        Compute model weights as normalised macro-F1 on a validation split.

        Parameters
        ----------
        X_val : ndarray, shape (n_val, n_features)
        y_val : ndarray, shape (n_val,)  — integer-encoded labels
        """
        from sklearn.metrics import f1_score

        raw = []
        for model in self.models:
            preds = np.array(model.predict(X_val)).flatten().astype(int)
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            raw.append(f1)

        raw = np.array(raw, dtype=float)
        total = raw.sum()
        self.weights = raw / total if total > 0 else np.ones(len(self.models)) / len(self.models)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Return the weighted mixture distribution P_wsv(y|x).
        Shape: (n_samples, n_classes)
        """
        if self.weights is None:
            raise RuntimeError("Call fit() before predict_proba().")

        mixture = None
        for w, model in zip(self.weights, self.models):
            proba = np.array(model.predict_proba(X), dtype=float)
            mixture = w * proba if mixture is None else mixture + w * proba

        return mixture  # valid probability distribution over n_classes

    def predict(self, X):
        """Return argmax of the weighted mixture distribution."""
        return np.argmax(self.predict_proba(X), axis=1)
