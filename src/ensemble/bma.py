"""
Bayesian Model Averaging (BMA)
================================
Theory — Hoeting et al. 1999 (Statistical Science)
----------------------------------------------------
BMA is a principled way to combine model predictions by marginalising over
model uncertainty.  Under the BMA framework, the posterior predictive distribution
over class y given data x is:

    P_bma(y | x, D_train) = sum_k [ P(M_k | D_train) * P_k(y | x) ]

where P(M_k | D_train) is the *posterior probability of model M_k* given the
training data D_train.

This is mathematically identical to Weighted Soft Voting, with one crucial
difference: the weights have a principled probabilistic interpretation
(posterior model probabilities) rather than being heuristic scores.

Computing Posterior Model Probabilities
---------------------------------------
The exact Bayesian posterior requires computing the marginal likelihood
(model evidence) P(D_train | M_k), which is intractable for gradient boosted trees.

We use the well-known *BIC approximation* (Schwarz 1978):

    log P(M_k | D_train) ≈ log P(D_train | M_k, theta_hat_k) - (d_k / 2) * log(n)
                         = LL_k - (d_k / 2) * log(n)

where:
  - LL_k  = log-likelihood of model k on training data
  - d_k   = number of effective parameters (approximated by n_estimators * depth
             as a proxy for model complexity in tree ensembles)
  - n     = number of training samples

The posterior weights are then obtained by:

    unnorm_weight_k = exp(log P(M_k | D_train))
    w_k = unnorm_weight_k / sum_j(unnorm_weight_j)

This is the standard Bayesian AIC/BIC model averaging formula (Burnham & Anderson 2002).

Per-Class BMA Variant (used here)
-----------------------------------
Rather than a global posterior weight, we estimate *per-class* model reliability:

    P_bma(y=c | x) = sum_k [ w_kc * P_k(y=c | x) ]

where w_kc is the model's precision on class c in the training set
(calibrated to form a valid mixture per class).

This per-class variant is more powerful than global BMA because it allows, e.g.,
CatBoost to dominate the DOM-based XSS class weight while LGBM dominates Reflected XSS.

Comparison with LCCDE
---------------------
LCCDE has no probabilistic model of which model is more reliable for which class.
Per-class BMA explicitly learns this from training data, giving it a richer
representation of model-class expertise.
"""

import numpy as np


class BayesianModelAveraging:
    """
    Per-class Bayesian Model Averaging ensemble.

    Parameters
    ----------
    models : list of fitted classifiers
        Each must implement predict_proba(X) -> (n_samples, n_classes).
    n_classes : int
        Number of target classes (default 4).
    """

    def __init__(self, models, n_classes=4):
        self.models = models
        self.n_classes = n_classes
        self.class_weights = None   # shape (n_classes, n_models) — set in fit()

    # ------------------------------------------------------------------
    # Estimate per-class weights from training data
    # ------------------------------------------------------------------
    def fit(self, X_train, y_train):
        """
        Compute per-class model weights as normalised per-class precision.

        For each class c, w_kc = precision_k(c) / sum_j precision_j(c).

        Precision for class c = TP_c / (TP_c + FP_c) on the training set.

        Parameters
        ----------
        X_train : ndarray, shape (n_train, n_features)
        y_train : ndarray, shape (n_train,)  — integer labels
        """
        from sklearn.metrics import precision_score

        n_models = len(self.models)
        # class_weights[c, k] = w_kc
        class_weights = np.zeros((self.n_classes, n_models), dtype=float)

        for k, model in enumerate(self.models):
            preds = np.array(model.predict(X_train)).flatten().astype(int)
            for c in range(self.n_classes):
                y_true_bin = (y_train == c).astype(int)
                y_pred_bin = (preds == c).astype(int)
                tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
                fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
                # Precision with Laplace smoothing (prevents 0-weight for unseen class)
                precision_kc = (tp + 1.0) / (tp + fp + n_models)
                class_weights[c, k] = precision_kc

        # Normalise per class so weights sum to 1 across models
        row_sums = class_weights.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid div-by-zero
        self.class_weights = class_weights / row_sums       # (n_classes, n_models)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Return the per-class BMA mixture distribution.

        For each sample i and class c:
            P_bma(c | x_i) = sum_k [ w_kc * P_k(c | x_i) ]

        Shape: (n_samples, n_classes)
        """
        if self.class_weights is None:
            raise RuntimeError("Call fit() before predict_proba().")

        n = X.shape[0]
        result = np.zeros((n, self.n_classes), dtype=float)

        for k, model in enumerate(self.models):
            proba_k = np.array(model.predict_proba(X), dtype=float)  # (n, C)
            for c in range(self.n_classes):
                result[:, c] += self.class_weights[c, k] * proba_k[:, c]

        # Renormalise rows to a valid probability distribution
        row_sums = result.sum(axis=1, keepdims=True)
        result = result / np.where(row_sums == 0, 1.0, row_sums)
        return result

    def predict(self, X):
        """Return argmax of the BMA predictive distribution."""
        return np.argmax(self.predict_proba(X), axis=1)
