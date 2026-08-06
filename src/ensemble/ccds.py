"""
Confidence-Calibrated Dynamic Selection (CCDS)
================================================
Novel algorithm designed for this project.

Motivation
----------
LCCDE treats disagreement as binary: either models agree (majority vote) or they
don't (pick the most confident single model).  This binary view ignores *how much*
the models disagree.  If two models produce nearly identical probability vectors but
a third is wildly different, majority-vote is appropriate.  If all three are
similarly uncertain, picking any single model's max-confidence answer is unreliable.

CCDS introduces a *disagreement spectrum* via information-theoretic quantities and
adapts its prediction strategy accordingly.

Algorithm (per sample)
-----------------------
Step 1: Compute per-model Shannon entropy (uncertainty measure).

    H_k = -sum_c [ P_k(c|x) * log P_k(c|x) ]

    Low H_k → model k is certain.  High H_k → model k is uncertain.

Step 2: Compute pairwise Jensen-Shannon Divergence (JSD) between each pair of
        model distributions.  JSD is the symmetric, bounded (0..1) version of
        KL-divergence, making it a true probability-theoretic distance:

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)    where M = 0.5*(P+Q)

    The *ensemble disagreement score* D is the mean of all pairwise JSDs.

Step 3: Route to one of three strategies based on thresholds:

    a) Low disagreement  (D < theta_low):
       → Soft-vote using all models weighted by inverse-entropy
         (certain models get higher weight).

    b) Medium disagreement (theta_low <= D <= theta_high):
       → Identify the *outlier model* (the one with highest mean JSD to the others)
         and drop it.  Soft-vote the remaining two.

    c) High disagreement  (D > theta_high):
       → The models are fundamentally uncertain.  Use the model with lowest
         entropy (highest individual certainty) as the authority, but only if its
         maximum probability exceeds a confidence threshold.  Otherwise fall back
         to inverse-entropy soft vote of all models.

Key Properties
--------------
- Never arbitrarily discards models; even in strategy (b) the outlier's probability
  vector is completely ignored only after a principled distance test.
- Entropy weighting is theoretically grounded: it assigns higher weight to models
  whose distributions are more peaked (concentrated mass → more information).
- JSD is bounded in [0, 1] and symmetric, making thresholds interpretable.
- Default thresholds (0.05, 0.15) chosen so that values near 0 indicate near-
  identical distributions and values near 0.15 indicate meaningful divergence
  given typical softmax outputs from calibrated gradient boosters.

Comparison with LCCDE
---------------------
LCCDE Case 3 (all disagree) picks the model with max P_k(argmax_k|x), ignoring
all other probability values.  CCDS instead uses the entire distribution's entropy
to rank model certainty, and applies JSD to determine *whether* a model is truly an
outlier before dropping it.
"""

import numpy as np


def _kl_divergence(p, q, eps=1e-12):
    """KL(p || q) — add eps for numerical stability."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _jsd(p, q):
    """Jensen-Shannon Divergence — symmetric, bounded in [0, log2] (nats: [0, ln2])."""
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _entropy(p, eps=1e-12):
    """Shannon entropy H(p) = -sum p * log(p) in nats."""
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


class CCDS:
    """
    Confidence-Calibrated Dynamic Selection ensemble.

    Parameters
    ----------
    models : list of fitted classifiers (in order: lgbm, xgb, cat)
        Each must implement predict_proba(X).
    theta_low : float
        JSD threshold below which low-disagreement soft-vote is used.
        Default 0.05 (≈ distributions very close together).
    theta_high : float
        JSD threshold above which high-disagreement authority selection is used.
        Default 0.15 (≈ meaningful divergence for typical boosting models).
    min_confidence : float
        Minimum max-probability required for authority selection in strategy (c).
        Default 0.5 (model must be more than half-confident in its top class).
    """

    def __init__(self, models, theta_low=0.05, theta_high=0.15, min_confidence=0.5):
        if not (0.0 <= theta_low < theta_high):
            raise ValueError("Require 0 <= theta_low < theta_high.")
        self.models = models
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Per-sample prediction logic
    # ------------------------------------------------------------------
    def _predict_sample(self, probas):
        """
        Parameters
        ----------
        probas : list of 1-D arrays, shape (n_classes,) each
            Probability distributions from each model for one sample.

        Returns
        -------
        int  — predicted class index
        """
        K = len(probas)

        # Step 1: Shannon entropy per model
        entropies = np.array([_entropy(p) for p in probas])   # (K,)

        # Step 2: Pairwise JSD matrix and mean disagreement score D
        jsd_matrix = np.zeros((K, K))
        for i in range(K):
            for j in range(i + 1, K):
                d = _jsd(probas[i], probas[j])
                jsd_matrix[i, j] = d
                jsd_matrix[j, i] = d

        # Mean JSD across all pairs (upper triangle)
        n_pairs = K * (K - 1) / 2
        D = jsd_matrix[np.triu_indices(K, k=1)].sum() / n_pairs

        # Step 3: Route to strategy
        if D < self.theta_low:
            # ---- Strategy (a): Low disagreement ----
            # Inverse-entropy weights: more certain model → higher weight
            # If entropy is 0 (perfect certainty) use a large weight cap.
            inv_ent = 1.0 / (entropies + 1e-12)
            weights = inv_ent / inv_ent.sum()
            mixture = sum(w * p for w, p in zip(weights, probas))
            return int(np.argmax(mixture))

        elif D <= self.theta_high:
            # ---- Strategy (b): Medium disagreement ----
            # Identify outlier: model with highest mean JSD to all others
            mean_jsd_to_others = jsd_matrix.mean(axis=1)     # (K,)
            outlier_idx = int(np.argmax(mean_jsd_to_others))
            # Drop outlier; soft-vote remaining with inverse-entropy weights
            remaining = [(i, probas[i], entropies[i])
                         for i in range(K) if i != outlier_idx]
            inv_ent = np.array([1.0 / (e + 1e-12) for _, _, e in remaining])
            weights = inv_ent / inv_ent.sum()
            mixture = sum(w * p for w, (_, p, _) in zip(weights, remaining))
            return int(np.argmax(mixture))

        else:
            # ---- Strategy (c): High disagreement ----
            # Authority: model with lowest entropy (highest individual certainty)
            authority_idx = int(np.argmin(entropies))
            authority_proba = probas[authority_idx]
            max_p = float(np.max(authority_proba))

            if max_p >= self.min_confidence:
                return int(np.argmax(authority_proba))
            else:
                # Authority is still uncertain; fall back to inverse-entropy soft vote
                inv_ent = 1.0 / (entropies + 1e-12)
                weights = inv_ent / inv_ent.sum()
                mixture = sum(w * p for w, p in zip(weights, probas))
                return int(np.argmax(mixture))

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------
    def predict(self, X):
        """
        Predict class labels for all samples in X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        ndarray of int, shape (n_samples,)
        """
        # Collect probability matrices: list of (n_samples, n_classes) arrays
        all_probas = [np.array(m.predict_proba(X), dtype=float)
                      for m in self.models]

        n = X.shape[0]
        preds = np.empty(n, dtype=int)

        for i in range(n):
            sample_probas = [all_probas[k][i] for k in range(len(self.models))]
            preds[i] = self._predict_sample(sample_probas)

        return preds
