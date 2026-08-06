import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

class StrictLCCDE:
    """
    Implementation of Strict Algorithm 2: Leader Class and Confidence Decision Ensemble (LCCDE).
    Pre-computes class leaders using Stratified Cross-Validation on training data.
    """
    def __init__(self, lgbm, xgb, cat):
        self.models = [lgbm, xgb, cat]
        self.leader_map = {}

    def fit_leaders_cv(self, X_train_raw, y_train_raw):
        """
        Determines the leader model for each class via 3-Fold Stratified CV on training data.
        """
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        y_train_raw_np = np.array(y_train_raw)
        oof_preds = np.zeros((3, len(y_train_raw)))

        for train_idx, val_idx in skf.split(X_train_raw, y_train_raw_np):
            X_tr, X_va = X_train_raw[train_idx], X_train_raw[val_idx]
            y_tr, y_va = y_train_raw_np[train_idx], y_train_raw_np[val_idx]

            # SMOTE for LGBM & XGB
            smote = SMOTE(random_state=42)
            X_tr_smote, y_tr_smote = smote.fit_resample(X_tr, y_tr)

            # Fit 3 models on fold (optimized n_estimators for fast leader estimation)
            m_lgbm = LGBMClassifier(objective="multiclass", num_class=4, n_estimators=30, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
            m_lgbm.fit(X_tr_smote, y_tr_smote)
            oof_preds[0, val_idx] = m_lgbm.predict(X_va).flatten()

            m_xgb = XGBClassifier(objective="multi:softprob", num_class=4, n_estimators=30, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1)
            m_xgb.fit(X_tr_smote, y_tr_smote)
            oof_preds[1, val_idx] = m_xgb.predict(X_va).flatten()

            class_counts = pd.Series(y_tr).value_counts()
            class_weights = {cls: len(y_tr) / (len(class_counts) * count) for cls, count in class_counts.items()}
            m_cat = CatBoostClassifier(loss_function="MultiClass", iterations=30, learning_rate=0.1, depth=4, class_weights=class_weights, random_seed=42, thread_count=-1, task_type="CPU", verbose=False)
            m_cat.fit(X_tr, y_tr)
            oof_preds[2, val_idx] = np.array(m_cat.predict(X_va)).flatten()

        num_classes = len(np.unique(y_train_raw_np))
        f1_scores = np.zeros((3, num_classes))
        for m_idx in range(3):
            f1s = f1_score(y_train_raw_np, oof_preds[m_idx].astype(int), average=None)
            for c in range(len(f1s)):
                f1_scores[m_idx, c] = f1s[c]

        for c in range(num_classes):
            self.leader_map[c] = int(np.argmax(f1_scores[:, c]))

        return self.leader_map, f1_scores

    def predict(self, X):
        preds = []
        probs = []
        for model in self.models:
            p = model.predict(X)
            if hasattr(p, "flatten"):
                p = np.array(p).flatten()
            preds.append(p.astype(int))
            probs.append(model.predict_proba(X))

        n_samples = X.shape[0]
        final_preds = []

        for i in range(n_samples):
            p0, p1, p2 = preds[0][i], preds[1][i], preds[2][i]
            sample_preds = [p0, p1, p2]
            sample_probs = [probs[0][i][p0], probs[1][i][p1], probs[2][i][p2]]

            # Case 1: All 3 agree
            if p0 == p1 and p1 == p2:
                final_preds.append(p0)
                continue

            # Case 2: All 3 disagree
            if p0 != p1 and p1 != p2 and p0 != p2:
                leader_matches = []
                leader_probs = []
                for m_idx in range(3):
                    predicted_class = sample_preds[m_idx]
                    if self.leader_map.get(predicted_class, -1) == m_idx:
                        leader_matches.append(predicted_class)
                        leader_probs.append(sample_probs[m_idx])

                if len(leader_matches) == 1:
                    final_preds.append(leader_matches[0])
                else:
                    best_m = np.argmax(sample_probs)
                    final_preds.append(sample_preds[best_m])
                continue

            # Case 3: Two agree, 1 differs
            vals, counts = np.unique(sample_preds, return_counts=True)
            maj_class = vals[np.argmax(counts)]

            leader_m_idx = self.leader_map.get(maj_class, -1)
            if leader_m_idx != -1:
                leader_pred = sample_preds[leader_m_idx]
                final_preds.append(leader_pred)
            else:
                final_preds.append(maj_class)

        return np.array(final_preds)
