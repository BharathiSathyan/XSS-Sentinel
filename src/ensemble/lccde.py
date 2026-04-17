import numpy as np

class LCCDE:

    def __init__(self, lgbm, xgb, cat):
        self.lgbm = lgbm
        self.xgb = xgb
        self.cat = cat

    def predict(self, X):

        lgbm_pred = self.lgbm.predict(X)
        xgb_pred = self.xgb.predict(X)
        cat_pred = self.cat.predict(X)

        lgbm_prob = self.lgbm.predict_proba(X)
        xgb_prob = self.xgb.predict_proba(X)
        cat_prob = self.cat.predict_proba(X)

        final_preds = []

        for i in range(len(X)):

            preds = [lgbm_pred[i], xgb_pred[i], cat_pred[i]]

            # Case 1: all agree
            if preds.count(preds[0]) == 3:
                final_preds.append(preds[0])
                continue

            # Case 2: two agree
            if len(set(preds)) == 2:
                final_preds.append(max(set(preds), key=preds.count))
                continue

            # Case 3: all disagree
            probs = [
                max(lgbm_prob[i]),
                max(xgb_prob[i]),
                max(cat_prob[i])
            ]

            best_model = np.argmax(probs)

            final_preds.append(preds[best_model])

        return np.array(final_preds)