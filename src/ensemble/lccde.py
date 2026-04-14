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

        for i in range(X.shape[0]):

            p1 = int(lgbm_pred[i])
            p2 = int(xgb_pred[i])
            p3 = int(np.array(cat_pred[i]).flatten()[0])

            preds = [p1, p2, p3]

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
                lgbm_prob[i][p1],
                xgb_prob[i][p2],
                cat_prob[i][p3]
            ]

            best_model = np.argmax(probs)
            final_preds.append(preds[best_model])

        return np.array(final_preds)