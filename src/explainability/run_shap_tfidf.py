"""
run_shap_tfidf.py
=================
Explainability (SHAP) analysis for XSS Sentinel on the CAXF + TF-IDF representation.

Why TF-IDF?  It is the only CAXF instantiation whose 5,282 columns are individually
human-readable (HTML tags, JS events, URL patterns, special characters, XSS keywords
and character n-grams).  CharCNN (random-init, frozen) and MiniLM columns are opaque
latent dimensions, so per-feature SHAP names would not be meaningful there.

What is explained
-----------------
The three gradient-boosting base learners that feed *every* ensemble strategy
(LightGBM, XGBoost, CatBoost) are explained exactly with TreeSHAP on the 3,276-sample
test set, using the models and feature matrices that produced the paper's numbers
(results/cache/tfidf).  Ensemble-level behaviour is explained through
  * the Stacking meta-learner coefficients,
  * the CPWA class-conditional weights and PCER expert table, and
  * an ensemble-average attribution (mean of the three base-model SHAP vectors),
    which is validated with a feature-deletion (faithfulness) test applied to the
    actual Stacking ensemble.

SHAP values are in the models' raw (log-odds / margin) output space.

Run from the project root:
    venv\\Scripts\\python.exe src\\explainability\\run_shap_tfidf.py
"""

import json
import os
import sys
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, accuracy_score, recall_score

warnings.filterwarnings("ignore")

_here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _here)
from feature_names import (GROUPS, CACHE_DIR, build_feature_names,          # noqa: E402
                           load_cached_features, load_split, _root)
from src.ensemble.bma import BayesianModelAveraging                           # noqa: E402  (CPWA)
from src.ensemble.pcer import PCER                                            # noqa: E402

SEED = 42
OUT = os.path.join(_root, "results", "explainability")
os.makedirs(OUT, exist_ok=True)

GROUP_COLORS = {
    "HTML tags": "#2563EB", "JS events": "#F97316", "URL patterns": "#A855F7",
    "Special chars": "#16A34A", "XSS keywords": "#DC2626", "Char n-gram TF-IDF": "#64748B",
}
MODEL_LABELS = {"lgbm": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost"}
LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


def disp(name):
    """Make a feature name safe for matplotlib (mathtext) labels."""
    return str(name).replace("$", r"\$")


def shap_3d(explainer, X, chunk=800):
    """TreeSHAP -> float32 array (n, f, C); handles list / ndarray return types."""
    parts = []
    for s in range(0, X.shape[0], chunk):
        sv = explainer.shap_values(X[s:s + chunk])
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)
        parts.append(np.asarray(sv, dtype=np.float32))
    return np.concatenate(parts, axis=0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def macro_f1(y, p):
    return f1_score(y, p, average="macro")


def main():
    t_start = time.time()
    # ------------------------------------------------------------------ data
    log("=" * 70)
    log("SHAP EXPLAINABILITY - CAXF + TF-IDF  (seed=42)")
    log("=" * 70)
    X_tr_txt, X_te_txt, y_tr, y_te, le = load_split(SEED)
    classes = list(le.classes_)
    names, groups, _ = build_feature_names(X_tr_txt)
    Xtr, Xte = load_cached_features(SEED)
    assert Xte.shape[1] == len(names) == 5282
    log(f"classes: {classes}")
    log(f"train {Xtr.shape}, test {Xte.shape}; test support: "
        f"{dict(zip(classes, np.bincount(y_te).tolist()))}")

    # ---------------------------------------------------------------- models
    lgbm = joblib.load(os.path.join(CACHE_DIR, "lgbm.pkl"))
    xgb = joblib.load(os.path.join(CACHE_DIR, "xgb_seed_42.pkl"))
    cat = joblib.load(os.path.join(CACHE_DIR, "cat.pkl"))
    stack = joblib.load(os.path.join(CACHE_DIR, "stacking_ensemble_seed_42.pkl"))
    models = {"lgbm": lgbm, "xgb": xgb, "cat": cat}

    log("\n[Sanity] cached models reproduce the paper's test metrics:")
    paper = {"lgbm": 0.9503, "xgb": 0.9188, "cat": 0.8601}
    for k, m in models.items():
        p = np.asarray(m.predict(Xte)).flatten().astype(int)
        log(f"  {MODEL_LABELS[k]:9s} macro-F1 = {macro_f1(y_te, p):.4f}  (paper {paper[k]:.4f})")
    p_stack = stack.predict(Xte)
    log(f"  Stacking  macro-F1 = {macro_f1(y_te, p_stack):.4f}  (paper 0.9530), "
        f"acc = {accuracy_score(y_te, p_stack):.4f}")

    # ----------------------------------------------------------------- SHAP
    log("\n[SHAP] computing TreeSHAP on the test set ...")
    sv, base = {}, {}
    for k, m in models.items():
        t0 = time.time()
        ex = shap.TreeExplainer(m)
        sv[k] = shap_3d(ex, Xte)
        ev = np.asarray(ex.expected_value, dtype=float).reshape(-1)
        base[k] = ev
        # additivity check: softmax(base + sum SHAP) must equal predict_proba
        proba = softmax(sv[k].sum(axis=1) + ev[None, :])
        err = np.abs(proba - np.asarray(m.predict_proba(Xte))).max()
        log(f"  {MODEL_LABELS[k]:9s} shape={sv[k].shape}  {time.time()-t0:5.1f}s  "
            f"max|softmax(base+sum(SHAP)) - predict_proba| = {err:.2e}")

    # mean |SHAP| tables  (feature x class), per model
    imp = {k: np.abs(sv[k]).mean(axis=0) for k in models}              # (f, C)
    imp_glob = {k: imp[k].mean(axis=1) for k in models}                 # (f,)
    norm_glob = {k: imp_glob[k] / imp_glob[k].sum() for k in models}
    consensus = np.mean([norm_glob[k] for k in models], axis=0)
    norm_cls = {k: imp[k] / imp[k].sum(axis=0, keepdims=True) for k in models}
    consensus_cls = np.mean([norm_cls[k] for k in models], axis=0)      # (f, C)

    # ------------------------------------------- Table: top features / model
    rows = []
    for k in models:
        for rank, j in enumerate(np.argsort(-imp_glob[k])[:20], 1):
            rows.append(dict(model=MODEL_LABELS[k], rank=rank, feature=names[j],
                             group=groups[j], mean_abs_shap=float(imp_glob[k][j])))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "top20_features_per_model.csv"), index=False)

    rows = []
    for c, cname in enumerate(classes):
        for rank, j in enumerate(np.argsort(-consensus_cls[:, c])[:10], 1):
            rows.append(dict(cls=cname, rank=rank, feature=names[j], group=groups[j],
                             consensus_share=float(consensus_cls[j, c])))
    df_cls = pd.DataFrame(rows)
    df_cls.to_csv(os.path.join(OUT, "top10_features_per_class_consensus.csv"), index=False)
    log("\nTop-5 consensus features per class (share of class-level mean|SHAP|):")
    for c, cname in enumerate(classes):
        top = df_cls[df_cls.cls == cname].head(5)
        log(f"  {cname:14s}: " + ", ".join(f"{r.feature} ({r.consensus_share*100:.1f}%)"
                                             for r in top.itertuples()))

    # ----------------------------------------- Fig 1: global top-15 bar plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    panels = [(MODEL_LABELS[k], imp_glob[k]) for k in models] + \
             [("Ensemble consensus (normalised mean over 3 models)", consensus)]
    for ax, (title, vec) in zip(axes.ravel(), panels):
        idx = np.argsort(-vec)[:15][::-1]
        ax.barh([disp(names[j]) for j in idx], vec[idx],
                color=[GROUP_COLORS[groups[j]] for j in idx], edgecolor="white")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("mean |SHAP| (avg. over 4 classes)" if "consensus" not in title
                      else "mean normalised |SHAP|")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    fig.legend(handles, GROUP_COLORS.keys(), loc="lower center", ncol=6, frameon=False)
    fig.suptitle("Global SHAP feature importance - CAXF+TF-IDF base models "
                 "(test n=3,276, seed=42)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(os.path.join(OUT, "shap_global_top15.png"), dpi=200)
    plt.close(fig)

    # ------------------------------------- Fig 2: component-group attribution
    grp_global = pd.DataFrame(
        {MODEL_LABELS[k]: [norm_glob[k][groups == g].sum() * 100 for g in GROUPS]
         for k in models}, index=GROUPS)
    grp_global["Consensus"] = [consensus[groups == g].sum() * 100 for g in GROUPS]
    grp_cls = pd.DataFrame(
        {cn: [consensus_cls[groups == g, c].sum() * 100 for g in GROUPS]
         for c, cn in enumerate(classes)}, index=GROUPS)
    grp_global.round(2).to_csv(os.path.join(OUT, "group_attribution_global.csv"))
    grp_cls.round(2).to_csv(os.path.join(OUT, "group_attribution_per_class.csv"))
    n_per_group = {g: int((groups == g).sum()) for g in GROUPS}
    log("\nShare of total mean|SHAP| by CAXF component (%):")
    log(grp_global.round(1).assign(n_features=pd.Series(n_per_group)).to_string())
    log("\nShare by CAXF component per class, consensus (%):")
    log(grp_cls.round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, df, title in [(axes[0], grp_global, "(a) Per base model"),
                          (axes[1], grp_cls, "(b) Per class (ensemble consensus)")]:
        left = np.zeros(df.shape[1])
        for g in GROUPS:
            ax.barh(df.columns, df.loc[g].values, left=left, color=GROUP_COLORS[g],
                    edgecolor="white", label=g)
            left += df.loc[g].values
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of total mean |SHAP| (%)")
        ax.set_title(title, fontweight="bold")
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in GROUPS]
    fig.legend(handles, GROUPS, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(os.path.join(OUT, "shap_group_attribution.png"), dpi=200)
    plt.close(fig)

    # ---------------------------------- Fig 3: per-class beeswarm (LightGBM)
    for c, cname in enumerate(classes):
        plt.figure(figsize=(7.2, 5.2))
        shap.summary_plot(sv["lgbm"][:, :, c], Xte, feature_names=[disp(n) for n in names],
                          max_display=12, show=False, plot_size=None)
        plt.title(f"{cname} - LightGBM SHAP beeswarm", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"shap_beeswarm_lgbm_{cname.split()[0].lower().replace('-', '')}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()

    # ------------------------------------- Cross-model agreement
    ks = list(models)
    rho = pd.DataFrame(index=[MODEL_LABELS[k] for k in ks], columns=[MODEL_LABELS[k] for k in ks], dtype=float)
    jac = rho.copy()
    for a in ks:
        for b in ks:
            rho.loc[MODEL_LABELS[a], MODEL_LABELS[b]] = spearmanr(imp_glob[a], imp_glob[b])[0]
            sa = set(np.argsort(-imp_glob[a])[:20]); sb = set(np.argsort(-imp_glob[b])[:20])
            jac.loc[MODEL_LABELS[a], MODEL_LABELS[b]] = len(sa & sb) / len(sa | sb)
    rho.round(3).to_csv(os.path.join(OUT, "agreement_spearman.csv"))
    jac.round(3).to_csv(os.path.join(OUT, "agreement_top20_jaccard.csv"))
    log("\nCross-model agreement of global importances")
    log("Spearman rho (all 5,282 features):\n" + rho.round(3).to_string())
    log("Top-20 Jaccard overlap:\n" + jac.round(3).to_string())

    # ----------------------------- Fig 4: deletion / faithfulness test
    rng = np.random.default_rng(SEED)
    ks_del = [0, 1, 2, 3, 5, 10, 20, 50, 100]
    order_cons = np.argsort(-consensus)

    def f1_after(predict, cols):
        Xp = Xte.copy()
        if len(cols):
            Xp[:, cols] = 0.0            # 0 == "feature absent" for counts / TF-IDF
        return macro_f1(y_te, predict(Xp))

    def pred_model(m):
        return lambda X: np.asarray(m.predict(X)).flatten().astype(int)

    targets = {"Stacking ensemble": (stack.predict, order_cons)}
    for k, m in models.items():
        targets[MODEL_LABELS[k]] = (pred_model(m), np.argsort(-imp_glob[k]))
    del_rows = []
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9), sharey=True)
    for ax, (tname, (pred, order)) in zip(axes, targets.items()):
        shap_curve, rand_mean, rand_std = [], [], []
        for k_ in ks_del:
            shap_curve.append(f1_after(pred, order[:k_]))
            r = [f1_after(pred, rng.choice(len(names), k_, replace=False)) for _ in range(5)]
            rand_mean.append(np.mean(r)); rand_std.append(np.std(r))
            del_rows.append(dict(target=tname, k=k_, f1_remove_top_shap=shap_curve[-1],
                                 f1_remove_random_mean=rand_mean[-1], f1_remove_random_std=rand_std[-1]))
        ax.plot(ks_del, shap_curve, "o-", color="#DC2626", label="remove top-k SHAP")
        ax.errorbar(ks_del, rand_mean, yerr=rand_std, fmt="s--", color="#64748B", label="remove random k (5 draws)")
        ax.set_title(tname, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("k features zeroed")
        ax.set_xscale("symlog", linthresh=1)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("test macro-F1")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "shap_deletion_test.png"), dpi=200)
    plt.close(fig)
    df_del = pd.DataFrame(del_rows)
    df_del.to_csv(os.path.join(OUT, "deletion_test.csv"), index=False)
    log("\nDeletion test (macro-F1 after zeroing k features; SHAP-ranked vs random):")
    log(df_del.round(4).to_string(index=False))

    # -------------------- Ensemble-level: Stacking coefficients, CPWA, PCER
    coef = stack.meta_learner.coef_                                     # (C, 3*C)
    mcols = [f"{MODEL_LABELS[k]}\nP({c.split()[0]})" for k in models for c in classes]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    im = ax.imshow(coef, cmap="RdBu_r", vmin=-np.abs(coef).max(), vmax=np.abs(coef).max(), aspect="auto")
    ax.set_xticks(range(coef.shape[1])); ax.set_xticklabels(mcols, fontsize=7.5)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=9)
    for i in range(coef.shape[0]):
        for j in range(coef.shape[1]):
            ax.text(j, i, f"{coef[i, j]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Stacking meta-learner (logistic regression) coefficients: output class (rows) "
                 "vs. base-model probability (columns)", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stacking_meta_coefficients.png"), dpi=200)
    plt.close(fig)
    pd.DataFrame(coef, index=classes, columns=[c.replace("\n", " ") for c in mcols]).round(3)\
        .to_csv(os.path.join(OUT, "stacking_meta_coefficients.csv"))

    # own-model block (coefficient of P_k(c) on logit c) summary
    own = np.array([[coef[c, k * 4 + c] for k in range(3)] for c in range(4)])
    log("\nStacking: coefficient of each base model's OWN-class probability (rows=class):")
    log(pd.DataFrame(own, index=classes, columns=[MODEL_LABELS[k] for k in models]).round(2).to_string())

    ms = [lgbm, xgb, cat]
    cpwa = BayesianModelAveraging(models=ms, n_classes=4).fit(Xtr, y_tr)
    pcer = PCER(models=ms, n_classes=4).fit(Xtr, y_tr)
    w = pd.DataFrame(cpwa.class_weights, index=classes, columns=[MODEL_LABELS[k] for k in models])
    w.round(4).to_csv(os.path.join(OUT, "cpwa_weights.csv"))
    experts = {classes[c]: MODEL_LABELS[list(models)[int(pcer.expert_table[c])]] for c in range(4)}
    log("\nCPWA class-conditional weights w_kc (in-sample precision based):")
    log(w.round(4).to_string())
    log(f"PCER expert table: {experts}")
    pd.DataFrame(pcer.class_precision, index=classes, columns=[MODEL_LABELS[k] for k in models])\
        .round(4).to_csv(os.path.join(OUT, "pcer_class_precision.csv"))

    # ------------- Fig 5: local explanations (ensemble-average attribution)
    sv_mean = np.mean([sv[k] for k in models], axis=0)                  # (n, f, C)
    base_mean = np.mean([base[k] for k in models], axis=0)              # (C,)
    txt = X_te_txt.reset_index(drop=True)
    dom = classes.index("DOM-based XSS")
    dom_idx = np.where(y_te == dom)[0]
    ok = [i for i in dom_idx if p_stack[i] == dom]
    bad = [i for i in dom_idx if p_stack[i] != dom]
    stored = classes.index("Stored XSS")
    st_ok = [i for i in np.where(y_te == stored)[0] if p_stack[i] == stored]
    picks = []
    if ok:
        picks.append(("local_dom_correct", ok[0], dom))
    if bad:
        picks.append(("local_dom_missed", bad[0], dom))
    if st_ok:
        picks.append(("local_stored_correct", st_ok[0], stored))
    log("\nLocal explanations (ensemble-average SHAP for the true class):")
    local_info = []
    for tag, i, c in picks:
        exp = shap.Explanation(values=sv_mean[i, :, c], base_values=base_mean[c],
                               data=Xte[i], feature_names=[disp(n) for n in names])
        plt.figure()
        shap.plots.waterfall(exp, max_display=10, show=False)
        plt.gcf().set_size_inches(7.6, 4.6)
        plt.savefig(os.path.join(OUT, f"shap_{tag}.png"), dpi=200, bbox_inches="tight")
        plt.close("all")
        pred_name = classes[int(p_stack[i])]
        payload = txt[i]
        local_info.append(dict(tag=tag, test_index=int(i), true=classes[c], stacking_pred=pred_name,
                               payload=payload[:300]))
        log(f"  [{tag}] test idx {i}: true={classes[c]}, Stacking predicted={pred_name}")
        log(f"      payload: {payload[:200]!r}")
        top = np.argsort(-np.abs(sv_mean[i, :, c]))[:6]
        log("      top contributors: " + ", ".join(f"{names[j]} ({sv_mean[i, j, c]:+.2f})" for j in top))
    json.dump(local_info, open(os.path.join(OUT, "local_examples.json"), "w"), indent=2)

    # save the machine-readable summary
    with open(os.path.join(OUT, "shap_results_seed_42.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    log(f"\nDone in {time.time()-t_start:.0f}s. Outputs in {OUT}")


if __name__ == "__main__":
    main()
