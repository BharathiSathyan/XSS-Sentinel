"""
generate_paper_plots.py
=======================
Generates publication-quality figures for XSS Sentinel paper
from existing result text files — no model re-running needed.

Original plots (table replacements batch 1):
  1. Macro-F1 Grouped Bar Chart      -> results/macro_f1_ensemble_comparison.png  (replaces tab:ensembles)
  2. Ablation Study Bar Chart        -> results/ablation_study_chart.png           (replaces tab:ablation)
  3. Inference Time vs Macro-F1      -> results/inference_time_vs_macro_f1.png    (supplements tab:fullmetrics)
  4. Class Distribution SMOTE        -> results/class_distribution_smote.png      (replaces tab:dataset)

New plots (table replacements batch 2):
  5. Baseline Model Performance      -> results/baseline_performance.png          (replaces tab:baselines)
  6. Full Metrics Heatmap            -> results/full_metrics_heatmap.png          (replaces tab:fullmetrics)
  7. Per-Class F1 Breakdown          -> results/perclass_f1_breakdown.png         (replaces tab:perclass)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# --- Output directory ---------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Style --------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.1,
})

PALETTE = {
    "TF-IDF":  "#2563EB",
    "CharCNN": "#16A34A",
    "SentEmb": "#DC2626",
}


# =============================================================================
# FIGURE 1 — Macro-F1 Grouped Bar Chart (all 18 configurations)
# =============================================================================

def plot_macro_f1_comparison():
    ensembles = ["Stacking", "BMA", "PCER", "WSV", "CCDS", "LCCDE"]

    f1 = {
        "TF-IDF":  [0.9530, 0.9521, 0.9521, 0.9521, 0.9522, 0.9472],
        "CharCNN": [0.9528, 0.9478, 0.9487, 0.9478, 0.9260, 0.9472],
        "SentEmb": [0.9591, 0.9598, 0.9598, 0.9598, 0.9590, 0.9580],
    }

    x = np.arange(len(ensembles))
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#F9FAFB")
    ax.set_facecolor("#F9FAFB")

    for i, (feat, color) in enumerate(PALETTE.items()):
        bars = ax.bar(x + offsets[i], f1[feat], width,
                      label=feat, color=color, alpha=0.88,
                      edgecolor="white", linewidth=0.6, zorder=3)
        for bar, val in zip(bars, f1[feat]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0008,
                    f"{val*100:.2f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color="#1E293B", rotation=90)

    ax.axhline(0.9472, color="#9CA3AF", linewidth=1.2,
               linestyle="--", zorder=2, label="LCCDE baseline (TF-IDF)")

    ax.set_ylim(0.90, 0.978)
    ax.set_ylabel("Macro-F1 Score")
    ax.set_xlabel("Ensemble Strategy")
    ax.set_title("Macro-F1 Comparison: All 6 Ensemble Strategies x 3 Feature Representations\n"
                 "(seed=42, test n=3,276)")
    ax.set_xticks(x)
    ax.set_xticklabels(ensembles, fontweight="semibold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="#CBD5E1")

    ax.annotate("Best overall:\nBMA/PCER/WSV + SentEmb (95.98%)",
                xy=(4 + offsets[2], 0.9598), xytext=(4.45, 0.962),
                fontsize=8.5, color="#DC2626",
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DC2626", alpha=0.85))

    out = os.path.join(RESULTS_DIR, "macro_f1_ensemble_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[1/4] Saved: {out}")


# =============================================================================
# FIGURE 2 — Ablation Study: Strict LCCDE vs MVE-CT
# =============================================================================

def plot_ablation_study():
    features   = ["CAXF+TF-IDF", "CAXF+CharCNN", "CAXF+SentEmb"]
    strict_f1  = [0.9630, 0.9505, 0.9580]
    mvect_f1   = [0.9521, 0.9641, 0.9580]
    strict_dom = [0.8889, 0.8421, 0.8750]
    mvect_dom  = [0.8421, 0.8889, 0.8750]

    x     = np.arange(len(features))
    width = 0.18

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#F9FAFB")

    colors_strict = "#2563EB"
    colors_mve    = "#F97316"

    for ax in (ax1, ax2):
        ax.set_facecolor("#F9FAFB")
        ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

    # Panel A: Macro F1
    b1 = ax1.bar(x - width/2, strict_f1, width, label="Strict LCCDE (OOF)",
                 color=colors_strict, alpha=0.88, edgecolor="white", zorder=3)
    b2 = ax1.bar(x + width/2, mvect_f1,  width, label="MVE-CT (heuristic)",
                 color=colors_mve,    alpha=0.88, edgecolor="white", zorder=3)

    for bars, vals in [(b1, strict_f1), (b2, mvect_f1)]:
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.0005,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=8.5)

    ax1.set_ylim(0.93, 0.975)
    ax1.set_ylabel("Macro-F1 Score")
    ax1.set_title("(a) Macro-F1: Strict LCCDE vs. MVE-CT")
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, fontweight="semibold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax1.legend(framealpha=0.9)

    deltas       = ["-1.09 pp\n(Strict wins)", "+1.36 pp\n(MVE-CT wins)", "0.00 pp\n(Tie)"]
    delta_colors = ["#16A34A", "#DC2626", "#6B7280"]
    for i, (d, c) in enumerate(zip(deltas, delta_colors)):
        ax1.text(i, 0.932, d, ha="center", va="bottom",
                 fontsize=8, color=c, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=c, alpha=0.85))

    # Panel B: DOM F1
    b3 = ax2.bar(x - width/2, strict_dom, width, label="Strict LCCDE (OOF)",
                 color=colors_strict, alpha=0.88, edgecolor="white", zorder=3)
    b4 = ax2.bar(x + width/2, mvect_dom,  width, label="MVE-CT (heuristic)",
                 color=colors_mve,    alpha=0.88, edgecolor="white", zorder=3)

    for bars, vals in [(b3, strict_dom), (b4, mvect_dom)]:
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.002,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=8.5)

    ax2.set_ylim(0.78, 0.94)
    ax2.set_ylabel("DOM-based XSS F1 Score")
    ax2.set_title("(b) DOM-class F1: Strict LCCDE vs. MVE-CT")
    ax2.set_xticks(x)
    ax2.set_xticklabels(features, fontweight="semibold")
    ax2.legend(framealpha=0.9)

    fig.suptitle("Ablation Study: Strict LCCDE (OOF Leaders) vs. MVE-CT (Heuristic Leaders)\n"
                 "(seed=42, test n=3,276)",
                 fontsize=13, fontweight="bold", y=1.02)

    out = os.path.join(RESULTS_DIR, "ablation_study_chart.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[2/4] Saved: {out}")


# =============================================================================
# FIGURE 3 — Inference Time vs Macro-F1 Scatter
# =============================================================================

def plot_inference_scatter():
    data = [
        # (label,               macro_f1, inf_time, feature)
        ("Stacking+TF-IDF",     0.9530,   0.41,  "TF-IDF",  "o"),
        ("BMA+TF-IDF",          0.9521,   0.81,  "TF-IDF",  "s"),
        ("PCER+TF-IDF",         0.9521,   0.31,  "TF-IDF",  "^"),
        ("WSV+TF-IDF",          0.9521,   0.45,  "TF-IDF",  "D"),
        ("CCDS+TF-IDF",         0.9522,   2.97,  "TF-IDF",  "P"),
        ("LCCDE+TF-IDF",        0.9472,   0.21,  "TF-IDF",  "X"),
        ("Stacking+CharCNN",    0.9528,   0.08,  "CharCNN", "o"),
        ("BMA+CharCNN",         0.9478,   0.08,  "CharCNN", "s"),
        ("PCER+CharCNN",        0.9487,   0.21,  "CharCNN", "^"),
        ("WSV+CharCNN",         0.9478,   0.08,  "CharCNN", "D"),
        ("CCDS+CharCNN",        0.9260,   1.78,  "CharCNN", "P"),
        ("LCCDE+CharCNN",       0.9472,   0.21,  "CharCNN", "X"),
        ("Stacking+SentEmb",    0.9591,   0.09,  "SentEmb", "o"),
        ("BMA+SentEmb",         0.9598,   0.11,  "SentEmb", "s"),
        ("PCER+SentEmb",        0.9598,   0.22,  "SentEmb", "^"),
        ("WSV+SentEmb",         0.9598,   0.09,  "SentEmb", "D"),
        ("CCDS+SentEmb",        0.9590,   1.52,  "SentEmb", "P"),
        ("LCCDE+SentEmb",       0.9580,   0.23,  "SentEmb", "X"),
    ]

    # Labels to annotate explicitly
    annotate_set = {
        "Stacking+TF-IDF", "BMA+SentEmb", "PCER+CharCNN", "CCDS+TF-IDF", "CCDS+CharCNN"
    }

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor("#F9FAFB")
    ax.set_facecolor("#F9FAFB")

    ax.axvspan(0, 0.5, color="#BBF7D0", alpha=0.22, zorder=1)
    ax.text(0.25, 0.9055, "Fast & Accurate\nZone", ha="center", fontsize=9,
            color="#166534", style="italic")

    plotted_feat = {}
    plotted_mk   = {}

    mk_labels = {"o": "Stacking", "s": "BMA", "^": "PCER",
                 "D": "WSV",      "P": "CCDS", "X": "LCCDE"}

    for label, f1, t, feat, mk in data:
        c = PALETTE[feat]
        ax.scatter(t, f1, c=c, marker=mk, s=115,
                   edgecolors="white", linewidths=0.8, zorder=4, alpha=0.92)
        if feat not in plotted_feat:
            plotted_feat[feat] = mpatches.Patch(color=c, label=feat)
        if mk not in plotted_mk:
            plotted_mk[mk] = plt.scatter([], [], marker=mk, c="#6B7280",
                                         s=80, label=mk_labels[mk])
        if label in annotate_set:
            xoff = 0.07 if t < 2.5 else -0.5
            ax.annotate(label, xy=(t, f1),
                        xytext=(t + xoff, f1 + 0.001),
                        fontsize=8, color="#1E293B",
                        arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=0.8))

    feat_legend = ax.legend(handles=list(plotted_feat.values()),
                            title="Feature", loc="lower right", framealpha=0.9)
    ax.add_artist(feat_legend)

    mk_legend = ax.legend(handles=list(plotted_mk.values()),
                          title="Ensemble", loc="lower center", ncol=3,
                          framealpha=0.9, bbox_to_anchor=(0.5, -0.24))
    ax.add_artist(mk_legend)
    ax.add_artist(feat_legend)

    ax.set_xlabel("Inference Time on 3,276 Test Samples (seconds)")
    ax.set_ylabel("Macro-F1 Score")
    ax.set_title("Speed-Accuracy Tradeoff: Inference Time vs. Macro-F1\n"
                 "for All 18 Configurations (seed=42, test n=3,276)")
    ax.set_xlim(-0.1, 3.2)
    ax.set_ylim(0.900, 0.975)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.grid(linestyle="--", alpha=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

    out = os.path.join(RESULTS_DIR, "inference_time_vs_macro_f1.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[3/4] Saved: {out}")


# =============================================================================
# FIGURE 4 — Class Distribution Before & After SMOTE
# =============================================================================

def plot_smote_distribution():
    classes      = ["DOM-based XSS", "Normal", "Reflected XSS", "Stored XSS"]
    before_full  = [30,   3594, 6795,  498]
    before_train = [21,   2515, 4756,  349]
    after_smote  = [4756, 4756, 4756, 4756]

    x     = np.arange(len(classes))
    width = 0.28

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#F9FAFB")

    for ax in (ax1, ax2):
        ax.set_facecolor("#F9FAFB")
        ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

    # Panel A: original distribution (log scale to show the severe imbalance)
    b1 = ax1.bar(x - width/2, before_full,  width, label="Full Dataset (10,917)",
                 color="#2563EB", alpha=0.85, edgecolor="white", zorder=3)
    b2 = ax1.bar(x + width/2, before_train, width, label="Train Split (7,641)",
                 color="#F97316", alpha=0.85, edgecolor="white", zorder=3)

    for bars, vals in [(b1, before_full), (b2, before_train)]:
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() * 1.08, str(v),
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax1.set_yscale("log")
    ax1.set_ylim(10, 20000)
    ax1.set_title("(a) Original Class Distribution\n(Severe Imbalance — IR ~226.5x)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, fontsize=9)
    ax1.set_ylabel("Sample Count (log scale)")
    ax1.legend(framealpha=0.9)
    ax1.annotate("Only 21 training\nsamples for DOM!",
                 xy=(0, 21), xytext=(0.7, 80),
                 fontsize=8.5, color="#DC2626", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.3),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DC2626", alpha=0.9))

    # Panel B: after SMOTE
    smote_colors = ["#7C3AED", "#16A34A", "#0891B2", "#EA580C"]
    bars = ax2.bar(x, after_smote, 0.55, color=smote_colors,
                   alpha=0.85, edgecolor="white", zorder=3)

    for bar, v in zip(bars, after_smote):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 30, str(v),
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax2.axhline(4756, color="#1E293B", linewidth=1.2, linestyle="--",
                zorder=2, label="Target (4,756 per class)")
    ax2.set_title("(b) After SMOTE Oversampling\n(Perfectly Balanced — IR = 1.0)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontsize=9)
    ax2.set_ylabel("Sample Count")
    ax2.set_ylim(0, 5900)
    ax2.legend(framealpha=0.9)
    ax2.text(1.5, 5350,
             "All 4 classes = 4,756 samples\n(19,024 total post-SMOTE)",
             ha="center", fontsize=8.5, color="#166534",
             bbox=dict(boxstyle="round,pad=0.4", fc="#DCFCE7", ec="#16A34A", alpha=0.9))

    fig.suptitle("Dataset Class Distribution: Before and After SMOTE Balancing\n"
                 "(LightGBM & XGBoost: SMOTE on 7,641 train; CatBoost: class weights on original)",
                 fontsize=13, fontweight="bold")

    out = os.path.join(RESULTS_DIR, "class_distribution_smote.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[4/4] Saved: {out}")


# =============================================================================
# FIGURE 5 — Baseline Model Performance Grouped Bar Chart
# Replaces: tab:baselines
# =============================================================================

def plot_baseline_performance():
    models   = ["LightGBM", "XGBoost", "CatBoost"]
    features = ["TF-IDF (5282-dim)", "CharCNN (410-dim)", "SentEmb (666-dim)"]

    # macro-F1 values from result txt files
    macro_f1 = [
        [0.9503, 0.9188, 0.8601],   # TF-IDF  : LGB, XGB, CAT
        [0.9628, 0.9613, 0.9086],   # CharCNN : LGB, XGB, CAT
        [0.9580, 0.9767, 0.9103],   # SentEmb : LGB, XGB, CAT
    ]
    # macro-precision
    macro_prec = [
        [0.9392, 0.8860, 0.8192],
        [0.9678, 0.9921, 0.8863],
        [0.9895, 0.9919, 0.8870],
    ]
    # macro-recall
    macro_rec = [
        [0.9625, 0.9688, 0.9258],
        [0.9581, 0.9374, 0.9324],
        [0.9335, 0.9631, 0.9267],
    ]

    x     = np.arange(len(models))
    width = 0.22
    feat_colors = ["#2563EB", "#16A34A", "#DC2626"]
    metric_alpha = [1.0, 0.6, 0.35]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    fig.patch.set_facecolor("#F9FAFB")

    metrics     = ["Macro-F1",  "Macro-Precision", "Macro-Recall"]
    metric_data = [macro_f1,     macro_prec,         macro_rec]
    offsets     = [-width, 0, width]

    for col, (ax, feat, color) in enumerate(zip(axes, features, feat_colors)):
        ax.set_facecolor("#F9FAFB")
        ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

        for mi, (metric, mdata, alpha) in enumerate(zip(metrics, metric_data, metric_alpha)):
            vals = mdata[col]
            bars = ax.bar(x + offsets[mi], vals, width,
                          label=metric, color=color, alpha=alpha,
                          edgecolor="white", linewidth=0.5, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.003,
                        f"{v*100:.1f}",
                        ha="center", va="bottom", fontsize=7.5, rotation=90)

        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9.5)
        ax.set_ylim(0.80, 1.01)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        if col == 0:
            ax.set_ylabel("Score")

    # single legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#555", alpha=1.0,  label="Macro-F1"),
        Patch(facecolor="#555", alpha=0.6,  label="Macro-Precision"),
        Patch(facecolor="#555", alpha=0.35, label="Macro-Recall"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Baseline Model Performance: LightGBM, XGBoost, CatBoost\n"
                 "across Three CAXF Feature Representations (seed=42, test n=3,276)",
                 fontsize=13, fontweight="bold")

    out = os.path.join(RESULTS_DIR, "baseline_performance.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[5/7] Saved: {out}")


# =============================================================================
# FIGURE 6 — Full Metrics Heatmap (all 18 configurations)
# Replaces: tab:fullmetrics
# =============================================================================

def plot_full_metrics_heatmap():
    import matplotlib.colors as mcolors

    # Rows: feature+ensemble, Columns: Acc, M-Prec, M-Rec, M-F1, Inf(s)
    row_labels = [
        "TF-IDF · Stacking",  "TF-IDF · CCDS",  "TF-IDF · BMA",
        "TF-IDF · PCER",      "TF-IDF · WSV",   "TF-IDF · LCCDE",
        "CharCNN · Stacking", "CharCNN · PCER",  "CharCNN · BMA",
        "CharCNN · WSV",      "CharCNN · LCCDE", "CharCNN · CCDS",
        "SentEmb · BMA",      "SentEmb · PCER",  "SentEmb · WSV",
        "SentEmb · Stacking", "SentEmb · CCDS",  "SentEmb · LCCDE",
    ]
    col_labels = ["Accuracy (%)", "Macro-Prec (%)", "Macro-Rec (%)", "Macro-F1 (%)", "Inf. Time (s)"]

    data = np.array([
        # Acc    M-Prec  M-Rec  M-F1   Inf
        [99.69,  93.97,  96.76, 95.30,  0.41],
        [99.66,  93.97,  96.59, 95.22,  2.97],
        [99.66,  94.10,  96.44, 95.21,  0.81],
        [99.66,  94.10,  96.44, 95.21,  0.31],
        [99.63,  94.10,  96.44, 95.21,  0.45],
        [99.60,  95.62,  93.98, 94.72,  0.21],
        [99.42,  97.38,  93.99, 95.28,  0.08],
        [99.66,  96.35,  93.55, 94.87,  0.21],
        [99.63,  96.18,  93.53, 94.78,  0.08],
        [99.63,  96.18,  93.53, 94.78,  0.08],
        [99.60,  95.62,  93.98, 94.72,  0.21],
        [99.57,  91.67,  93.64, 92.60,  1.78],
        [99.63,  99.29,  93.38, 95.98,  0.11],
        [99.63,  99.29,  93.38, 95.98,  0.22],
        [99.63,  99.29,  93.38, 95.98,  0.09],
        [99.66,  98.28,  94.29, 95.91,  0.09],
        [99.60,  99.12,  93.37, 95.90,  1.52],
        [99.57,  98.95,  93.35, 95.80,  0.23],
    ])

    # Separate normalisation per column (Inf time: lower=better so invert)
    norm_data = np.zeros_like(data)
    for c in range(data.shape[1]):
        col = data[:, c]
        lo, hi = col.min(), col.max()
        if c == 4:  # inference time: invert
            norm_data[:, c] = 1.0 - (col - lo) / (hi - lo + 1e-9)
        else:
            norm_data[:, c] = (col - lo) / (hi - lo + 1e-9)

    fig, ax = plt.subplots(figsize=(11, 10))
    fig.patch.set_facecolor("#F9FAFB")
    ax.set_facecolor("#F9FAFB")

    cmap = plt.cm.RdYlGn
    im = ax.imshow(norm_data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # Row separator lines between feature groups
    for sep in [5.5, 11.5]:
        ax.axhline(sep, color="#1E293B", linewidth=2.0)

    # Feature group labels on left margin
    group_labels = {2: "CAXF+TF-IDF", 8: "CAXF+CharCNN", 14: "CAXF+SentEmb"}
    for row_idx, lbl in group_labels.items():
        ax.text(-1.55, row_idx, lbl, ha="right", va="center",
                fontsize=9.5, fontweight="bold", color="#1E293B",
                rotation=90)

    # Cell text
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            txt = f"{val:.2f}" if c < 4 else f"{val:.2f}s"
            brightness = norm_data[r, c]
            fc = "white" if brightness < 0.35 or brightness > 0.75 else "#1E293B"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=8.5, color=fc, fontweight="semibold")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.tick_params(axis="both", length=0)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Relative Performance (green=best, red=worst)\n[Inf. time: green=fastest]",
                   fontsize=9)
    cbar.set_ticks([])

    ax.set_title("Full Metrics Heatmap: All 18 Configurations\n"
                 "(seed=42, test n=3,276  |  Inf. time colour-coded: green=faster)",
                 fontsize=12, fontweight="bold", pad=14)

    out = os.path.join(RESULTS_DIR, "full_metrics_heatmap.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[6/7] Saved: {out}")


# =============================================================================
# FIGURE 7 — Per-Class F1 Breakdown (CAXF+TF-IDF, all 6 ensembles)
# Replaces: tab:perclass
# =============================================================================

def plot_perclass_f1():
    ensembles = ["Stacking", "CCDS", "BMA", "PCER", "WSV", "LCCDE"]
    classes   = ["DOM-based\nXSS (9)", "Normal\n(1,079)", "Reflected\nXSS (2,039)", "Stored\nXSS (149)"]

    # F1 per ensemble × class  (from ensemble_comparison_seed_42.txt, TF-IDF)
    f1_data = np.array([
        # DOM   Normal  Reflected  Stored
        [0.84,  1.00,   1.00,      0.97],   # Stacking
        [0.84,  1.00,   1.00,      0.97],   # CCDS
        [0.84,  1.00,   1.00,      0.97],   # BMA
        [0.84,  1.00,   1.00,      0.97],   # PCER
        [0.84,  1.00,   1.00,      0.97],   # WSV
        [0.82,  1.00,   1.00,      0.97],   # LCCDE
    ])

    x     = np.arange(len(classes))
    width = 0.13
    ens_colors = ["#2563EB", "#7C3AED", "#16A34A", "#EA580C", "#0891B2", "#9CA3AF"]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("#F9FAFB")
    ax.set_facecolor("#F9FAFB")

    n = len(ensembles)
    offsets = np.linspace(-(n//2)*width, (n//2)*width, n)

    for i, (ens, color, offset) in enumerate(zip(ensembles, ens_colors, offsets)):
        bars = ax.bar(x + offset, f1_data[i], width,
                      label=ens, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, v in zip(bars, f1_data[i]):
            if v < 0.999:   # only annotate non-trivial values
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.002,
                        f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7.5, rotation=90)

    # Perfect score reference line
    ax.axhline(1.00, color="#16A34A", linewidth=1.2,
               linestyle="--", zorder=2, label="Perfect F1=1.00")

    # DOM bottleneck annotation
    ax.annotate("DOM-based XSS:\nUniversal bottleneck\n(only 9 test samples)",
                xy=(0, 0.845), xytext=(0.55, 0.875),
                fontsize=8.5, color="#DC2626", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DC2626", alpha=0.9))

    ax.set_ylim(0.78, 1.025)
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Class")
    ax.set_title("Per-Class F1 Breakdown — CAXF+TF-IDF, All 6 Ensemble Strategies\n"
                 "(seed=42, test n=3,276  |  test support shown in parentheses)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", framealpha=0.9, ncol=2)

    out = os.path.join(RESULTS_DIR, "perclass_f1_breakdown.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[7/7] Saved: {out}")


# =============================================================================
if __name__ == "__main__":
    print("Generating XSS Sentinel paper figures...\n")
    plot_macro_f1_comparison()
    plot_ablation_study()
    plot_inference_scatter()
    plot_smote_distribution()
    plot_baseline_performance()
    plot_full_metrics_heatmap()
    plot_perclass_f1()
    print("\nAll 7 figures saved to results/")
