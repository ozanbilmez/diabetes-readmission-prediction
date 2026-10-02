"""Figures for the README. All curves are computed with the from-scratch metrics."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from src.metrics import MetricsFromScratch

BACKGROUND = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_LIGHT = "#52514e"
GRAY = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order: LR, KNN, RF, ensemble
RAISES_RISK, LOWERS_RISK = "#e34948", "#2a78d6"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])

plt.rcParams.update({
    "figure.facecolor": BACKGROUND, "axes.facecolor": BACKGROUND, "savefig.facecolor": BACKGROUND,
    "axes.edgecolor": AXIS, "axes.labelcolor": TEXT_LIGHT, "axes.titlecolor": TEXT,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": TEXT_LIGHT, "ytick.labelcolor": TEXT_LIGHT,
    "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": TEXT_LIGHT,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "savefig.dpi": 150, "savefig.bbox": "tight",
})


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def plot_roc_pr(y_true, probs_by_model, title, path):
    """ROC and precision-recall curves of every model on one dataset's test set."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.6))
    prevalence = np.mean(y_true)

    for color, (name, prob) in zip(MODEL_COLORS, probs_by_model.items()):
        m = MetricsFromScratch(y_true, (prob >= 0.5).astype(int), prob)
        fpr, tpr, _ = m.roc_curve()
        precision, recall, _ = m.precision_recall_curve()
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {m.roc_auc():.3f})")
        ax_pr.plot(recall, precision, color=color, lw=2, label=f"{name} (AP {m.average_precision():.3f})")

    ax_roc.plot([0, 1], [0, 1], color=GRAY, lw=1, label="Chance")
    ax_pr.axhline(prevalence, color=GRAY, lw=1, label=f"Chance (readmission rate {prevalence:.1%})")

    ax_roc.set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate (recall)",
               xlim=(0, 1), ylim=(0, 1.01))
    ax_pr.set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision",
              xlim=(0, 1), ylim=(0, 1.01))
    ax_roc.legend(loc="lower right")
    ax_pr.legend(loc="upper right")
    fig.suptitle(f"{title} - test set", color=TEXT, fontsize=13, fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)


def plot_scratch_vs_sklearn(table, path):
    """Test ROC-AUC of each from-scratch model next to its scikit-learn counterpart."""
    rows = table.dropna(subset=["sklearn ROC-AUC"])
    datasets = rows["Dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 3.4), sharex=True, squeeze=False)

    for ax, dataset in zip(axes[0], datasets):
        part = rows[rows["Dataset"] == dataset].iloc[::-1]
        ypos = np.arange(len(part))
        # scikit-learn as a ring and our model as a dot inside it: equal scores read as a filled ring
        ax.scatter(part["sklearn ROC-AUC"], ypos, s=170, facecolor="none", edgecolor=MODEL_COLORS[1], linewidth=2,
                   zorder=2, label="scikit-learn")
        ax.scatter(part["ROC-AUC"], ypos, s=45, color=MODEL_COLORS[0], zorder=3, label="From scratch")
        ax.set_yticks(ypos, part["Model"])
        ax.set(title=dataset, xlabel="Test ROC-AUC")
        ax.set_ylim(-0.6, len(part) - 0.4)
        ax.grid(axis="y", visible=False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    _save(fig, path)


def plot_feature_importance(weights, feature_names, path, top=15):
    """Largest logistic regression coefficients. Features are standardised, so sizes are comparable."""
    order = np.argsort(np.abs(weights))[::-1][:top][::-1]
    values = weights[order]
    ypos = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(8, 0.34 * len(values) + 1.2))
    ax.barh(ypos, values, height=0.6, color=[RAISES_RISK if v > 0 else LOWERS_RISK for v in values])
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(ypos, [feature_names[i] for i in order])
    ax.set(title="Logistic regression: largest coefficients", xlabel="Coefficient (standardised features)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=RAISES_RISK, label="Raises predicted risk"),
                       Patch(color=LOWERS_RISK, label="Lowers predicted risk")], loc="lower right")
    fig.tight_layout()
    _save(fig, path)


def plot_confusion_matrix(cm, title, path):
    """Counts and row percentages (share of each actual class)."""
    shares = cm / cm.sum(axis=1, keepdims=True)
    labels = ["Not readmitted", "Readmitted"]

    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(shares, cmap=BLUES, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{shares[i, j]:.0%} of actual",
                    ha="center", va="center", fontsize=10,
                    color="white" if shares[i, j] > 0.55 else TEXT)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    _save(fig, path)
