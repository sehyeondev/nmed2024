"""
Re-plot NACC ROC curves for only the 4 primary labels (NC, MCI, DE, AD).

Loads the saved predictions produced by eval_nacc.py and writes a compact
2x2 ROC grid to refs/nacc_eval/roc_curves_primary.png.

Usage:
    cd C:/Users/shkim/codes/nmed2024
    python scripts/plot_nacc_primary_roc.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

NPZ_PATH = os.path.join(_REPO_ROOT, "refs", "nacc_eval", "test_predictions.npz")
OUT_PATH = os.path.join(_REPO_ROOT, "refs", "nacc_eval", "roc_curves_primary.png")

PRIMARY = [
    ("NC",  "Normal Cognition"),
    ("MCI", "Mild Cognitive Impairment"),
    ("DE",  "Dementia  (overall)"),
    ("AD",  "Alzheimer's Disease"),
]

CURVE_COLOR = "#3B82C4"   # match slide-deck blue


def main():
    d = np.load(NPZ_PATH, allow_pickle=True)
    y_true     = d["y_true"]
    scores     = d["scores"]
    mask       = d["mask"]
    label_keys = list(d["label_keys"])

    # Map label -> column index
    idx_of = {k: i for i, k in enumerate(label_keys)}

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.4))
    axes = axes.reshape(-1)

    for ax, (key, pretty) in zip(axes, PRIMARY):
        i = idx_of[key]
        m = mask[:, i].astype(bool)
        y = y_true[m, i].astype(int)
        s = scores[m, i]
        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = auc(fpr, tpr)
        pos = int(y.sum())
        neg = int(len(y) - pos)

        ax.plot(fpr, tpr, linewidth=2.2, color=CURVE_COLOR,
                label="AUC = {:.3f}".format(roc_auc))
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
        ax.fill_between(fpr, tpr, alpha=0.08, color=CURVE_COLOR)

        ax.set_title("{}  -- {}".format(key, pretty), fontsize=11,
                     fontweight="bold")
        ax.set_xlabel("1 - Specificity", fontsize=9)
        ax.set_ylabel("Sensitivity", fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.01])
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.legend(loc="lower right", fontsize=10, frameon=True)

        # Annotate class balance
        ax.text(0.98, 0.05,
                "pos = {}   neg = {}".format(pos, neg),
                transform=ax.transAxes, fontsize=8, color="#666666",
                ha="right", va="bottom")

    fig.suptitle("NACC test-set ROC  --  primary dementia labels",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=160)
    plt.close(fig)

    print("Saved -> {}".format(OUT_PATH))


if __name__ == "__main__":
    main()
