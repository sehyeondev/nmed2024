"""
Compute micro / macro / weighted AUROC & AP on NACC test set,
separately for the (NC, MCI, DE) block and the 10-etiology block,
so the numbers can be placed side-by-side with Xu et al. Fig. 2 / Fig. 3.

Usage:
    cd C:/Users/shkim/codes/nmed2024
    python scripts/compute_nacc_aggregate.py
"""
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
NPZ_PATH = os.path.join(_REPO_ROOT, "refs", "nacc_eval", "test_predictions.npz")

GROUPS = {
    "NC / MCI / DE  (Fig. 2)": ["NC", "MCI", "DE"],
    "10 etiologies  (Fig. 3)": ["AD", "LBD", "VD", "PRD", "FTD",
                                 "NPH", "SEF", "PSY", "TBI", "ODE"],
    "All 13 labels":           None,  # use every label
}


def safe_metric(fn, y_true, scores, mask, average):
    """Drop rows where ANY column in the group is missing."""
    row_ok = mask.sum(axis=1) == mask.shape[1]
    y = y_true[row_ok]
    s = scores[row_ok]
    if len(y) == 0 or y.sum() == 0:
        return np.nan, 0
    try:
        return float(fn(y, s, average=average)), int(row_ok.sum())
    except ValueError:
        return np.nan, int(row_ok.sum())


def main():
    d = np.load(NPZ_PATH, allow_pickle=True)
    y_true     = d["y_true"].astype(int)
    scores     = d["scores"]
    mask       = d["mask"].astype(int)
    label_keys = list(d["label_keys"])
    idx_of = {k: i for i, k in enumerate(label_keys)}

    print("\n=== NACC test aggregate metrics  vs  Xu et al. ===")
    for name, labels in GROUPS.items():
        cols = [idx_of[k] for k in labels] if labels is not None \
               else list(range(len(label_keys)))
        yt = y_true[:, cols]
        sc = scores[:, cols]
        mk = mask[:, cols]

        row_ok = mk.sum(axis=1) == mk.shape[1]
        n_used = int(row_ok.sum())

        auc_micro,   _ = safe_metric(roc_auc_score,          yt, sc, mk, "micro")
        auc_macro,   _ = safe_metric(roc_auc_score,          yt, sc, mk, "macro")
        auc_weight,  _ = safe_metric(roc_auc_score,          yt, sc, mk, "weighted")
        ap_micro,    _ = safe_metric(average_precision_score, yt, sc, mk, "micro")
        ap_macro,    _ = safe_metric(average_precision_score, yt, sc, mk, "macro")
        ap_weight,   _ = safe_metric(average_precision_score, yt, sc, mk, "weighted")

        print("\n--- {} ---".format(name))
        print("  n (all-labeled rows used): {}".format(n_used))
        print("  AUROC    micro = {:.4f}   macro = {:.4f}   weighted = {:.4f}"
              .format(auc_micro, auc_macro, auc_weight))
        print("  AUPR     micro = {:.4f}   macro = {:.4f}   weighted = {:.4f}"
              .format(ap_micro, ap_macro, ap_weight))

    print("\n=== Paper (Xu et al. 2024, NACC held-out test) ===")
    print("  Fig. 2  NC/MCI/DE:     AUROC micro 0.94  macro 0.93  weighted 0.94")
    print("                         AUPR  micro 0.90  macro 0.84  weighted 0.87")
    print("  Fig. 3  10 etiologies: AUROC micro 0.96  macro 0.91  weighted 0.94")
    print("                         AUPR  micro 0.70  macro 0.36  weighted 0.73")


if __name__ == "__main__":
    main()
