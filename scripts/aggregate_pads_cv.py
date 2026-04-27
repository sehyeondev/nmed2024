"""
Aggregate 5-fold CV predictions on PADS and compare to Varghese et al. (2024).

For each fold k, reads refs/pads_eval_cv/fold{k}/test_predictions.npz and:
  - Stacks per-fold test predictions into (sum(n_k), 3) arrays
     (Varghese et al. style: pooled CV predictions -> one ROC).
  - Also computes per-fold AUROC/AUPR and reports fold-mean +/- SD
     for each label.

Outputs:
    refs/pads_eval_cv/
        cv_per_fold_metrics.csv
        cv_pooled_metrics.csv
        cv_pooled_predictions.npz    (y_true, scores, label_keys)
        cv_roc_pooled.png

Also prints a side-by-side vs Varghese et al. Table 3:
    - PD vs HC (binary)            AUC 0.96
    - PD vs DD (binary)            AUC 0.89

We reproduce those binary AUCs from the 3-class model by taking the PD
score column and restricting to the relevant subsets.

Usage:
    cd C:/Users/shkim/codes/nmed2024
    python scripts/aggregate_pads_cv.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, auc)

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
CV_DIR = os.path.join(_REPO_ROOT, "refs", "pads_eval_cv")

N_FOLDS = 5
LABELS  = ["HC", "PD", "DD"]


def load_fold(k):
    path = os.path.join(CV_DIR, "fold{}".format(k), "test_predictions.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {
        "y_true":  d["y_true"].astype(int),
        "scores":  d["scores"],
        "labels":  list(d["label_keys"]),
    }


def per_fold_metrics(folds):
    rows = []
    for k, f in enumerate(folds):
        if f is None:
            rows.append({"fold": k})
            continue
        assert f["labels"] == LABELS, f["labels"]
        for i, lab in enumerate(LABELS):
            y = f["y_true"][:, i]
            s = f["scores"][:, i]
            try:
                auroc = float(roc_auc_score(y, s))
            except ValueError:
                auroc = np.nan
            try:
                aupr = float(average_precision_score(y, s))
            except ValueError:
                aupr = np.nan
            rows.append({"fold": k, "label": lab,
                         "n": int(len(y)), "pos": int(y.sum()),
                         "auroc": auroc, "aupr": aupr})
    return pd.DataFrame(rows)


def pooled_metrics(folds):
    ys = []
    ss = []
    for f in folds:
        if f is None: continue
        ys.append(f["y_true"]); ss.append(f["scores"])
    if not ys:
        return None, None, None
    y_pool = np.concatenate(ys, axis=0)
    s_pool = np.concatenate(ss, axis=0)

    rows = []
    for i, lab in enumerate(LABELS):
        y = y_pool[:, i]; s = s_pool[:, i]
        try:
            auroc = float(roc_auc_score(y, s))
        except ValueError:
            auroc = np.nan
        try:
            aupr = float(average_precision_score(y, s))
        except ValueError:
            aupr = np.nan
        rows.append({"label": lab, "n": int(len(y)),
                      "pos": int(y.sum()),
                      "auroc": auroc, "aupr": aupr})
    return y_pool, s_pool, pd.DataFrame(rows)


def pd_vs_subset(y_pool, s_pool, mask_keep, pd_col=1):
    """Reduce 3-class pooled preds to a binary PD-vs-other task
    over rows where `mask_keep` is True."""
    y = y_pool[mask_keep, pd_col]
    s = s_pool[mask_keep, pd_col]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.nan, 0, 0
    auroc = float(roc_auc_score(y, s))
    aupr  = float(average_precision_score(y, s))
    return auroc, aupr, int(y.sum()), int(len(y) - y.sum())


def plot_pooled_roc(y_pool, s_pool, out_path):
    colors = {"HC": "#3B82C4", "PD": "#E56B1F", "DD": "#C0392B"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, lab in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y_pool[:, i], s_pool[:, i])
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, color=colors[lab],
                label="{}  (AUC = {:.3f})".format(lab, a))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("5-fold CV (pooled) ROC  --  PADS (n=469)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    folds = [load_fold(k) for k in range(N_FOLDS)]
    n_have = sum(1 for f in folds if f is not None)
    print("Found predictions for {}/{} folds".format(n_have, N_FOLDS))
    if n_have == 0:
        print("Nothing to aggregate. Run scripts/run_pads_cv.py first.")
        return

    # Per-fold metrics + summary
    df_per = per_fold_metrics(folds)
    csv_per = os.path.join(CV_DIR, "cv_per_fold_metrics.csv")
    df_per.to_csv(csv_per, index=False)
    print("Saved -> {}".format(csv_per))

    if "label" in df_per.columns:
        print("\nPer-fold summary (mean +/- SD across folds):")
        for lab in LABELS:
            sub = df_per[df_per["label"] == lab]
            print("  {:3s}  AUROC {:.3f} +/- {:.3f}   AUPR {:.3f} +/- {:.3f}"
                  .format(lab,
                          sub["auroc"].mean(), sub["auroc"].std(),
                          sub["aupr"].mean(),  sub["aupr"].std()))

    # Pooled metrics
    y_pool, s_pool, df_pool = pooled_metrics(folds)
    if df_pool is not None:
        csv_pool = os.path.join(CV_DIR, "cv_pooled_metrics.csv")
        df_pool.to_csv(csv_pool, index=False)
        print("Saved -> {}".format(csv_pool))
        print("\nPooled 5-fold (n={}): ".format(y_pool.shape[0]))
        for _, r in df_pool.iterrows():
            print("  {:3s}  pos={:4d}  AUROC {:.4f}  AUPR {:.4f}"
                  .format(r["label"], r["pos"], r["auroc"], r["aupr"]))

        # Pooled npz
        npz = os.path.join(CV_DIR, "cv_pooled_predictions.npz")
        np.savez(npz, y_true=y_pool, scores=s_pool,
                 label_keys=np.array(LABELS))
        print("Saved -> {}".format(npz))

        # Pooled ROC
        roc_out = os.path.join(CV_DIR, "cv_roc_pooled.png")
        plot_pooled_roc(y_pool, s_pool, roc_out)
        print("Saved -> {}".format(roc_out))

        # --- Varghese et al. head-to-head binary tasks ----------------
        # PD vs HC: rows where HC=1 OR PD=1
        pd_col = LABELS.index("PD")
        hc_col = LABELS.index("HC")
        dd_col = LABELS.index("DD")

        keep_pd_hc = (y_pool[:, hc_col] + y_pool[:, pd_col]) > 0
        keep_pd_dd = (y_pool[:, dd_col] + y_pool[:, pd_col]) > 0

        a_hc, p_hc, pos_hc, neg_hc = pd_vs_subset(y_pool, s_pool,
                                                   keep_pd_hc, pd_col)
        a_dd, p_dd, pos_dd, neg_dd = pd_vs_subset(y_pool, s_pool,
                                                   keep_pd_dd, pd_col)

        print("\n=== PADS 5-fold pooled  vs  Varghese et al. (2024) Table 3 ===")
        print("  PD vs HC   ours AUROC={:.3f} ({} PD / {} HC)  "
              "  paper {{0.91, 0.97}} range, reported 0.96"
              .format(a_hc, pos_hc, neg_hc))
        print("  PD vs DD   ours AUROC={:.3f} ({} PD / {} DD)  "
              "  paper reported 0.89"
              .format(a_dd, pos_dd, neg_dd))

        # Also save a machine-readable summary
        summary = pd.DataFrame([
            {"task": "PD_vs_HC", "n_pos": pos_hc, "n_neg": neg_hc,
             "auroc": a_hc, "aupr": p_hc, "paper_auroc": 0.96},
            {"task": "PD_vs_DD", "n_pos": pos_dd, "n_neg": neg_dd,
             "auroc": a_dd, "aupr": p_dd, "paper_auroc": 0.89},
        ])
        csv_cmp = os.path.join(CV_DIR, "cv_pads_vs_paper.csv")
        summary.to_csv(csv_cmp, index=False)
        print("Saved -> {}".format(csv_cmp))


if __name__ == "__main__":
    main()
