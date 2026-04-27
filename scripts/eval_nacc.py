"""
Evaluate the reproduced NACC Transformer (Xu et al. 2024) on the held-out test set.

Loads the checkpoint produced by dev/train.sh, runs inference on
nacc_test_with_np_cli.csv (~9,048 subjects), and reports:
  - Per-class metrics for all 13 dementia labels (AUC-ROC, AUC-PR, Bal.Acc,
    Sens, Spec, F1) at thr=0.5 and at Youden-optimal threshold
  - Mean AUC-ROC and Mean AUC-PR
  - CSV dump, ROC-curve grid, confusion-matrix grid

Usage:
    cd C:/Users/shkim/codes/nmed2024
    conda activate viterbi_net
    python scripts/eval_nacc.py
"""
import argparse
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)

# PyTorch 2.6+ compat: allow loading checkpoints that include optimizer state
_orig_torch_load = torch.load
def _patched_torch_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _patched_torch_load

from dev.data.dataset_csv import CSVDataset
from adrd.model import ADRDModel
from adrd.utils.misc import get_metrics_multitask


def parse_args():
    p = argparse.ArgumentParser(description="NACC Transformer test-set evaluation")
    p.add_argument("--test_path",
                   default="data/train_vld_test_split_updated/nacc_test_with_np_cli.csv")
    p.add_argument("--cnf_file",
                   default="dev/data/toml_files/default_conf_new.toml")
    p.add_argument("--ckpt",
                   default="dev/ckpt/debug/model.pt",
                   help="Path to NACC checkpoint (best-val AUROC by default)")
    p.add_argument("--out_dir",
                   default="refs/nacc_eval",
                   help="Where to write metrics CSV and plots")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--cuda_idx", default=0, type=int)
    return p.parse_args()


def dict_list_to_array(lst, keys, fill=np.nan):
    """Convert a list of {label: value} dicts into (N, K). Missing -> fill."""
    out = np.full((len(lst), len(keys)), fill, dtype=float)
    for i, d in enumerate(lst):
        for j, k in enumerate(keys):
            v = d.get(k, None)
            if v is None:
                continue
            try:
                if isinstance(v, float) and np.isnan(v):
                    continue
            except Exception:
                pass
            out[i, j] = float(v)
    return out


def print_table(title, met, label_keys, metric_names):
    print("\n" + "=" * (28 + 11 * len(label_keys)))
    print(title)
    print("=" * (28 + 11 * len(label_keys)))
    header = "{:22s}".format("") + "".join("{:>11s}".format(k) for k in label_keys)
    print(header)
    print("-" * len(header))
    for mk in metric_names:
        row = [met[i].get(mk, np.nan) for i in range(len(label_keys))]
        print("{:22s}".format(mk) +
              "".join("{:>11.4f}".format(v) for v in row))


def save_metrics_csv(path, met, label_keys, metric_names, extras=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write("metric," + ",".join(label_keys) + "\n")
        for mk in metric_names:
            row = [met[i].get(mk, np.nan) for i in range(len(label_keys))]
            f.write(mk + "," + ",".join("{:.4f}".format(v) for v in row) + "\n")
        if extras:
            f.write("\n")
            for k, v in extras.items():
                f.write("{},{:.4f}\n".format(k, v))


def plot_roc_grid(y_true, scores, mask, label_keys, out_path, title, ncols=5):
    """Grid of per-class ROC curves (13 labels -> 3x5)."""
    n = len(label_keys)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, k in enumerate(label_keys):
        ax = axes[i]
        m = mask[:, i].astype(bool)
        y = y_true[m, i].astype(int)
        s = scores[m, i]
        if len(np.unique(y)) < 2:
            ax.text(0.5, 0.5, "no positives", ha="center", va="center")
            ax.set_title(k)
            continue
        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.8, color="#3B82C4")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
        ax.set_title("{}  (AUC={:.3f})".format(k, roc_auc), fontsize=10)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_grid(y_true, y_pred, mask, label_keys, out_path, title, ncols=5):
    n = len(label_keys)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, k in enumerate(label_keys):
        ax = axes[i]
        m = mask[:, i].astype(bool)
        y = y_true[m, i].astype(int)
        p = y_pred[m, i].astype(int)
        cm = confusion_matrix(y, p, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm,
                display_labels=["not-{}".format(k), k])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("{}  (pos={})".format(k, int(y.sum())), fontsize=9)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontsize(7)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading test dataset from {} ...".format(args.test_path))
    dat_tst = CSVDataset(
        dat_file=args.test_path,
        cnf_file=args.cnf_file,
        mode=2,
        img_mode=-1,
        arch="NonImg",
        transforms=None,
        stripped=None,
    )
    n_test = len(dat_tst.features)
    print("Test samples: {}".format(n_test))

    print("Loading checkpoint: {}".format(args.ckpt))
    mdl = ADRDModel(None, None, None,
                    device=args.device,
                    cuda_devices=[args.cuda_idx])
    if args.device == "cuda":
        mdl.device = "cuda:{}".format(args.cuda_idx)
    else:
        mdl.device = "cpu"
    print("Target device: {}".format(mdl.device))
    mdl.load(args.ckpt, map_location=mdl.device)

    label_keys = list(mdl.tgt_modalities.keys())
    print("Target labels ({}): {}".format(len(label_keys), label_keys))

    print("Running inference on test set ...")
    _, proba, preds_05 = mdl.predict(dat_tst.features)

    scores     = dict_list_to_array(proba, label_keys)
    y_pred_05  = dict_list_to_array(preds_05, label_keys, fill=0).astype(int)
    y_true_raw = dict_list_to_array(dat_tst.labels, label_keys)
    # Mask: 1 where label is available (non-nan), 0 where missing
    mask = (~np.isnan(y_true_raw)).astype(int)
    y_true = np.where(np.isnan(y_true_raw), 0, y_true_raw).astype(int)

    # Per-class label balance
    print("\nTest-set per-class counts  (pos / neg / missing):")
    for i, k in enumerate(label_keys):
        m = mask[:, i].astype(bool)
        pos = int(y_true[m, i].sum())
        neg = int(m.sum() - pos)
        miss = int(n_test - m.sum())
        print("  {:4s}  pos = {:5d}  neg = {:5d}  missing = {:5d}".format(
            k, pos, neg, miss))

    # ---- Metrics @ thr=0.5 ---------------------------------------------------
    met_05 = get_metrics_multitask(y_true, y_pred_05, scores, mask)
    metric_names = [
        "AUC (ROC)", "AUC (PR)",
        "Balanced Accuracy",
        "Sensitivity/Recall", "Specificity",
        "Precision", "F1 score",
    ]
    print_table("Test metrics  --threshold = 0.5",
                met_05, label_keys, metric_names)

    mean_auc_roc = float(np.nanmean([met_05[i]["AUC (ROC)"] for i in range(len(label_keys))]))
    mean_auc_pr  = float(np.nanmean([met_05[i]["AUC (PR)"]  for i in range(len(label_keys))]))
    print("\nMean AUC-ROC : {:.4f}".format(mean_auc_roc))
    print("Mean AUC-PR  : {:.4f}".format(mean_auc_pr))

    # ---- Metrics @ Youden-optimal ------------------------------------------
    thresholds_opt = {}
    y_pred_opt = np.zeros_like(y_pred_05)
    for i, k in enumerate(label_keys):
        m = mask[:, i].astype(bool)
        y = y_true[m, i]
        s = scores[m, i]
        if len(np.unique(y)) < 2:
            thresholds_opt[k] = 0.5
        else:
            fpr, tpr, thr = roc_curve(y, s)
            youden = tpr - fpr
            thresholds_opt[k] = float(thr[int(np.argmax(youden))])
        y_pred_opt[:, i] = (scores[:, i] > thresholds_opt[k]).astype(int)

    met_opt = get_metrics_multitask(y_true, y_pred_opt, scores, mask)
    print_table("Test metrics  --Youden-optimal threshold per class",
                met_opt, label_keys, metric_names)
    print("\nOptimal thresholds:")
    for k, thr in thresholds_opt.items():
        print("  {:4s}  {:.4f}".format(k, thr))

    # ---- Save ---------------------------------------------------------------
    csv_05 = os.path.join(args.out_dir, "test_metrics_thr05.csv")
    save_metrics_csv(csv_05, met_05, label_keys, metric_names,
                     extras={"mean_auc_roc": mean_auc_roc,
                             "mean_auc_pr":  mean_auc_pr})
    print("\nSaved -> {}".format(csv_05))

    csv_opt = os.path.join(args.out_dir, "test_metrics_youden.csv")
    save_metrics_csv(csv_opt, met_opt, label_keys, metric_names,
                     extras={"thr_{}".format(k): thresholds_opt[k]
                             for k in label_keys})
    print("Saved -> {}".format(csv_opt))

    roc_path = os.path.join(args.out_dir, "roc_curves.png")
    plot_roc_grid(y_true, scores, mask, label_keys, roc_path,
                  "NACC test-set ROC curves  (n={})".format(n_test))
    print("Saved -> {}".format(roc_path))

    cm_05_path = os.path.join(args.out_dir, "confusion_matrices_thr05.png")
    plot_confusion_grid(y_true, y_pred_05, mask, label_keys, cm_05_path,
                        "NACC Confusion Matrices  (threshold = 0.5)")
    print("Saved -> {}".format(cm_05_path))

    cm_opt_path = os.path.join(args.out_dir, "confusion_matrices_youden.png")
    plot_confusion_grid(y_true, y_pred_opt, mask, label_keys, cm_opt_path,
                        "NACC Confusion Matrices  (Youden-optimal)")
    print("Saved -> {}".format(cm_opt_path))

    np.savez(os.path.join(args.out_dir, "test_predictions.npz"),
             y_true=y_true, scores=scores, mask=mask,
             y_pred_05=y_pred_05, y_pred_opt=y_pred_opt,
             label_keys=np.array(label_keys),
             thresholds_opt=np.array([thresholds_opt[k] for k in label_keys]))
    print("Saved -> {}/test_predictions.npz".format(args.out_dir))

    print("\nDone.")


if __name__ == "__main__":
    main()
