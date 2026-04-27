"""
Evaluate the trained PADS Transformer on the held-out test set.

Loads a checkpoint produced by scripts/train_pads.py, runs inference on
test.csv (94 subjects), and reports:
  - Per-class metrics (AUC-ROC, AUC-PR, Bal.Acc, Sens, Spec, F1) at thr=0.5
  - The same metrics at the Youden-optimal threshold per class
  - Mean AUC-ROC and Mean AUC-PR
  - CSV dump of all numbers
  - ROC-curve plot (3 classes overlaid)
  - Confusion-matrix plots (one per class) at both thresholds

Usage:
    cd <repo root>
    conda activate viterbi_net
    python scripts/eval_pads.py

    # or with custom paths:
    python scripts/eval_pads.py \
        --ckpt data/pads/checkpoints/pads_transformer.pt \
        --out_dir refs/pads_eval
"""
import argparse
import os
import sys

# Make repo root importable (so `adrd` and `dev.data` work regardless of cwd)
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

# --- Compat: PyTorch 2.6+ defaults torch.load(weights_only=True), but the
# training code saves the optimizer object (AdamW) inside the checkpoint.
# We trust our own checkpoint, so force weights_only=False for all
# torch.load calls inside the adrd package.
_orig_torch_load = torch.load
def _patched_torch_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _patched_torch_load

from dev.data.dataset_csv import CSVDataset
from adrd.model import ADRDModel
from adrd.utils.misc import get_metrics_multitask


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PADS Transformer test-set evaluation")
    p.add_argument("--data_path",
                   default="data/pads/converted_pads.csv",
                   help="(unused by eval, kept for parity with train_pads.py)")
    p.add_argument("--test_path",
                   default="data/pads/test.csv")
    p.add_argument("--cnf_file",
                   default="dev/data/toml_files/pads_conf.toml")
    p.add_argument("--ckpt",
                   default="data/pads/checkpoints/pads_transformer.pt",
                   help="Path to checkpoint (best-val AUC-ROC by default)")
    p.add_argument("--out_dir",
                   default="refs/pads_eval",
                   help="Where to write metrics CSV and plots")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--cuda_idx", default=0, type=int,
                   help="CUDA device index (default 0). "
                        "Note: checkpoint may have been saved with cuda:1; "
                        "we override to avoid device-count mismatches.")
    return p.parse_args()


def dict_list_to_array(lst, keys):
    """Convert a list of {label: value} dicts into an (N, K) numpy array."""
    return np.array([[d[k] for k in keys] for d in lst], dtype=float)


def print_table(title, met, label_keys, metric_names):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    header = "{:22s}".format("") + "".join("{:>12s}".format(k) for k in label_keys)
    print(header)
    print("-" * len(header))
    for mk in metric_names:
        row = [met[i].get(mk, np.nan) for i in range(len(label_keys))]
        print("{:22s}".format(mk) +
              "".join("{:>12.4f}".format(v) for v in row))


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


def plot_roc(y_true, scores, label_keys, out_path, title):
    colors = {"HC": "#3B82C4", "PD": "#E56B1F", "DD": "#C0392B"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, k in enumerate(label_keys):
        fpr, tpr, _ = roc_curve(y_true[:, i], scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                label="{}  (AUC = {:.3f})".format(k, roc_auc),
                color=colors.get(k, None), linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(y_true, y_pred, label_keys, out_path, title,
                            threshold_labels=None):
    fig, axes = plt.subplots(1, len(label_keys),
                             figsize=(4 * len(label_keys), 3.8))
    if len(label_keys) == 1:
        axes = [axes]
    for i, (k, ax) in enumerate(zip(label_keys, axes)):
        cm = confusion_matrix(y_true[:, i].astype(int), y_pred[:, i].astype(int))
        disp = ConfusionMatrixDisplay(cm, display_labels=["not-{}".format(k), k])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        if threshold_labels is not None:
            ax.set_title("{}  (thr = {:.3f})".format(k, threshold_labels[k]))
        else:
            ax.set_title("{}  (pos = {})".format(k, int(y_true[:, i].sum())))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Ensure relative paths in args resolve against the repo root
    os.chdir(_REPO_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load test dataset ------------------------------------------------
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

    # ---- Load model from checkpoint ---------------------------------------
    # We avoid ADRDModel.from_ckpt because it hardcodes cuda_devices=[1]
    # (see adrd_model.py:80), which breaks on single-GPU machines and on
    # checkpoints saved from a cuda:1 device. We construct the object
    # manually so we can pin the device index.
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
    print("Target labels: {}".format(label_keys))

    # ---- Predict ----------------------------------------------------------
    print("Running inference on test set ...")
    # predict() returns (logits, proba, preds) where preds use threshold 0.5
    _, proba, preds_05 = mdl.predict(dat_tst.features)

    # Build (N, K) arrays
    scores = dict_list_to_array(proba, label_keys)
    y_pred_05 = dict_list_to_array(preds_05, label_keys).astype(int)
    y_true = dict_list_to_array(dat_tst.labels, label_keys).astype(int)
    mask = np.ones_like(y_true, dtype=int)  # PADS: every sample has all 3 labels

    # ---- Per-class label balance (sanity) ---------------------------------
    print("\nTest-set positive counts:")
    for i, k in enumerate(label_keys):
        print("  {:3s}  pos = {:3d}  /  neg = {:3d}".format(
            k, int(y_true[:, i].sum()), int(n_test - y_true[:, i].sum())))

    # ---- Metrics at threshold 0.5 -----------------------------------------
    met_05 = get_metrics_multitask(y_true, y_pred_05, scores, mask)
    metric_names = [
        "AUC (ROC)", "AUC (PR)",
        "Balanced Accuracy",
        "Sensitivity/Recall", "Specificity",
        "Precision", "F1 score",
    ]
    print_table("Test metrics  --threshold = 0.5",
                met_05, label_keys, metric_names)

    mean_auc_roc = float(np.mean([met_05[i]["AUC (ROC)"] for i in range(len(label_keys))]))
    mean_auc_pr  = float(np.mean([met_05[i]["AUC (PR)"]  for i in range(len(label_keys))]))
    print("\nMean AUC-ROC : {:.4f}".format(mean_auc_roc))
    print("Mean AUC-PR  : {:.4f}".format(mean_auc_pr))

    # ---- Metrics at Youden-optimal threshold per class --------------------
    thresholds_opt = {}
    y_pred_opt = np.zeros_like(y_pred_05)
    for i, k in enumerate(label_keys):
        fpr, tpr, thr = roc_curve(y_true[:, i], scores[:, i])
        youden = tpr - fpr
        thresholds_opt[k] = float(thr[int(np.argmax(youden))])
        y_pred_opt[:, i] = (scores[:, i] > thresholds_opt[k]).astype(int)

    met_opt = get_metrics_multitask(y_true, y_pred_opt, scores, mask)
    print_table("Test metrics  --Youden-optimal threshold per class",
                met_opt, label_keys, metric_names)
    print("\nOptimal thresholds:")
    for k, thr in thresholds_opt.items():
        print("  {:3s}  {:.4f}".format(k, thr))

    # ---- Save CSV ---------------------------------------------------------
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

    # ---- Plots ------------------------------------------------------------
    roc_path = os.path.join(args.out_dir, "roc_curves.png")
    plot_roc(y_true, scores, label_keys, roc_path,
             "Test-set ROC curves  (PADS, n={})".format(n_test))
    print("Saved -> {}".format(roc_path))

    cm_05_path = os.path.join(args.out_dir, "confusion_matrices_thr05.png")
    plot_confusion_matrices(y_true, y_pred_05, label_keys, cm_05_path,
                            "Test-set Confusion Matrices  (threshold = 0.5)")
    print("Saved -> {}".format(cm_05_path))

    cm_opt_path = os.path.join(args.out_dir, "confusion_matrices_youden.png")
    plot_confusion_matrices(y_true, y_pred_opt, label_keys, cm_opt_path,
                            "Test-set Confusion Matrices  (Youden-optimal)",
                            threshold_labels=thresholds_opt)
    print("Saved -> {}".format(cm_opt_path))

    # ---- Also dump the raw scores so you can re-plot later ---------------
    np.savez(os.path.join(args.out_dir, "test_predictions.npz"),
             y_true=y_true, scores=scores,
             y_pred_05=y_pred_05, y_pred_opt=y_pred_opt,
             label_keys=np.array(label_keys),
             thresholds_opt=np.array([thresholds_opt[k] for k in label_keys]))
    print("Saved -> {}/test_predictions.npz".format(args.out_dir))

    print("\nDone.")


if __name__ == "__main__":
    main()
