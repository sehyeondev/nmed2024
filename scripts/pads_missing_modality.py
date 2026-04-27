"""
Evaluate PADS model robustness to missing modalities at inference time.

For the existing held-out test set (94 subjects), we load the best-AUROC
checkpoint and run inference under several "input availability" scenarios:

    full        : all 79 features
    no_mov      : drop 44 movement-sensor features
    no_nms      : drop 30 non-motor-symptoms features
    no_demog    : drop 2 demographics + 3 physical features
    mov_only    : only the 44 movement features
    nms_only    : only the 30 NMS features
    mov+nms     : movement + NMS (drop demog+physical)

The data pipeline masks any key missing from the input dict; the
transformer's attention zeroes out masked tokens. So we just delete keys
per scenario.

Outputs:
    refs/pads_missing_modality/
        missing_modality_metrics.csv    (per-class & mean AUROC/AUPR per scenario)
        missing_modality_delta.csv      (AUROC drop from full)

Usage:
    cd C:/Users/shkim/codes/nmed2024
    python scripts/pads_missing_modality.py
"""
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

# PyTorch 2.6+ compat: checkpoint was saved with optimizer object
_orig_torch_load = torch.load
def _patched(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _patched

from dev.data.dataset_csv import CSVDataset
from adrd.model import ADRDModel

CKPT     = os.path.join(_REPO_ROOT, "data", "pads", "checkpoints",
                         "pads_transformer_v2.pt")
TEST_CSV = os.path.join(_REPO_ROOT, "data", "pads", "test.csv")
CNF_FILE = os.path.join(_REPO_ROOT, "dev", "data", "toml_files",
                         "pads_conf.toml")
OUT_DIR  = os.path.join(_REPO_ROOT, "refs", "pads_missing_modality")
os.makedirs(OUT_DIR, exist_ok=True)


def group_keys(all_keys):
    """Split keys into the 4 PADS modalities by prefix."""
    groups = {"demog": [], "physical": [], "nms": [], "mov": []}
    for k in all_keys:
        if k.startswith("his_"):    groups["demog"].append(k)
        elif k.startswith("ph_"):   groups["physical"].append(k)
        elif k.startswith("nms_"):  groups["nms"].append(k)
        elif k.startswith("mov_"):  groups["mov"].append(k)
    return groups


def filter_sample(sample, keep):
    """Return a copy of `sample` with only keys in `keep`."""
    return {k: v for k, v in sample.items() if k in keep}


def run_scenario(mdl, features, label_keys, keep_set):
    """Filter each sample to `keep_set` keys, run predict_proba."""
    filtered = [filter_sample(smp, keep_set) for smp in features]
    _, proba = mdl.predict_proba(filtered)
    scores = np.array([[p[k] for k in label_keys] for p in proba])
    return scores


def main():
    os.chdir(_REPO_ROOT)

    print("Loading test dataset ...")
    dat = CSVDataset(dat_file=TEST_CSV, cnf_file=CNF_FILE,
                     mode=2, img_mode=-1, arch="NonImg",
                     transforms=None, stripped=None)
    features = dat.features
    labels   = dat.labels
    n = len(features)
    print("  n = {}".format(n))

    print("Loading model on GPU ...")
    mdl = ADRDModel(None, None, None, device="cuda", cuda_devices=[0])
    mdl.device = "cuda:0"
    mdl.load(CKPT, map_location="cuda:0")
    label_keys = list(mdl.tgt_modalities.keys())
    print("  labels = {}".format(label_keys))

    # Use the keys actually present in the loaded samples (CSV-level: 79
    # aggregated features), not mdl.src_modalities (which holds 4127
    # expanded tokens). The dataset expands each CSV mov_* into many
    # sub-tokens downstream; we only need to control presence at the
    # CSV granularity.
    sample_keys = list(features[0].keys())
    print("  n sample keys = {}".format(len(sample_keys)))

    groups = group_keys(sample_keys)
    print("  groups:  " + "  ".join(
        "{} ({})".format(k, len(v)) for k, v in groups.items()))

    # Build y_true
    y_true = np.array([[s[k] for k in label_keys] for s in labels],
                       dtype=int)

    # Scenario definitions: name -> set of key names to keep
    demog_phys = set(groups["demog"]) | set(groups["physical"])
    mov        = set(groups["mov"])
    nms        = set(groups["nms"])
    full       = set(sample_keys)

    scenarios = {
        "full":     full,
        "no_mov":   full - mov,
        "no_nms":   full - nms,
        "no_demog": full - demog_phys,
        "mov_only": mov,
        "nms_only": nms,
        "mov+nms":  mov | nms,
    }

    rows = []
    base_auroc = None
    for name, keep in scenarios.items():
        print("\n--- Scenario: {}  (kept {} features) ---".format(
            name, len(keep)))
        scores = run_scenario(mdl, features, label_keys, keep)
        per_auc = {}
        per_ap  = {}
        for i, k in enumerate(label_keys):
            y = y_true[:, i]
            s = scores[:, i]
            try:
                per_auc[k] = float(roc_auc_score(y, s))
            except ValueError:
                per_auc[k] = np.nan
            try:
                per_ap[k] = float(average_precision_score(y, s))
            except ValueError:
                per_ap[k] = np.nan
        mean_auc = float(np.nanmean(list(per_auc.values())))
        mean_ap  = float(np.nanmean(list(per_ap.values())))
        print("  AUROC  HC={:.3f}  PD={:.3f}  DD={:.3f}  mean={:.3f}".format(
            per_auc.get("HC", np.nan), per_auc.get("PD", np.nan),
            per_auc.get("DD", np.nan), mean_auc))
        print("  AUPR   HC={:.3f}  PD={:.3f}  DD={:.3f}  mean={:.3f}".format(
            per_ap.get("HC", np.nan), per_ap.get("PD", np.nan),
            per_ap.get("DD", np.nan), mean_ap))

        row = {"scenario": name, "n_features": len(keep),
               "auroc_HC": per_auc.get("HC", np.nan),
               "auroc_PD": per_auc.get("PD", np.nan),
               "auroc_DD": per_auc.get("DD", np.nan),
               "auroc_mean": mean_auc,
               "aupr_HC":  per_ap.get("HC", np.nan),
               "aupr_PD":  per_ap.get("PD", np.nan),
               "aupr_DD":  per_ap.get("DD", np.nan),
               "aupr_mean": mean_ap}
        rows.append(row)
        if name == "full":
            base_auroc = mean_auc

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "missing_modality_metrics.csv")
    df.to_csv(csv_path, index=False)
    print("\nSaved -> {}".format(csv_path))

    if base_auroc is not None:
        df["delta_auroc"] = df["auroc_mean"] - base_auroc
        delta_csv = os.path.join(OUT_DIR, "missing_modality_delta.csv")
        df[["scenario", "n_features", "auroc_mean",
            "delta_auroc", "aupr_mean"]].to_csv(delta_csv, index=False)
        print("Saved -> {}".format(delta_csv))
        print("\nAUROC change vs full:")
        for _, r in df.iterrows():
            print("  {:12s}  AUROC={:.3f}  Δ={:+.3f}  ".format(
                r["scenario"], r["auroc_mean"], r["delta_auroc"]))


if __name__ == "__main__":
    main()
