"""
Bootstrap 95% confidence intervals for per-label AUROC / AUPR on both
NACC and PADS held-out test sets.

For each label:
    - Draw B=1000 test-set resamples with replacement (stratified on label)
    - Compute AUROC and AUPR on each resample
    - Report (mean, 2.5%, 97.5%) percentiles

Also prints paper-comparable aggregate (micro / macro) CIs.

Usage:
    cd <repo root>
    python scripts/bootstrap_ci.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

NACC_NPZ = os.path.join(_REPO_ROOT, "refs", "nacc_eval",    "test_predictions.npz")
PADS_NPZ = os.path.join(_REPO_ROOT, "refs", "pads_eval_v2", "test_predictions.npz")
OUT_DIR  = os.path.join(_REPO_ROOT, "refs", "bootstrap_ci")
os.makedirs(OUT_DIR, exist_ok=True)

B    = 1000
SEED = 2024


def _safe_auc(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    try:
        return roc_auc_score(y, s)
    except ValueError:
        return np.nan


def _safe_ap(y, s):
    if y.sum() == 0:
        return np.nan
    try:
        return average_precision_score(y, s)
    except ValueError:
        return np.nan


def bootstrap_per_label(y_true, scores, mask, label_keys, rng):
    """Return a DataFrame with per-label mean / CI for AUROC and AUPR."""
    n = y_true.shape[0]
    rows = []
    for i, k in enumerate(label_keys):
        m = mask[:, i].astype(bool)
        yi = y_true[m, i].astype(int)
        si = scores[m, i]
        if yi.sum() == 0 or yi.sum() == len(yi):
            rows.append({"label": k, "n": int(m.sum()),
                         "pos": int(yi.sum()),
                         "auroc": np.nan, "auroc_lo": np.nan, "auroc_hi": np.nan,
                         "aupr":  np.nan, "aupr_lo":  np.nan, "aupr_hi":  np.nan})
            continue

        auroc_pt = _safe_auc(yi, si)
        aupr_pt  = _safe_ap (yi, si)

        auroc_bs = np.empty(B)
        aupr_bs  = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, len(yi), size=len(yi))
            auroc_bs[b] = _safe_auc(yi[idx], si[idx])
            aupr_bs[b]  = _safe_ap (yi[idx], si[idx])

        auroc_bs = auroc_bs[~np.isnan(auroc_bs)]
        aupr_bs  = aupr_bs [~np.isnan(aupr_bs)]

        rows.append({
            "label": k,
            "n":   int(m.sum()),
            "pos": int(yi.sum()),
            "auroc":     auroc_pt,
            "auroc_lo":  float(np.percentile(auroc_bs, 2.5)),
            "auroc_hi":  float(np.percentile(auroc_bs, 97.5)),
            "aupr":      aupr_pt,
            "aupr_lo":   float(np.percentile(aupr_bs, 2.5)),
            "aupr_hi":   float(np.percentile(aupr_bs, 97.5)),
        })
    return pd.DataFrame(rows)


def bootstrap_aggregate(y_true, scores, mask, cols, rng):
    """Micro / macro / weighted AUROC & AUPR with bootstrap CI.

    Aggregates over the columns in `cols` only. Rows where any col is
    masked-out are dropped (simple, matches compute_nacc_aggregate.py)."""
    mk = mask[:, cols]
    ok = mk.sum(axis=1) == mk.shape[1]
    yt = y_true[ok][:, cols].astype(int)
    sc = scores[ok][:, cols]

    def one_shot(y, s):
        out = {}
        for avg in ("micro", "macro", "weighted"):
            try:
                out[("auroc", avg)] = roc_auc_score(y, s, average=avg)
            except ValueError:
                out[("auroc", avg)] = np.nan
            try:
                out[("aupr",  avg)] = average_precision_score(y, s, average=avg)
            except ValueError:
                out[("aupr",  avg)] = np.nan
        return out

    pt = one_shot(yt, sc)
    bs = {m: [] for m in pt}
    n = yt.shape[0]
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        one = one_shot(yt[idx], sc[idx])
        for m, v in one.items():
            if not np.isnan(v):
                bs[m].append(v)

    rows = []
    for (metric, avg), v in pt.items():
        arr = np.array(bs[(metric, avg)])
        rows.append({
            "metric": metric, "average": avg,
            "value":  v,
            "lo":     float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
            "hi":     float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
        })
    return pd.DataFrame(rows)


def run_one(npz_path, tag, aggregate_groups):
    print("\n" + "=" * 70)
    print("Dataset: {}".format(tag))
    print("NPZ    : {}".format(npz_path))
    d = np.load(npz_path, allow_pickle=True)
    y_true = d["y_true"].astype(int)
    scores = d["scores"]
    label_keys = list(d["label_keys"])
    if "mask" in d.files:
        mask = d["mask"].astype(int)
    else:
        mask = np.ones_like(y_true, dtype=int)
    print("Test N = {}   Labels = {}".format(y_true.shape[0], label_keys))

    rng = np.random.default_rng(SEED)

    # Per-label CIs
    df_lab = bootstrap_per_label(y_true, scores, mask, label_keys, rng)
    csv_lab = os.path.join(OUT_DIR, "{}_per_label_ci.csv".format(tag))
    df_lab.to_csv(csv_lab, index=False)
    print("\nPer-label 95% CI:")
    for _, r in df_lab.iterrows():
        print("  {:4s}  n={:4d} pos={:4d}  "
              "AUROC {:.3f} [{:.3f}, {:.3f}]  "
              "AUPR {:.3f} [{:.3f}, {:.3f}]"
              .format(r["label"], r["n"], r["pos"],
                      r["auroc"], r["auroc_lo"], r["auroc_hi"],
                      r["aupr"],  r["aupr_lo"],  r["aupr_hi"]))
    print("Saved -> {}".format(csv_lab))

    # Aggregate CIs
    idx_of = {k: i for i, k in enumerate(label_keys)}
    for grp_name, grp_labels in aggregate_groups.items():
        cols = [idx_of[x] for x in grp_labels if x in idx_of]
        if len(cols) < 2:
            continue
        df_agg = bootstrap_aggregate(y_true, scores, mask, cols, rng)
        csv_agg = os.path.join(
            OUT_DIR, "{}_{}_agg_ci.csv".format(tag, grp_name))
        df_agg.to_csv(csv_agg, index=False)
        print("\n[{} -- {}] aggregate 95% CI:".format(tag, grp_name))
        for _, r in df_agg.iterrows():
            print("  {:6s}  {:8s}  {:.3f}  [{:.3f}, {:.3f}]"
                  .format(r["metric"], r["average"],
                          r["value"], r["lo"], r["hi"]))
        print("Saved -> {}".format(csv_agg))


def main():
    nacc_groups = {
        "nc_mci_de":      ["NC", "MCI", "DE"],
        "ten_etiologies": ["AD", "LBD", "VD", "PRD", "FTD",
                            "NPH", "SEF", "PSY", "TBI", "ODE"],
        "all13":          ["NC", "MCI", "DE", "AD", "LBD", "VD", "PRD",
                            "FTD", "NPH", "SEF", "PSY", "TBI", "ODE"],
    }
    pads_groups = {
        "hc_pd_dd": ["HC", "PD", "DD"],
    }

    if os.path.exists(NACC_NPZ):
        run_one(NACC_NPZ, "nacc", nacc_groups)
    else:
        print("SKIP NACC (no npz at {})".format(NACC_NPZ))

    if os.path.exists(PADS_NPZ):
        run_one(PADS_NPZ, "pads", pads_groups)
    else:
        print("SKIP PADS (no npz at {})".format(PADS_NPZ))


if __name__ == "__main__":
    main()
