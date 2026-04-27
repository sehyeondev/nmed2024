"""
Build 5 stratified folds from the full PADS dataset for 5-fold CV.

For each fold k in {0, 1, 2, 3, 4}:
    - fold k = test  (1/5 of subjects, stratified by 3-class label)
    - remaining 4 folds -> split 87.5% train / 12.5% val (matches original ratio)

Outputs  (under data/pads/cv/):
    fold{k}_train.csv   fold{k}_val.csv   fold{k}_test.csv
    fold_summary.csv    (per-fold class counts)

Usage:
    python scripts/make_pads_cv_splits.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
PADS_DIR = os.path.join(_REPO_ROOT, "data", "pads")
OUT_DIR  = os.path.join(PADS_DIR, "cv")

SEED_OUTER = 42      # fold split
SEED_INNER = 43      # train/val split within each fold
N_FOLDS    = 5
VAL_RATIO  = 0.125   # matches original 328/375


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    full = pd.read_csv(os.path.join(PADS_DIR, "converted_pads.csv"))
    print("Full dataset: {} subjects".format(len(full)))

    # Derive 3-class label from one-hot HC/PD/DD columns
    labels3 = np.argmax(full[["HC", "PD", "DD"]].values, axis=1)
    class_names = {0: "HC", 1: "PD", 2: "DD"}
    print("Class counts: {}".format(
        {class_names[c]: int((labels3 == c).sum()) for c in range(3)}))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED_OUTER)

    summary_rows = []
    for k, (trainval_idx, test_idx) in enumerate(skf.split(full, labels3)):
        tv_df = full.iloc[trainval_idx].reset_index(drop=True)
        tv_y  = labels3[trainval_idx]
        te_df = full.iloc[test_idx].reset_index(drop=True)

        # Inner stratified split for train / val
        sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO,
                                      random_state=SEED_INNER + k)
        tr_rel, vl_rel = next(sss.split(tv_df, tv_y))
        tr_df = tv_df.iloc[tr_rel].reset_index(drop=True)
        vl_df = tv_df.iloc[vl_rel].reset_index(drop=True)

        # Save
        tr_df.to_csv(os.path.join(OUT_DIR, "fold{}_train.csv".format(k)),
                     index=False)
        vl_df.to_csv(os.path.join(OUT_DIR, "fold{}_val.csv".format(k)),
                     index=False)
        te_df.to_csv(os.path.join(OUT_DIR, "fold{}_test.csv".format(k)),
                     index=False)

        def cnt(df, c):
            return int(df[c].sum())
        row = {
            "fold": k,
            "n_train": len(tr_df), "n_val": len(vl_df), "n_test": len(te_df),
            "train_HC": cnt(tr_df, "HC"), "train_PD": cnt(tr_df, "PD"),
            "train_DD": cnt(tr_df, "DD"),
            "val_HC":   cnt(vl_df, "HC"), "val_PD":   cnt(vl_df, "PD"),
            "val_DD":   cnt(vl_df, "DD"),
            "test_HC":  cnt(te_df, "HC"), "test_PD":  cnt(te_df, "PD"),
            "test_DD":  cnt(te_df, "DD"),
        }
        summary_rows.append(row)
        print("fold {}: train {:3d} (HC{:2d}/PD{:3d}/DD{:2d})  "
              "val {:2d} (HC{:d}/PD{:2d}/DD{:d})  "
              "test {:2d} (HC{:d}/PD{:2d}/DD{:2d})"
              .format(k, row["n_train"], row["train_HC"], row["train_PD"],
                      row["train_DD"],
                      row["n_val"],  row["val_HC"],  row["val_PD"],
                      row["val_DD"],
                      row["n_test"], row["test_HC"], row["test_PD"],
                      row["test_DD"]))

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUT_DIR, "fold_summary.csv"), index=False)
    print("\nSaved splits to: {}".format(OUT_DIR))


if __name__ == "__main__":
    main()
