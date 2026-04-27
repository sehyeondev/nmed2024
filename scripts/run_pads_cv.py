"""
Run 5-fold CV on PADS: for each fold, train then evaluate.

For fold k (0..4):
    - Train with data/pads/cv/fold{k}_{train,val,test}.csv
    - Checkpoint to data/pads/cv_ckpt/fold{k}.pt   (+ _AUPR)
    - Evaluate best-AUROC checkpoint on fold's test split,
      output to refs/pads_eval_cv/fold{k}/

Resumable:  if fold{k}.pt AND refs/pads_eval_cv/fold{k}/test_predictions.npz
            both exist, the fold is skipped.

Usage:
    python scripts/run_pads_cv.py                 # run folds 0..4
    python scripts/run_pads_cv.py --folds 2 3     # run specified folds
    python scripts/run_pads_cv.py --skip_train    # eval-only mode
"""
import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

PY = r"C:\Users\shkim\anaconda3\envs\viterbi_net\python.exe"

CV_DIR       = os.path.join(_REPO_ROOT, "data", "pads", "cv")
CKPT_DIR     = os.path.join(_REPO_ROOT, "data", "pads", "cv_ckpt")
LOG_DIR      = os.path.join(_REPO_ROOT, "data", "pads", "cv_logs")
EVAL_DIR     = os.path.join(_REPO_ROOT, "refs",  "pads_eval_cv")
CNF_FILE     = os.path.join(_REPO_ROOT, "dev",   "data", "toml_files",
                             "pads_conf.toml")

TRAIN_ARGS_BASE = [
    "--cnf_file", CNF_FILE,
    "--img_net",  "NonImg",
    "--img_mode", "-1",
    "--img_size", "(1,1,1)",
    "--d_model",  "128",
    "--nhead",    "4",
    "--batch_size", "16",
    "--num_epochs", "256",
    "--lr",       "1e-4",
    "--save_intermediate_ckpts",
]


def _fwd(p):
    """Windows backslash -> forward slash (dev/train.py does .split('/'))."""
    return p.replace("\\", "/")


def train_fold(k, data_path_full):
    """Launch a single training run for fold k."""
    fold_tr = os.path.join(CV_DIR, "fold{}_train.csv".format(k))
    fold_vl = os.path.join(CV_DIR, "fold{}_val.csv".format(k))
    fold_te = os.path.join(CV_DIR, "fold{}_test.csv".format(k))
    ckpt    = os.path.join(CKPT_DIR, "fold{}.pt".format(k))
    log_f   = os.path.join(LOG_DIR,  "fold{}_train.log".format(k))
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    cmd = [
        PY, "dev/train.py",
        "--data_path", _fwd(data_path_full),
        "--train_path", _fwd(fold_tr),
        "--vld_path",   _fwd(fold_vl),
        "--test_path",  _fwd(fold_te),
        "--ckpt_path",  _fwd(ckpt),
    ] + TRAIN_ARGS_BASE

    print("[fold {}] training \u2192 {}".format(k, ckpt))
    print("         log      \u2192 {}".format(log_f))
    t0 = time.time()
    with open(log_f, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=_REPO_ROOT)
    dt = time.time() - t0
    if proc.returncode != 0:
        print("[fold {}] TRAIN FAILED (rc={}) after {:.1f}s. See log.".format(
            k, proc.returncode, dt))
        return False
    print("[fold {}] train done in {:.1f}s.".format(k, dt))
    return True


def eval_fold(k):
    """Run eval_pads.py on fold k."""
    fold_te = os.path.join(CV_DIR, "fold{}_test.csv".format(k))
    ckpt    = os.path.join(CKPT_DIR, "fold{}.pt".format(k))
    out_dir = os.path.join(EVAL_DIR, "fold{}".format(k))
    log_f   = os.path.join(LOG_DIR,  "fold{}_eval.log".format(k))
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        PY, "scripts/eval_pads.py",
        "--test_path", fold_te,
        "--cnf_file",  CNF_FILE,
        "--ckpt",      ckpt,
        "--out_dir",   out_dir,
    ]
    print("[fold {}] evaluating \u2192 {}".format(k, out_dir))
    t0 = time.time()
    with open(log_f, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=_REPO_ROOT)
    dt = time.time() - t0
    if proc.returncode != 0:
        print("[fold {}] EVAL FAILED (rc={}) after {:.1f}s. See log.".format(
            k, proc.returncode, dt))
        return False
    print("[fold {}] eval done in {:.1f}s.".format(k, dt))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Ignore existing checkpoints/predictions and re-run")
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    data_path_full = os.path.join(_REPO_ROOT, "data", "pads",
                                    "converted_pads.csv")

    for k in args.folds:
        ckpt = os.path.join(CKPT_DIR, "fold{}.pt".format(k))
        preds = os.path.join(EVAL_DIR, "fold{}".format(k),
                              "test_predictions.npz")
        ckpt_ok = os.path.exists(ckpt)
        eval_ok = os.path.exists(preds)

        if not args.force and ckpt_ok and eval_ok:
            print("[fold {}] already done \u2014 skipping "
                  "(use --force to re-run).".format(k))
            continue

        if not args.skip_train and (args.force or not ckpt_ok):
            ok = train_fold(k, data_path_full)
            if not ok:
                print("[fold {}] aborting remaining work".format(k))
                sys.exit(1)

        ok = eval_fold(k)
        if not ok:
            sys.exit(1)

    print("\nAll requested folds complete.")


if __name__ == "__main__":
    main()
