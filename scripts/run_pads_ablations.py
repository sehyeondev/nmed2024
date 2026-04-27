"""
Retrain PADS Transformer with only one modality group at a time.

Runs three trainings in sequence (on GPU):
    mov_only        — 44 movement features
    nms_only        — 30 NMS-questionnaire features
    mov_nms         — 74 features, drop demographics + physical

Each uses the same train/val/test split as the original single-split
run (data/pads/{train,val,test}.csv) and the same hyperparameters as
train_pads.py. After training, evaluate on the held-out test split.

Resumable: skips a condition if both ckpt and test_predictions.npz exist.

Outputs:
    data/pads/checkpoints_ablation/{name}.pt
    refs/pads_ablation/{name}/test_metrics_thr05.csv + predictions

Usage:
    python scripts/run_pads_ablations.py
    python scripts/run_pads_ablations.py --only nms_only
"""
import argparse
import os
import subprocess
import time

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

PY = r"C:\Users\shkim\anaconda3\envs\viterbi_net\python.exe"

CKPT_DIR = os.path.join(_REPO_ROOT, "data", "pads", "checkpoints_ablation")
EVAL_DIR = os.path.join(_REPO_ROOT, "refs", "pads_ablation")
LOG_DIR  = os.path.join(_REPO_ROOT, "data", "pads", "ablation_logs")
TOML_DIR = os.path.join(_REPO_ROOT, "dev",  "data", "toml_files")

TRAIN_PATH = os.path.join(_REPO_ROOT, "data", "pads", "train.csv")
VAL_PATH   = os.path.join(_REPO_ROOT, "data", "pads", "val.csv")
TEST_PATH  = os.path.join(_REPO_ROOT, "data", "pads", "test.csv")
DATA_PATH  = os.path.join(_REPO_ROOT, "data", "pads", "converted_pads.csv")

TRAIN_ARGS_BASE = [
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

CONDITIONS = {
    "mov_only":  "pads_conf_mov_only.toml",
    "nms_only":  "pads_conf_nms_only.toml",
    "mov_nms":   "pads_conf_mov_nms.toml",
}


def _fwd(p): return p.replace("\\", "/")


def train_one(name, toml_name):
    cnf  = os.path.join(TOML_DIR, toml_name)
    ckpt = os.path.join(CKPT_DIR, "{}.pt".format(name))
    log  = os.path.join(LOG_DIR,  "{}_train.log".format(name))
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    os.makedirs(os.path.dirname(log),  exist_ok=True)

    cmd = [PY, "dev/train.py",
           "--data_path",  _fwd(DATA_PATH),
           "--train_path", _fwd(TRAIN_PATH),
           "--vld_path",   _fwd(VAL_PATH),
           "--test_path",  _fwd(TEST_PATH),
           "--cnf_file",   _fwd(cnf),
           "--ckpt_path",  _fwd(ckpt)] + TRAIN_ARGS_BASE
    print("\n[{}] training -> {}".format(name, ckpt))
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                             cwd=_REPO_ROOT).returncode
    dt = time.time() - t0
    print("[{}] train rc={} in {:.1f}s".format(name, rc, dt))
    return rc == 0


def eval_one(name, toml_name):
    cnf  = os.path.join(TOML_DIR, toml_name)
    ckpt = os.path.join(CKPT_DIR, "{}.pt".format(name))
    out_dir = os.path.join(EVAL_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [PY, "scripts/eval_pads.py",
           "--test_path", _fwd(TEST_PATH),
           "--cnf_file",  _fwd(cnf),
           "--ckpt",      _fwd(ckpt),
           "--out_dir",   _fwd(out_dir)]

    log = os.path.join(LOG_DIR, "{}_eval.log".format(name))
    print("[{}] eval -> {}".format(name, out_dir))
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                             cwd=_REPO_ROOT).returncode
    dt = time.time() - t0
    print("[{}] eval  rc={} in {:.1f}s".format(name, rc, dt))
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(CONDITIONS.keys()), default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    conds = {args.only: CONDITIONS[args.only]} if args.only else CONDITIONS

    for name, toml_name in conds.items():
        ckpt  = os.path.join(CKPT_DIR, "{}.pt".format(name))
        preds = os.path.join(EVAL_DIR, name, "test_predictions.npz")
        if not args.force and os.path.exists(ckpt) and os.path.exists(preds):
            print("[{}] already done - skipping.".format(name))
            continue
        if not args.skip_train and (args.force or not os.path.exists(ckpt)):
            if not train_one(name, toml_name):
                print("[{}] training failed -- stopping.".format(name))
                return
        if not eval_one(name, toml_name):
            print("[{}] eval failed -- stopping.".format(name))
            return

    print("\nAll ablations complete.")


if __name__ == "__main__":
    main()
