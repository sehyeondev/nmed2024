"""
Train nmed2024 Transformer on PADS dataset (Phase 2: grouped tokens).
Run this after convert_pads_data.py has been executed.

Usage:
    cd <repo root>
    conda activate viterbi_net
    python scripts/train_pads.py
"""
import subprocess
import sys
import os

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
os.chdir(_REPO_ROOT)
os.makedirs("data/pads/checkpoints", exist_ok=True)

cmd = [
    sys.executable, "dev/train.py",
    "--data_path", "data/pads/converted_pads.csv",
    "--train_path", "data/pads/train.csv",
    "--vld_path", "data/pads/val.csv",
    "--test_path", "data/pads/test.csv",
    "--cnf_file", "dev/data/toml_files/pads_conf.toml",
    "--img_net", "NonImg",
    "--img_mode", "-1",
    "--img_size", "(1,1,1)",
    "--d_model", "128",
    "--nhead", "4",
    "--batch_size", "16",
    "--num_epochs", "256",
    "--lr", "1e-4",
    "--ckpt_path", "data/pads/checkpoints/pads_transformer_v2",
    "--save_intermediate_ckpts",
]

print("Running command:")
print(" ".join(cmd))
print()

subprocess.run(cmd)
