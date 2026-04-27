# PADS Extension — Modality-Agnostic Transformer for Parkinson's Disease

This document describes how to reproduce the PADS (Parkinson's Disease
Smartwatch) experiments built on top of the dementia framework in
[`README.md`](README.md). The same modality-agnostic transformer
backbone (Xu et al., *Nature Medicine* 2024) is extended to a
3-class differential diagnosis on wearable + questionnaire data
(Varghese et al., *npj Parkinson's Disease* 2024).

---

## 1. What this extension adds

| Component | File(s) |
|:---|:---|
| Raw PADS → CSV/TOML conversion (Phase 2 token grouping) | `scripts/convert_pads_data.py` |
| Training entry-point | `scripts/train_pads.py` |
| Single-split evaluation (ROC, AUPR, confusion matrices) | `scripts/eval_pads.py` |
| 5-fold cross-validation runner + aggregator | `scripts/make_pads_cv_splits.py`, `scripts/run_pads_cv.py`, `scripts/aggregate_pads_cv.py` |
| Bootstrap 95 % CIs (NACC + PADS) | `scripts/bootstrap_ci.py` |
| Missing-modality robustness (zero-out at inference) | `scripts/pads_missing_modality.py` |
| Per-modality retrain ablations | `scripts/make_pads_modality_confs.py`, `scripts/run_pads_ablations.py` |
| Attention visualization | `scripts/pads_attention.py` |
| Modality TOML configs | `dev/data/toml_files/pads_conf*.toml` (full / mov+nms / mov-only / nms-only) |
| NACC re-evaluation utilities | `scripts/eval_nacc.py`, `scripts/compute_nacc_aggregate.py`, `scripts/plot_nacc_primary_roc.py` |

The 79-token PADS scheme is:
**44 movement** (11 task × 2 wrist × 2 sensor, each 93-dim) **+ 30 NMS
binary items + 5 demographics** = 79 tokens, projected to a shared
`d_model = 128` token space and consumed by the same transformer
encoder used for the dementia model.

---

## 2. Data — what you must download yourself

Nothing under `data/pads/` or `data/nacc/` is committed to the
repository (see `.gitignore`). Each user must obtain the source data
separately.

### 2.1 PADS (open access)

Source: Varghese et al. 2024 — Zenodo
[https://zenodo.org/records/11091518](https://zenodo.org/records/11091518)
(or the link given in the paper; verify the version).

Expected layout after extraction (the conversion script reads these
folders):

```
<PADS_ROOT>/
├── movement/                       # raw smartwatch recordings
├── questionnaire/                  # NMS questionnaire JSONs
├── patients/                       # demographics + label
└── ...
```

Point the conversion script at the extracted archive via either a CLI
flag or the `PADS_ROOT` environment variable:

```bash
python scripts/convert_pads_data.py --pads_root /path/to/pads_zenodo
# or
PADS_ROOT=/path/to/pads_zenodo python scripts/convert_pads_data.py
```

Output paths default to `<repo>/data/pads/` and
`<repo>/dev/data/toml_files/`, but are also overridable via
`--output_dir` and `--toml_dir`.

### 2.2 NACC (restricted)

The dementia experiments require NACC data, which is **not
redistributable**. You must request access yourself
([https://naccdata.org/](https://naccdata.org/)) and place the
processed files at:

```
data/nacc/                          # raw NACC dump
data/train_vld_test_split_updated/  # generated splits
data/training_cohorts/              # cohort selections
```

If you only need the PADS results, you can skip the NACC sections.

---

## 3. Environment

Same as the parent project (see `README.md` § Prerequisites). Python
3.11+, PyTorch ≥ 2.1, scikit-learn, scipy, pandas, matplotlib. From
the repo root:

```bash
pip install -e .
```

A single GPU with ≥ 8 GB memory is sufficient for PADS.

---

## 4. End-to-end PADS pipeline

```bash
# (1) Convert Zenodo archive to nmed2024 CSV format
python scripts/convert_pads_data.py --pads_root /path/to/pads_zenodo
#   produces: data/pads/{converted_pads,train,val,test}.csv
#             dev/data/toml_files/pads_conf.toml

# (2) Train on the original 328/47/94 split (~1 h on a single GPU)
python scripts/train_pads.py
#   produces: data/pads/checkpoints/pads_transformer_v2.pt

# (3) Evaluate single-split: per-class AUROC/AUPR + ROC + confusion mat
python scripts/eval_pads.py
#   produces: refs/pads_eval/  (metrics CSV, roc_curves.png, predictions.npz)

# (4) (optional) 5-fold cross-validation in the Varghese et al. style
python scripts/make_pads_cv_splits.py
python scripts/run_pads_cv.py
python scripts/aggregate_pads_cv.py
#   produces: refs/pads_eval_cv/  (per-fold metrics + pooled ROC)

# (5) (optional) Robustness, ablation, attention
python scripts/pads_missing_modality.py
python scripts/make_pads_modality_confs.py
python scripts/run_pads_ablations.py
python scripts/pads_attention.py

# (6) (optional) Bootstrap 95 % CIs over predictions from (3) or (4)
python scripts/bootstrap_ci.py
```

All result CSVs and figures land under `refs/<experiment_name>/`. The
`refs/` directory is git-ignored except for `refs/papers/` (reference
PDFs).

---

## 5. Headline numbers (single-split, Phase 2 tokenization)

| Class | n_pos (test=94) | AUROC | AUPR |
|:---|---:|:---:|:---:|
| HC | 16 | 0.952 | 0.760 |
| PD | 55 | 0.850 | 0.872 |
| DD | 23 | 0.794 | 0.676 |
| **mean** | — | **0.865** | **0.770** |

5-fold CV (469 subjects, pooled): mean AUROC **0.834 ± 0.033**.

---

## 6. NACC reproduction (requires NACC access)

Once `data/nacc/` and the derived split CSVs are in place, the
upstream training pipeline (`dev/train.py` / `dev/train.sh`) produces
the model checkpoint, then:

```bash
python scripts/eval_nacc.py
python scripts/compute_nacc_aggregate.py
python scripts/plot_nacc_primary_roc.py
python scripts/bootstrap_ci.py
```

reproduces the per-label and aggregate metrics with bootstrap CIs.

---

## 7. Pre-trained checkpoints

Checkpoints (`data/pads/checkpoints/*.pt`, `data/pads/cv_ckpt/`) are
**not in the repository** because of size. If you need them without
retraining, contact the maintainer for a direct file transfer or a
shared-storage link.

---

## 8. Layout recap

```
adrd/                                # backbone model code (transformer, embeddings, heads)
dev/                                 # original training infrastructure
  data/toml_files/pads_conf*.toml      # PADS modality configs
  train.py, train.sh                   # generic trainer used by train_pads.py
scripts/                             # PADS + NACC reproduction scripts
data/
  create_splits.py
  generate_data_stats.py
  datasets/                          # cohort ID lists & conversion notebooks (upstream)
  pads/         (gitignored)         # populate by running convert_pads_data.py
  nacc/         (gitignored)         # populate from NACC request
refs/
  papers/                            # reference PDFs (publicly available)
  (other contents gitignored)
```
