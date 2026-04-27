"""
Visualize attention weights on the PADS held-out test set.

The transformer uses nn.TransformerEncoderLayer, whose self_attn module
is a MultiheadAttention. We register a wrapper around each layer's
self_attn.forward that (a) forces need_weights=True,
average_attn_weights=False and (b) stashes the returned
attn_output_weights on the module.

Token ordering in this model (see adrd/nn/transformer.py: forward_trf):
    [TGT_0 ... TGT_{T-1}, SRC_0 ... SRC_{S-1}]
where TGT tokens are the HC / PD / DD CLS tokens and SRC tokens are
source features from mdl.src_modalities (in insertion order).

For each PADS test subject:
    - Run forward once with attention capture on.
    - From the last encoder layer's attention tensor
        (B, heads, L, L)   where L = T + S
      take the row of each TGT token (0..T-1).
    - Slice columns T..T+S (source attention only).
    - Average over heads -> per-subject (T, S) attention matrix.

We then:
    - Average per class (HC / PD / DD subjects)
    - Rank source features by mean attention for each class CLS token.

Outputs:
    refs/pads_attention/
        attention_raw.npz             # (n_subj, T, S) source-attn tensor
        attention_per_class.csv       # mean attention per (cls_token, feature)
        top_features_by_class.csv     # top-K features per class
        attention_heatmap.png         # rows TGT x cols top-K SRC

Usage:
    cd <repo root>
    python scripts/pads_attention.py
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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

_orig_torch_load = torch.load
def _patched(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _patched

from dev.data.dataset_csv import CSVDataset
from adrd.model import ADRDModel
from adrd.utils.transformer_dataset import TransformerTestingDataset

CKPT     = os.path.join(_REPO_ROOT, "data", "pads", "checkpoints",
                        "pads_transformer_v2.pt")
TEST_CSV = os.path.join(_REPO_ROOT, "data", "pads", "test.csv")
CNF_FILE = os.path.join(_REPO_ROOT, "dev", "data", "toml_files",
                        "pads_conf.toml")
OUT_DIR  = os.path.join(_REPO_ROOT, "refs", "pads_attention")
os.makedirs(OUT_DIR, exist_ok=True)

TOP_K = 15      # how many top features to show per class
USE_LAYER = -1  # which encoder layer's attention to use (-1 = last)


def _attach_attention_capture(transformer_encoder):
    """Wrap each layer.self_attn.forward to force need_weights=True and
    stash the returned (B, H, L, L) weights on the module as `.attn_w`."""
    for lyr in transformer_encoder.layers:
        mha = lyr.self_attn
        orig_forward = mha.forward

        def make_wrapper(_orig, _mha):
            def wrapped(q, k, v, key_padding_mask=None, need_weights=True,
                        attn_mask=None, average_attn_weights=False,
                        is_causal=False):
                out, aw = _orig(q, k, v,
                                 key_padding_mask=key_padding_mask,
                                 need_weights=True,
                                 attn_mask=attn_mask,
                                 average_attn_weights=False,
                                 is_causal=is_causal)
                _mha.attn_w = aw.detach()
                return out, aw
            return wrapped
        mha.forward = make_wrapper(orig_forward, mha)


def main():
    os.chdir(_REPO_ROOT)

    print("Loading test dataset ...")
    dat = CSVDataset(dat_file=TEST_CSV, cnf_file=CNF_FILE,
                     mode=2, img_mode=-1, arch="NonImg",
                     transforms=None, stripped=None)
    features = dat.features
    labels   = dat.labels

    print("Loading model ...")
    mdl = ADRDModel(None, None, None, device="cuda", cuda_devices=[0])
    mdl.device = "cuda:0"
    mdl.load(CKPT, map_location="cuda:0")
    label_keys = list(mdl.tgt_modalities.keys())
    src_keys   = list(mdl.src_modalities.keys())
    T = len(label_keys); S = len(src_keys)
    print("  T = {}  (targets: {})".format(T, label_keys))
    print("  S = {}  (source tokens)".format(S))

    # Hook the transformer encoder
    transformer_encoder = mdl.net_.transformer
    _attach_attention_capture(transformer_encoder)

    # Build a test loader (batch size 1 for safety)
    ds = TransformerTestingDataset(features, mdl.src_modalities)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=TransformerTestingDataset.collate_fn)

    mdl.net_.eval()
    torch.set_grad_enabled(False)

    attn_list = []  # per subject: (T, S)
    y_true    = []
    for i, (x, mask) in enumerate(loader):
        x    = {k: x[k].to(mdl.device) for k in x}
        mask = {k: mask[k].to(mdl.device) for k in mask}
        _ = mdl.net_(x, mask, None)

        lyr = transformer_encoder.layers[USE_LAYER]
        aw = lyr.self_attn.attn_w      # (B=1, H, L, L)
        aw = aw.mean(dim=1)            # average over heads -> (1, L, L)
        aw = aw.squeeze(0).cpu().numpy()  # (L, L)

        # rows [0..T-1] are target CLS tokens; columns T..T+S-1 are src tokens
        sub = aw[:T, T:T + S]
        attn_list.append(sub)

        y_true.append([labels[i][k] for k in label_keys])

        if (i + 1) % 20 == 0:
            print("  {}/{}".format(i + 1, len(loader)))

    A = np.stack(attn_list, axis=0)    # (N, T, S)
    y = np.array(y_true, dtype=int)    # (N, T)
    print("Collected attention: {}".format(A.shape))

    np.savez(os.path.join(OUT_DIR, "attention_raw.npz"),
             attn=A, y_true=y,
             label_keys=np.array(label_keys),
             src_keys=np.array(src_keys))
    print("Saved -> attention_raw.npz")

    # Per-class mean attention (average over subjects positive for that class)
    per_class = np.zeros((T, S))
    for j, _ in enumerate(label_keys):
        m = y[:, j] == 1
        if m.sum() == 0:
            continue
        per_class[j] = A[m, j, :].mean(axis=0)

    df = pd.DataFrame(per_class.T, index=src_keys, columns=label_keys)
    df.to_csv(os.path.join(OUT_DIR, "attention_per_class.csv"))
    print("Saved -> attention_per_class.csv")

    # Top-K per class
    top_rows = []
    for j, cls in enumerate(label_keys):
        order = np.argsort(per_class[j])[::-1][:TOP_K]
        for rank, idx in enumerate(order):
            top_rows.append({"class": cls, "rank": rank + 1,
                              "feature": src_keys[idx],
                              "attention": float(per_class[j, idx])})
    pd.DataFrame(top_rows).to_csv(
        os.path.join(OUT_DIR, "top_features_by_class.csv"), index=False)
    print("Saved -> top_features_by_class.csv")

    # Heatmap: pick union of top-K features from all classes
    top_union = []
    for j in range(T):
        order = np.argsort(per_class[j])[::-1][:TOP_K]
        top_union.extend(order.tolist())
    top_union = list(dict.fromkeys(top_union))    # preserve order, dedupe
    sub = per_class[:, top_union]

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(top_union)), 3.5))
    im = ax.imshow(sub, aspect="auto", cmap="viridis")
    ax.set_yticks(range(T)); ax.set_yticklabels(label_keys)
    ax.set_xticks(range(len(top_union)))
    ax.set_xticklabels([src_keys[i] for i in top_union],
                       rotation=70, ha="right", fontsize=7)
    ax.set_title("PADS mean attention  (last encoder layer, heads avg)  "
                 "top-{} features per class".format(TOP_K))
    fig.colorbar(im, ax=ax, shrink=0.8, label="attention")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "attention_heatmap.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print("Saved -> {}".format(out_png))


if __name__ == "__main__":
    main()
