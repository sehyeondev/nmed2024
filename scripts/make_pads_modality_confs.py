"""
Write derivative PADS config TOMLs that keep a single modality group:
    pads_conf_mov_only.toml
    pads_conf_nms_only.toml
    pads_conf_mov_nms.toml            (movement + NMS, drop demog + phys)

Each derived toml is the original pads_conf.toml with unwanted
[feature.*] sections removed. The [label.*] section and other
top-level config are preserved verbatim.

Usage:
    python scripts/make_pads_modality_confs.py
"""
import os

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
TOML_DIR = os.path.join(_REPO_ROOT, "dev", "data", "toml_files")
SRC = os.path.join(TOML_DIR, "pads_conf.toml")


def keep_feature(name, keep_prefixes):
    return any(name.startswith(p) for p in keep_prefixes)


def filter_toml(src_path, dst_path, keep_prefixes):
    with open(src_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into blocks separated by blank lines so we can decide
    # block-by-block. (The file uses indented [feature.X] subtables,
    # each with its type / shape / num_categories below it.)
    blocks = text.split("\n\n")
    kept = []
    for b in blocks:
        # Find the first [feature.X] marker inside the block.
        is_feature = False
        feat_name = None
        for line in b.splitlines():
            s = line.strip()
            if s.startswith("[feature.") and s.endswith("]"):
                feat_name = s[len("[feature."):-1]
                is_feature = True
                break
        if is_feature and feat_name is not None \
                and not keep_feature(feat_name, keep_prefixes):
            continue          # drop
        kept.append(b)

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(kept))
    print("Wrote {}".format(dst_path))


def main():
    assert os.path.exists(SRC), SRC

    configs = {
        "pads_conf_mov_only.toml":  ("mov_",),
        "pads_conf_nms_only.toml":  ("nms_",),
        "pads_conf_mov_nms.toml":   ("mov_", "nms_"),
    }
    for name, prefixes in configs.items():
        dst = os.path.join(TOML_DIR, name)
        filter_toml(SRC, dst, prefixes)


if __name__ == "__main__":
    main()
