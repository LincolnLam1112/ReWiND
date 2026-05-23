"""Drop OpenX trajectories whose flow_signal is identically zero.

These trajectories correspond to videos where the DINO encoder produced
identical embeddings for every frame (camera static or change too subtle to
register), so `flow_progress` is degenerate (all zeros). Under uniform-group
sampling, ~68% of the affected groups dominate the resulting batches, which
makes ~65% of OpenX samples in a batch end up with all-zero progress targets.

This script writes a cleaned h5 with the same layout as the source:

    <group_name>/
        <traj_id>                  (32, 768) float32 DINOv2 embedding
        flow_progress_<traj_id>    (32,)     float32
        flow_signal_<traj_id>      (32,)     float32
        minilm_lang_embedding      (1, 384)  float32   (copied once per group)
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np


def _is_traj_key(name):
    return (not name.startswith("flow_")) and ("lang" not in name)


def filter_h5(src_path, dst_path):
    src = h5py.File(src_path, "r")
    dst = h5py.File(dst_path, "w")

    n_groups_in = len(src.keys())
    n_groups_out = 0
    n_traj_in = 0
    n_traj_kept = 0

    t0 = time.time()
    last_report = t0

    try:
        for gi, group_name in enumerate(src.keys()):
            src_group = src[group_name]
            traj_keys = [k for k in src_group.keys() if _is_traj_key(k)]
            kept = []
            for tk in traj_keys:
                n_traj_in += 1
                fs_key = f"flow_signal_{tk}"
                if fs_key not in src_group:
                    continue
                if float(np.asarray(src_group[fs_key]).max()) <= 0.0:
                    continue
                kept.append(tk)

            if not kept:
                continue

            dst_group = dst.create_group(group_name)
            for tk in kept:
                dst_group.create_dataset(
                    tk, data=np.asarray(src_group[tk]), compression=None
                )
                fp_key = f"flow_progress_{tk}"
                fs_key = f"flow_signal_{tk}"
                if fp_key in src_group:
                    dst_group.create_dataset(fp_key, data=np.asarray(src_group[fp_key]))
                if fs_key in src_group:
                    dst_group.create_dataset(fs_key, data=np.asarray(src_group[fs_key]))
                n_traj_kept += 1

            if "minilm_lang_embedding" in src_group:
                dst_group.create_dataset(
                    "minilm_lang_embedding",
                    data=np.asarray(src_group["minilm_lang_embedding"]),
                )

            n_groups_out += 1

            now = time.time()
            if now - last_report > 5.0:
                pct = (gi + 1) / n_groups_in
                rate = (gi + 1) / (now - t0)
                eta = (n_groups_in - (gi + 1)) / max(rate, 1e-6)
                print(
                    f"  [{gi+1:5d}/{n_groups_in}] {pct:6.1%}  "
                    f"kept_groups={n_groups_out}  kept_traj={n_traj_kept}  "
                    f"rate={rate:.1f} groups/s  eta={eta:5.0f}s"
                )
                last_report = now
    finally:
        src.close()
        dst.close()

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  groups: {n_groups_in} -> {n_groups_out}  (kept {n_groups_out/n_groups_in:.2%})")
    print(f"  traj:   {n_traj_in} -> {n_traj_kept}  (kept {n_traj_kept/n_traj_in:.2%})")
    src_sz = Path(src_path).resolve().stat().st_size / 1e9
    dst_sz = Path(dst_path).stat().st_size / 1e9
    print(f"  size:   {src_sz:.2f} GB -> {dst_sz:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default="datasets/rewind_openx_flow_embeddings/full_openx_flow_embeddings_train.h5",
    )
    parser.add_argument(
        "--dst",
        default="datasets/rewind_openx_flow_embeddings/full_openx_flow_embeddings_train_clean.h5",
    )
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        raise FileExistsError(
            f"{dst} already exists — refusing to overwrite. Delete it first."
        )

    print(f"src: {src}")
    print(f"dst: {dst}")
    print()
    filter_h5(str(src), str(dst))


if __name__ == "__main__":
    main()
