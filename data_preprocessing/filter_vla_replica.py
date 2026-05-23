"""Filter the full vla_replica_sft h5 down to 15 selected instructions and
split into three h5 files for training and two-way evaluation:

  train.h5         — 80% of trajectories per *seen* instruction
  eval_seen.h5     — remaining 20% of trajectories per *seen* instruction
                     (tests same-task generalization to new trajectories)
  eval_unseen.h5   — 100% of trajectories of 4 *held-out* instructions
                     (tests language-side generalization to unseen
                     object/color combinations within familiar task families)

The 15 instructions covered span 5 task categories chosen by the user:
bread-on-plate, bowl-on-coaster, stack-block-on-block, fold-towel, open-oven.
"Open the oven" is the only single-instruction category, so nothing is
held out for it (eval_unseen has 4 instructions, not 5).
"""

import argparse
import random
from pathlib import Path

import h5py
import numpy as np

# Exact instruction strings (must match h5 group_name sanitization).
SEEN_INSTRUCTIONS = [
    "Put the bread on the red plate.",
    "Put the yellow bowl on the purple coaster.",
    "Put the red bowl on the green coaster.",
    "Put the blue bowl on the orange coaster.",
    "Put the blue bowl on the green coaster.",
    "Stack the red block on the blue block.",
    "Stack the blue block on the red block.",
    "Stack the blue block on the yellow block.",
    "Stack the yellow block on the blue block.",
    "Fold the pink towel in half.",
    "Open the oven.",
]

# One held-out instruction per multi-instruction category.
# Open oven has no held-out twin (only 1 instruction in dataset).
HELD_OUT_INSTRUCTIONS = [
    "Put the bread on the blue plate.",
    "Put the red bowl on the purple coaster.",
    "Stack the red block on the yellow block.",
    "Fold the yellow towel in half.",
]


def _is_traj_key(k):
    return not k.startswith("flow_") and "lang" not in k


def _copy_traj(src_group, dst_group, traj_key):
    dst_group.create_dataset(traj_key, data=np.asarray(src_group[traj_key]))
    fp = f"flow_progress_{traj_key}"
    fs = f"flow_signal_{traj_key}"
    if fp in src_group:
        dst_group.create_dataset(fp, data=np.asarray(src_group[fp]))
    if fs in src_group:
        dst_group.create_dataset(fs, data=np.asarray(src_group[fs]))


def _copy_lang(src_group, dst_group):
    if "minilm_lang_embedding" in src_group and "minilm_lang_embedding" not in dst_group:
        dst_group.create_dataset(
            "minilm_lang_embedding", data=np.asarray(src_group["minilm_lang_embedding"])
        )


def build_split(src_h5_path, out_dir, eval_ratio, seed):
    src = h5py.File(src_h5_path, "r")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "vla_replica_filtered_train.h5"
    eval_seen_path = out_dir / "vla_replica_filtered_eval_seen.h5"
    eval_unseen_path = out_dir / "vla_replica_filtered_eval_unseen.h5"

    missing_seen = [n for n in SEEN_INSTRUCTIONS if n not in src]
    missing_held = [n for n in HELD_OUT_INSTRUCTIONS if n not in src]
    if missing_seen or missing_held:
        raise KeyError(
            f"Source h5 missing instructions: seen={missing_seen}, held_out={missing_held}"
        )
    overlap = set(SEEN_INSTRUCTIONS) & set(HELD_OUT_INSTRUCTIONS)
    if overlap:
        raise ValueError(f"Instruction listed in both seen and held-out: {overlap}")

    rng = random.Random(seed)
    summary = []

    train_h5 = h5py.File(train_path, "w")
    eval_seen_h5 = h5py.File(eval_seen_path, "w")
    try:
        for inst in SEEN_INSTRUCTIONS:
            src_group = src[inst]
            traj_keys = sorted([k for k in src_group.keys() if _is_traj_key(k)])
            rng.shuffle(traj_keys)
            n_eval = max(1, int(round(len(traj_keys) * eval_ratio)))
            eval_keys = traj_keys[:n_eval]
            train_keys = traj_keys[n_eval:]

            train_group = train_h5.create_group(inst)
            eval_group = eval_seen_h5.create_group(inst)
            for tk in train_keys:
                _copy_traj(src_group, train_group, tk)
            for tk in eval_keys:
                _copy_traj(src_group, eval_group, tk)
            _copy_lang(src_group, train_group)
            _copy_lang(src_group, eval_group)
            summary.append(("seen", inst, len(train_keys), len(eval_keys), 0))
    finally:
        train_h5.close()
        eval_seen_h5.close()

    eval_unseen_h5 = h5py.File(eval_unseen_path, "w")
    try:
        for inst in HELD_OUT_INSTRUCTIONS:
            src_group = src[inst]
            traj_keys = sorted([k for k in src_group.keys() if _is_traj_key(k)])
            unseen_group = eval_unseen_h5.create_group(inst)
            for tk in traj_keys:
                _copy_traj(src_group, unseen_group, tk)
            _copy_lang(src_group, unseen_group)
            summary.append(("held_out", inst, 0, 0, len(traj_keys)))
    finally:
        eval_unseen_h5.close()
    src.close()

    print(f"\nWrote:\n  {train_path}\n  {eval_seen_path}\n  {eval_unseen_path}\n")
    print(f"{'kind':10s} {'instruction':55s} {'train':>5s} {'eval_seen':>10s} {'eval_unseen':>11s}")
    n_train = n_eval_seen = n_eval_unseen = 0
    for kind, inst, nt, nes, neu in summary:
        print(f"{kind:10s} {inst:55s} {nt:5d} {nes:10d} {neu:11d}")
        n_train += nt
        n_eval_seen += nes
        n_eval_unseen += neu
    print(f"{'TOTAL':10s} {'':55s} {n_train:5d} {n_eval_seen:10d} {n_eval_unseen:11d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-h5", default="datasets/vla_replica_sft_embeddings_train.h5"
    )
    parser.add_argument("--out-dir", default="datasets")
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_split(args.src_h5, args.out_dir, args.eval_ratio, args.seed)


if __name__ == "__main__":
    main()
