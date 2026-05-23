"""Confusion matrix + reward-alignment metrics for a trained VLA Replica
reward model. Reuses the original metaworld methodology
(`generate_rewind_data`, `plot_matrix_as_image_for_paper`,
`compute_correlation_from_sequences`) so numbers are directly comparable
across runs.

Two figure styles per checkpoint:
  - paper: min-max normalized, no axis labels (calls the original
    `plot_matrix_as_image_for_paper`; logs to wandb +
    `confusion_matrix_for_paper/<run>/...pdf` when --pdf)
  - diagnostic: absolute reward values, axis labels with instruction
    names, per-cell numbers. PNG only.

Runs against any combination of eval h5 files (seen + unseen) merged
in-memory, so we get one matrix covering all instructions.

Example:
    python -m utils.vla_confusion_matrix \
        --checkpoint checkpoints_flow_freeze/rewind_metaworld_epoch_19.pth \
        --use-wandb --pdf
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import ReWiNDTransformer
from utils.eval_confusion_matrix import plot_matrix_as_image_for_paper
from utils.utils import (
    compute_correlation_from_sequences,
    generate_rewind_data,
)

matplotlib.use("Agg")


# 15 instructions in the filtered dataset map to 5 user-defined task categories.
INSTRUCTION_TO_CATEGORY = {
    # Put bread on plate
    "Put the bread on the red plate.": "Put bread on plate",
    "Put the bread on the blue plate.": "Put bread on plate",
    # Put bowl on coaster
    "Put the yellow bowl on the purple coaster.": "Put bowl on coaster",
    "Put the red bowl on the green coaster.": "Put bowl on coaster",
    "Put the blue bowl on the orange coaster.": "Put bowl on coaster",
    "Put the blue bowl on the green coaster.": "Put bowl on coaster",
    "Put the red bowl on the purple coaster.": "Put bowl on coaster",
    # Stack block on block
    "Stack the red block on the blue block.": "Stack block on block",
    "Stack the blue block on the red block.": "Stack block on block",
    "Stack the blue block on the yellow block.": "Stack block on block",
    "Stack the yellow block on the blue block.": "Stack block on block",
    "Stack the red block on the yellow block.": "Stack block on block",
    # Fold towel
    "Fold the pink towel in half.": "Fold towel",
    "Fold the yellow towel in half.": "Fold towel",
    # Open oven
    "Open the oven.": "Open oven",
}

CATEGORY_ORDER = [
    "Put bread on plate",
    "Put bowl on coaster",
    "Stack block on block",
    "Fold towel",
    "Open oven",
]


def collapse_matrix_to_categories(matrix, instructions, n_videos_per_inst):
    """Aggregate (N_inst, N_inst) matrix → (5, 5) by mapping instructions to
    their categories. Each (cat_i, cat_j) cell is the mean over all
    (video v ∈ cat_i, language l ∈ cat_j) pairs, weighted by the number of
    videos contributing to row v (since the per-instruction confusion cell
    is already mean-over-videos)."""
    cat_to_inst_rows = {c: [] for c in CATEGORY_ORDER}
    for i, inst in enumerate(instructions):
        cat = INSTRUCTION_TO_CATEGORY.get(inst)
        if cat is None:
            raise KeyError(f"No category mapping for {inst!r}")
        cat_to_inst_rows[cat].append((i, n_videos_per_inst[i]))

    n_cats = len(CATEGORY_ORDER)
    out = np.zeros((n_cats, n_cats), dtype=np.float32)
    cat_n_videos = []
    for ci, cat_i in enumerate(CATEGORY_ORDER):
        rows = cat_to_inst_rows[cat_i]
        weights_i = np.array([w for _, w in rows], dtype=np.float32)
        cat_n_videos.append(int(weights_i.sum()))
        for cj, cat_j in enumerate(CATEGORY_ORDER):
            cols = cat_to_inst_rows[cat_j]
            # Each matrix cell is already a per-video mean; to combine across
            # instructions in a column we average uniformly, and across
            # instructions in a row we weight by video count (so a tiny
            # 2-traj instruction doesn't dominate a 50-traj one).
            sub = np.array(
                [
                    np.mean([matrix[i, j] for j, _ in cols])
                    for i, _ in rows
                ],
                dtype=np.float32,
            )
            out[ci, cj] = float(np.average(sub, weights=weights_i))
    return out, CATEGORY_ORDER, cat_n_videos


def _merge_eval_h5s(paths):
    """Open paths and copy every group into a fresh in-memory h5.
    Caller is responsible for closing the returned handle."""
    combined = h5py.File("vla_replica_merged_eval.h5",
                         "w", driver="core", backing_store=False)
    seen_groups = set()
    for p in paths:
        with h5py.File(p, "r") as src:
            for group_name in src.keys():
                if group_name in seen_groups:
                    raise ValueError(
                        f"Duplicate instruction group {group_name!r} across input h5s; "
                        "filter_vla_replica.py should keep seen/unseen disjoint."
                    )
                src.copy(group_name, combined)
                seen_groups.add(group_name)
    return combined


def _short(s, n=42):
    return s if len(s) <= n else s[: n - 1] + "…"


def plot_diagnostic(matrix, instructions, out_path, title, n_videos_per_task=None):
    """Labeled, absolute-value, per-cell-numbered view for inspection."""
    fig, ax = plt.subplots(
        figsize=(max(8, len(instructions) * 0.7),
                 max(8, len(instructions) * 0.6))
    )
    vmax = max(1.0, float(np.nanmax(matrix)))
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=vmax)

    labels = [
        _short(inst) + (f"  (n={n_videos_per_task[i]})" if n_videos_per_task else "")
        for i, inst in enumerate(instructions)
    ]
    ax.set_xticks(range(len(instructions)))
    ax.set_yticks(range(len(instructions)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Language prompt (column)")
    ax.set_ylabel("Video task (row)")
    ax.set_title(title, fontsize=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center",
                    color="white" if v > 0.55 * vmax else "black",
                    fontsize=7)

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved diagnostic PNG: {out_path}")
    return fig


def _diag_stats(matrix):
    diag = np.diag(matrix).mean()
    off_total = matrix.sum() - np.diag(matrix).sum()
    n_off = matrix.size - matrix.shape[0]
    off = off_total / n_off if n_off > 0 else float("nan")
    return float(diag), float(off), float(diag - off)


def run_for_checkpoint(ckpt_path, eval_h5_paths, output_dir, args):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = state["args"]
    saved_args.max_length = args.max_length
    saved_args.subsample_video = True
    saved_args.eval_max_samples = -1
    saved_args.pdf = args.pdf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReWiNDTransformer(
        args=saved_args, video_dim=768, text_dim=384, hidden_dim=512
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    epoch = state.get("epoch")
    print(f"Loaded {ckpt_path} (saved epoch {epoch})")

    combined = _merge_eval_h5s(eval_h5_paths)
    try:
        instructions = sorted(combined.keys())
        n_videos = [
            len([k for k in combined[g].keys() if k.isdigit()])
            for g in instructions
        ]
        task_subset = {"eval_tasks": instructions}

        confusion_matrix, all_seqs, tasks, _ = generate_rewind_data(
            h5_file=combined,
            task_subset=task_subset,
            set_type="eval",
            rewind_model=model,
            args=saved_args,
        )
    finally:
        combined.close()

    diag, off, gap = _diag_stats(confusion_matrix)
    print(f"[full 15x15] diag_mean={diag:.4f}  off_diag_mean={off:.4f}  gap={gap:+.4f}")

    inst_tasks = list(tasks)
    inst_all_seqs = all_seqs
    if args.collapse_by_category:
        confusion_matrix, tasks, n_videos = collapse_matrix_to_categories(
            confusion_matrix, tasks, n_videos
        )
        cat_diag, cat_off, cat_gap = _diag_stats(confusion_matrix)
        print(f"[5x5 category] diag_mean={cat_diag:.4f}  off_diag_mean={cat_off:.4f}  gap={cat_gap:+.4f}")
        diag, off, gap = cat_diag, cat_off, cat_gap

    run_name = f"vla_replica_{ckpt_path.stem}"
    if args.collapse_by_category:
        run_name += "_5cat"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Paper-style figure (min-max normalized, no labels, wandb + optional pdf)
    plot_matrix_as_image_for_paper(
        args=saved_args,
        matrix=confusion_matrix,
        names=tasks,
        set=f"vla_{ckpt_path.stem}",
        text=None,
        epoch=epoch,
        run_name=run_name,
    )

    # 2) Diagnostic figure (absolute values, labels, numbers)
    diag_png = output_dir / f"diagnostic_{ckpt_path.stem}.png"
    fig = plot_diagnostic(
        matrix=confusion_matrix,
        instructions=tasks,
        out_path=diag_png,
        title=(
            f"VLA Replica confusion (last-frame reward, original convention)\n"
            f"{ckpt_path.name} | rows=video task, cols=language prompt"
        ),
        n_videos_per_task=n_videos,
    )

    # 3) Pearson/Spearman vs linear-GT on diagonal (original convention).
    # Correlations are inherently per-instruction (one diagonal sequence per
    # instruction), so we always evaluate on the 15 instructions regardless
    # of whether the matrix was collapsed for plotting.
    pearson_avg, _, spearman_avg, _ = compute_correlation_from_sequences(
        all_seqs=inst_all_seqs,
        env_names=inst_tasks,
        set_type="vla_eval",
        epoch=epoch,
    )

    import wandb
    if wandb.run is not None:
        wandb.log({
            "confusion/diag_mean": diag,
            "confusion/off_diag_mean": off,
            "confusion/gap": gap,
            "confusion/Average_Pearson_vs_LinearGT": pearson_avg,
            "confusion/Average_Spearman_vs_LinearGT": spearman_avg,
            "confusion/diagnostic_image": wandb.Image(fig, caption=str(diag_png)),
            "epoch": epoch,
        })

    plt.close(fig)
    return {
        "epoch": epoch,
        "diag_mean": diag,
        "off_diag_mean": off,
        "gap": gap,
        "pearson_vs_linear": pearson_avg,
        "spearman_vs_linear": spearman_avg,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="One or more checkpoint .pth paths.")
    parser.add_argument(
        "--eval-h5", nargs="+",
        default=[
            "datasets/vla_replica_filtered_eval_seen.h5",
            "datasets/vla_replica_filtered_eval_unseen.h5",
        ],
        help="Eval h5 files to merge; default = seen + unseen.",
    )
    parser.add_argument("--output-dir", default="confusion_matrix_for_paper/vla_replica")
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--pdf", action="store_true",
                        help="Save the paper figure to PDF "
                             "under confusion_matrix_for_paper/<run>/.")
    parser.add_argument("--collapse-by-category", action="store_true",
                        help="Aggregate the 15-instruction matrix into the "
                             "5 user-defined task categories (bread/bowl/stack/fold/oven).")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", default="yusenluo")
    parser.add_argument("--wandb-project", default="rewind-vla-replica-flow")
    args = parser.parse_args()

    if args.use_wandb:
        import wandb
        run_label = "+".join(Path(c).stem for c in args.checkpoint)[:60]
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"vla_confusion_{run_label}",
            config={
                "checkpoints": [str(c) for c in args.checkpoint],
                "eval_h5": [str(h) for h in args.eval_h5],
            },
        )

    output_dir = Path(args.output_dir)
    rows = []
    for ckpt in args.checkpoint:
        rows.append(run_for_checkpoint(Path(ckpt), args.eval_h5, output_dir, args))

    print("\n=== Summary ===")
    print(f"{'epoch':>5}  {'diag':>6}  {'off':>6}  {'gap':>7}  {'pearson':>8}  {'spearman':>8}")
    for r in rows:
        print(
            f"{str(r['epoch']):>5}  {r['diag_mean']:6.3f}  {r['off_diag_mean']:6.3f}  "
            f"{r['gap']:+7.3f}  {r['pearson_vs_linear']:+8.3f}  {r['spearman_vs_linear']:+8.3f}"
        )

    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
