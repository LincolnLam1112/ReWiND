"""Holdout evaluation for the VLA Replica reward model.

For each (group, trajectory) pair in an eval h5:
  - predict reward sequence with the CORRECT language instruction
  - predict reward sequence with a WRONG language instruction
    (randomly sampled from a different group in the same eval h5; if the
    h5 only has one group, draw from `negative_pool_h5` instead)

Aggregate per-trajectory metrics across the whole eval h5 and log to wandb
under a configurable prefix.
"""

import random

import h5py
import numpy as np
import torch
import wandb
from scipy.stats import spearmanr


def _list_traj_keys(group):
    return sorted(
        k for k in group.keys()
        if not k.startswith("flow_") and "lang" not in k
    )


def _pad_or_truncate(seq, max_length):
    """seq: numpy array of shape (T, D) or (T,). Returns (max_length, D)/(max_length,)
    via last-frame padding or linspace subsampling, mirroring dataset.padding_video."""
    t = seq.shape[0]
    if t == max_length:
        return seq
    if t > max_length:
        idx = np.linspace(0, t - 1, max_length, dtype=int)
        return seq[idx]
    pad_n = max_length - t
    pad = np.repeat(seq[-1:], pad_n, axis=0)
    return np.concatenate([seq, pad], axis=0)


def _load_lang(group):
    lang = np.asarray(group["minilm_lang_embedding"], dtype=np.float32)
    if lang.ndim == 2:
        lang = lang[0]
    return lang  # (384,)


def _sample_wrong_lang(eval_h5, exclude_group, rng, negative_pool_h5=None):
    """Return a (384,) language embedding from a group that is NOT `exclude_group`.
    Falls back to negative_pool_h5 if eval_h5 has only one group."""
    candidates = [g for g in eval_h5.keys() if g != exclude_group]
    if not candidates and negative_pool_h5 is not None:
        candidates = [g for g in negative_pool_h5.keys() if g != exclude_group]
        pool = negative_pool_h5
    else:
        pool = eval_h5
    if not candidates:
        return None
    chosen = rng.choice(candidates)
    return _load_lang(pool[chosen])


@torch.no_grad()
def _predict_reward(rewind_model, video_padded, lang, device, max_length):
    """video_padded: (max_length, D) numpy. lang: (384,) numpy.
    Returns numpy reward array of length max_length."""
    v = torch.from_numpy(video_padded).float().to(device).unsqueeze(0)  # (1, T, D)
    l = torch.from_numpy(lang).float().to(device).unsqueeze(0)          # (1, 384)
    out = rewind_model(v, l).squeeze(0).squeeze(-1).detach().cpu().numpy()  # (T,)
    return out


def compute_vla_eval(
    args,
    rewind_model,
    eval_h5_path,
    epoch,
    log_prefix="eval_seen",
    negative_pool_h5_path=None,
    seed=0,
):
    """Run holdout eval over a single h5 and log per-prefix metrics.

    Returns the dict of metrics that was logged."""
    rewind_model.eval()
    device = next(rewind_model.parameters()).device
    rng = random.Random(seed + epoch)
    max_length = args.max_length

    eval_h5 = h5py.File(eval_h5_path, "r")
    neg_h5 = h5py.File(negative_pool_h5_path, "r") if negative_pool_h5_path else None

    mses, spearmans, monos = [], [], []
    pos_means, neg_means = [], []
    pos_peaks, neg_peaks = [], []
    pos_end, neg_end = [], []
    per_inst_spearman = {}

    try:
        for group_name in eval_h5.keys():
            group = eval_h5[group_name]
            traj_keys = _list_traj_keys(group)
            if not traj_keys:
                continue
            true_lang = _load_lang(group)
            inst_spearmans = []
            for tk in traj_keys:
                video = np.asarray(group[tk], dtype=np.float32)            # (T_raw, D)
                target_full = np.asarray(group[f"flow_progress_{tk}"], dtype=np.float32)

                video_padded = _pad_or_truncate(video, max_length)
                target_padded = _pad_or_truncate(target_full, max_length)

                pos_pred = _predict_reward(rewind_model, video_padded, true_lang, device, max_length)
                wrong_lang = _sample_wrong_lang(eval_h5, group_name, rng, negative_pool_h5=neg_h5)
                if wrong_lang is not None:
                    neg_pred = _predict_reward(rewind_model, video_padded, wrong_lang, device, max_length)
                else:
                    neg_pred = np.zeros_like(pos_pred)

                mses.append(float(np.mean((pos_pred - target_padded) ** 2)))
                if pos_pred.std() > 1e-6 and target_padded.std() > 1e-6:
                    corr, _ = spearmanr(pos_pred, target_padded)
                    if not np.isnan(corr):
                        spearmans.append(float(corr))
                        inst_spearmans.append(float(corr))
                if len(pos_pred) > 1:
                    monos.append(float(np.mean(np.diff(pos_pred) >= 0)))
                pos_means.append(float(pos_pred.mean()))
                neg_means.append(float(neg_pred.mean()))
                pos_peaks.append(float(pos_pred.max()))
                neg_peaks.append(float(neg_pred.max()))
                pos_end.append(float(pos_pred[-1]))
                neg_end.append(float(neg_pred[-1]))

            if inst_spearmans:
                per_inst_spearman[group_name] = float(np.mean(inst_spearmans))
    finally:
        eval_h5.close()
        if neg_h5 is not None:
            neg_h5.close()

    log = {
        f"{log_prefix}/mse": float(np.mean(mses)) if mses else float("nan"),
        f"{log_prefix}/spearman": float(np.mean(spearmans)) if spearmans else float("nan"),
        f"{log_prefix}/monotonicity_rate": float(np.mean(monos)) if monos else float("nan"),
        f"{log_prefix}/pos_pred_mean": float(np.mean(pos_means)) if pos_means else float("nan"),
        f"{log_prefix}/neg_pred_mean": float(np.mean(neg_means)) if neg_means else float("nan"),
        f"{log_prefix}/pos_pred_peak": float(np.mean(pos_peaks)) if pos_peaks else float("nan"),
        f"{log_prefix}/neg_pred_peak": float(np.mean(neg_peaks)) if neg_peaks else float("nan"),
        f"{log_prefix}/pos_pred_end": float(np.mean(pos_end)) if pos_end else float("nan"),
        f"{log_prefix}/neg_pred_end": float(np.mean(neg_end)) if neg_end else float("nan"),
        f"{log_prefix}/discrimination_mean_gap":
            float(np.mean(pos_means) - np.mean(neg_means)) if pos_means and neg_means else float("nan"),
        f"{log_prefix}/discrimination_peak_gap":
            float(np.mean(pos_peaks) - np.mean(neg_peaks)) if pos_peaks and neg_peaks else float("nan"),
        f"{log_prefix}/n_trajectories": len(pos_means),
        "epoch": epoch,
    }
    for inst, s in per_inst_spearman.items():
        safe = inst.replace("/", "_")[:60]
        log[f"{log_prefix}_per_inst_spearman/{safe}"] = s
    if wandb.run is not None:
        wandb.log(log)
    return log
