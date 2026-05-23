"""Generate DINOv2 + frame-diff flow_progress embeddings for the
HenryZhang/VLAReplica_SFT_data LeRobot-v3 dataset.

This dataset packs ~50 episodes into each MP4 (with per-episode
from_timestamp/to_timestamp ranges given in meta/episodes/*.parquet),
so the existing v2-targeted generate_lerobot_embeddings.py does not fit.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from data_preprocessing.generate_openx_bridge_embeddings import (
    center_crop,
    encode_text,
    get_dino_embeddings,
    sample_frames,
    sanitize_h5_key,
)
from utils.progress_utils import compute_frame_diff_progress


def _load_episodes_meta(dataset_dir, camera_key):
    paths = sorted((dataset_dir / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No meta/episodes parquet under {dataset_dir}")
    frames = [pd.read_parquet(p) for p in paths]
    meta = pd.concat(frames, ignore_index=True)

    chunk_col = f"videos/{camera_key}/chunk_index"
    file_col = f"videos/{camera_key}/file_index"
    from_col = f"videos/{camera_key}/from_timestamp"
    to_col = f"videos/{camera_key}/to_timestamp"
    for c in (chunk_col, file_col, from_col, to_col, "length", "tasks", "episode_index"):
        if c not in meta.columns:
            raise KeyError(f"episodes meta missing column {c!r}")
    return meta, chunk_col, file_col, from_col, to_col


def _video_path(dataset_dir, camera_key, chunk_index, file_index):
    return (
        dataset_dir
        / "videos"
        / camera_key
        / f"chunk-{int(chunk_index):03d}"
        / f"file-{int(file_index):03d}.mp4"
    )


def _episode_task_text(tasks_field):
    if isinstance(tasks_field, np.ndarray):
        tasks_field = tasks_field.tolist()
    if isinstance(tasks_field, (list, tuple)):
        if not tasks_field:
            return ""
        return str(tasks_field[0])
    return str(tasks_field)


def _plan_frames_for_video(episodes_in_video, fps, max_length, total_video_frames):
    """For each episode that lives in this video, compute the absolute frame
    indices we want to decode (subsampled to max_length frames). Returns:
      - per_episode: list of dicts with episode_index, task, slots (target indices)
      - needed_frames: dict {abs_frame_idx -> [(ep_pos_in_list, slot_in_episode)]}
    """
    per_episode = []
    needed = defaultdict(list)
    for ep_pos, row in enumerate(episodes_in_video):
        length = int(row["length"])
        from_ts = float(row["from_ts"])
        # Convert episode-relative frame range to absolute frame indices in the video.
        # round(from_ts * fps) is the start frame of this episode inside the video.
        ep_start_abs = int(round(from_ts * fps))
        ep_end_abs = min(ep_start_abs + length, total_video_frames)
        ep_len_abs = ep_end_abs - ep_start_abs
        if ep_len_abs < 3:
            per_episode.append(None)
            continue

        n = min(max_length, ep_len_abs)
        rel_indices = np.linspace(0, ep_len_abs - 1, n, dtype=int)
        abs_indices = ep_start_abs + rel_indices
        for slot, abs_idx in enumerate(abs_indices):
            needed[int(abs_idx)].append((ep_pos, slot))
        per_episode.append(
            {
                "episode_index": int(row["episode_index"]),
                "task": row["task"],
                "n_frames": int(n),
            }
        )
    return per_episode, needed


def _decode_needed_frames(video_path, needed):
    """Stream the video once and pull out only the frames we need.
    Returns dict {abs_idx -> np.uint8 frame (H, W, 3)}.
    """
    out = {}
    if not needed:
        return out
    target_max = max(needed.keys())
    reader = imageio.get_reader(str(video_path), "ffmpeg")
    try:
        for i, frame in enumerate(reader):
            if i in needed:
                if frame.ndim == 2:
                    frame = np.repeat(frame[:, :, None], 3, axis=2)
                out[i] = np.asarray(frame[:, :, :3], dtype=np.uint8)
            if i >= target_max:
                break
    finally:
        reader.close()
    missing = [k for k in needed if k not in out]
    if missing:
        raise RuntimeError(
            f"Failed to decode {len(missing)} frames from {video_path}; "
            f"first few missing: {missing[:5]}"
        )
    return out


def build_vla_replica_h5(
    dataset_dir,
    output_path,
    camera_key,
    max_length,
    max_episodes,
    crop_size,
):
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    fps = float(info["fps"])

    meta, chunk_col, file_col, from_col, to_col = _load_episodes_meta(dataset_dir, camera_key)
    meta = meta.copy()
    meta["task"] = meta["tasks"].map(_episode_task_text)
    meta["chunk_index"] = meta[chunk_col].astype(int)
    meta["file_index"] = meta[file_col].astype(int)
    meta["from_ts"] = meta[from_col].astype(float)
    meta["to_ts"] = meta[to_col].astype(float)
    meta = meta.sort_values(["chunk_index", "file_index", "episode_index"]).reset_index(drop=True)

    if max_episodes > 0:
        meta = meta.head(max_episodes).reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14", force_reload=False
    ).to(device)
    dino_model.eval()
    minilm_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(device)
    minilm_model.eval()

    per_group_counts = defaultdict(int)
    lang_cache = {}

    with h5py.File(output_path, "w") as h5_file:
        for (chunk_index, file_index), group_df in tqdm(
            meta.groupby(["chunk_index", "file_index"], sort=True),
            desc=f"videos[{camera_key}]",
        ):
            video_path = _video_path(dataset_dir, camera_key, chunk_index, file_index)
            if not video_path.exists():
                raise FileNotFoundError(f"Missing video {video_path}")

            reader = imageio.get_reader(str(video_path), "ffmpeg")
            try:
                total_video_frames = int(getattr(reader, "count_frames", lambda: 0)())
            finally:
                reader.close()

            episodes_in_video = group_df.to_dict("records")
            per_episode, needed = _plan_frames_for_video(
                episodes_in_video, fps, max_length, total_video_frames
            )
            decoded = _decode_needed_frames(video_path, needed)

            # Reassemble per-episode frame stacks
            episode_frames = [
                np.zeros((info_e["n_frames"], 480, 640, 3), dtype=np.uint8)
                if info_e is not None else None
                for info_e in per_episode
            ]
            for abs_idx, slots in needed.items():
                frame = decoded[abs_idx]
                for ep_pos, slot in slots:
                    episode_frames[ep_pos][slot] = frame

            for ep_pos, info_e in enumerate(per_episode):
                if info_e is None:
                    continue
                frames = episode_frames[ep_pos]
                sampled_frames = [center_crop(f, crop_size) for f in frames]

                flow_progress, flow_signal = compute_frame_diff_progress(sampled_frames)
                dino_embeddings = get_dino_embeddings(
                    sampled_frames, dino_model=dino_model, device=device
                )

                instruction = info_e["task"] or f"vla_replica_episode_{info_e['episode_index']}"
                group_name = sanitize_h5_key(instruction)
                if group_name not in h5_file:
                    h5_file.create_group(group_name)

                traj_id = str(per_group_counts[group_name])
                per_group_counts[group_name] += 1

                h5_file[group_name].create_dataset(traj_id, data=dino_embeddings)
                h5_file[group_name].create_dataset(f"flow_progress_{traj_id}", data=flow_progress)
                h5_file[group_name].create_dataset(f"flow_signal_{traj_id}", data=flow_signal)

                if "minilm_lang_embedding" not in h5_file[group_name]:
                    if instruction not in lang_cache:
                        lang_cache[instruction] = encode_text(
                            instruction,
                            tokenizer=minilm_tokenizer,
                            model=minilm_model,
                            device=device,
                        )
                    h5_file[group_name].create_dataset(
                        "minilm_lang_embedding", data=lang_cache[instruction]
                    )

    print(f"Wrote {output_path}")
    print(f"Total task groups: {len(per_group_counts)}, total trajectories: {sum(per_group_counts.values())}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="datasets/VLAReplica_SFT_data")
    parser.add_argument("--output-path", default="datasets/vla_replica_sft_embeddings_train.h5")
    parser.add_argument("--camera-key", default="observation.images.top")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--crop-size", type=int, default=224)
    args = parser.parse_args()

    build_vla_replica_h5(
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        camera_key=args.camera_key,
        max_length=args.max_length,
        max_episodes=args.max_episodes,
        crop_size=args.crop_size,
    )


if __name__ == "__main__":
    main()
