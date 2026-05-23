import torch
import wandb
import math
from torch.optim import Optimizer
from torch.nn.functional import mse_loss

from utils.progress_utils import compute_directional_penalty

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _summarize_progress_deltas(progress_targets, rewind_mask):
    deltas = progress_targets[:, 1:] - progress_targets[:, :-1]
    rewind_steps = rewind_mask[:, 1:] > 0.5
    forward_steps = ~rewind_steps

    zero = deltas.new_tensor(0.0)

    if forward_steps.any():
        forward_delta_mean = deltas[forward_steps].mean()
        forward_decrease_rate = (deltas[forward_steps] < 0).float().mean()
    else:
        forward_delta_mean = zero
        forward_decrease_rate = zero

    if rewind_steps.any():
        rewind_delta_mean = deltas[rewind_steps].mean()
        rewind_non_decreasing_rate = (deltas[rewind_steps] >= 0).float().mean()
    else:
        rewind_delta_mean = zero
        rewind_non_decreasing_rate = zero

    rewind_step_rate = rewind_steps.float().mean() if rewind_steps.numel() > 0 else zero
    return (
        forward_delta_mean.detach(),
        forward_decrease_rate.detach(),
        rewind_delta_mean.detach(),
        rewind_non_decreasing_rate.detach(),
        rewind_step_rate.detach(),
    )

class CosineWithMinLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, max_steps: int, max_lr: float, min_lr: float, last_epoch: int = -1):
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.max_steps:
            # Cosine decay for the first max_steps
            cos_decay = 0.5 * (1 + math.cos(math.pi * self.last_epoch / self.max_steps))
            return [self.min_lr + (self.max_lr - self.min_lr) * cos_decay for _ in self.base_lrs]
        else:
            # Keep the minimum learning rate
            return [self.min_lr for _ in self.base_lrs]

def _train_step_openx_only(args, openx_data, rewind_model, optimizer, scheduler):
    rewind_model.train()
    optimizer.zero_grad()

    pos_video = openx_data["video_array"].to(device).float()
    pos_text = openx_data["text_array"].squeeze(1).to(device).float()
    pos_progress = openx_data["progress"].to(device).float()
    pos_goal_distance = openx_data["goal_distance"].to(device).float()
    pos_rewind_mask = openx_data["rewind_mask"].to(device).float()

    n = pos_video.shape[0]
    roll = max(1, n // 2)
    neg_video = torch.roll(pos_video, roll, 0)
    neg_text = pos_text.clone()
    neg_progress = torch.zeros_like(pos_progress)
    neg_goal_distance = torch.zeros_like(pos_goal_distance)
    neg_rewind_mask = torch.zeros_like(pos_rewind_mask)

    video = torch.cat([pos_video, neg_video], dim=0)
    text = torch.cat([pos_text, neg_text], dim=0)
    progress = torch.cat([pos_progress, neg_progress], dim=0)
    goal_distance = torch.cat([pos_goal_distance, neg_goal_distance], dim=0)
    rewind_mask = torch.cat([pos_rewind_mask, neg_rewind_mask], dim=0)
    positive_mask = torch.cat([torch.ones(n), torch.zeros(n)], dim=0).bool().to(device)

    progress_pred = rewind_model(video, text)
    progress_loss = mse_loss(progress_pred[:, 1:].squeeze(-1), progress[:, 1:])

    directional_loss = progress_loss.new_tensor(0.0)
    directional_violation_rate = progress_loss.new_tensor(0.0)
    away_step_rate = progress_loss.new_tensor(0.0)
    if args.lambda_dir > 0:
        directional_loss, directional_violation_rate, away_step_rate = compute_directional_penalty(
            reward_predictions=progress_pred[positive_mask],
            goal_distances=goal_distance[positive_mask],
            tau_away=args.tau_away,
            margin=args.margin,
        )

    (
        forward_delta_mean,
        forward_decrease_rate,
        rewind_delta_mean,
        rewind_non_decreasing_rate,
        rewind_step_rate,
    ) = _summarize_progress_deltas(
        progress_targets=progress[positive_mask],
        rewind_mask=rewind_mask[positive_mask],
    )

    loss = (args.lambda_prog * progress_loss) + (args.lambda_dir * directional_loss)
    loss.backward()
    if args.clip_grad:
        torch.nn.utils.clip_grad_norm_(rewind_model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    wandb.log({
        "train/loss": loss.item(),
        "train/progress_loss": progress_loss.item(),
        "train/directional_loss": directional_loss.item(),
        "train/directional_violation_rate": directional_violation_rate.item(),
        "train/away_step_rate": away_step_rate.item(),
        "train/progress_target_mean": progress[positive_mask].mean().item(),
        "train/progress_target_start": progress[positive_mask][:, 0].mean().item(),
        "train/progress_target_end": progress[positive_mask][:, -1].mean().item(),
        "train/goal_distance_mean": goal_distance[positive_mask].mean().item(),
        "train/forward_progress_delta_mean": forward_delta_mean.item(),
        "train/forward_progress_decrease_rate": forward_decrease_rate.item(),
        "train/rewind_progress_delta_mean": rewind_delta_mean.item(),
        "train/rewind_progress_non_decreasing_rate": rewind_non_decreasing_rate.item(),
        "train/rewind_step_rate": rewind_step_rate.item(),
        "lr": optimizer.param_groups[0]["lr"],
    })
    return loss.item()


def train_step_fn(args, batch, rewind_model, optimizer, scheduler):
    #set to cuda
    openx_data, extra_data = batch
    if extra_data is None:
        return _train_step_openx_only(args, openx_data, rewind_model, optimizer, scheduler)
    openx_len = len(openx_data["video_array"])
    extra_len = len(extra_data["video_array"])

    rewind_model.train()
    optimizer.zero_grad()
    positive_video_array = torch.cat([openx_data["video_array"], extra_data["video_array"]], dim = 0).to(device).float()
    
    positive_text_array = torch.cat([openx_data["text_array"].squeeze(1), extra_data["text_array"].squeeze(1)], dim=0).to(device).float()
    # positive_text_array = torch.cat([openx_data["text_array"].squeeze(1), extra_data["text_array"].squeeze()], dim = 0).to(device).float()              
    positive_progress = torch.cat([openx_data["progress"], extra_data["progress"]], dim = 0).to(device)
    positive_goal_distance = torch.cat([openx_data["goal_distance"], extra_data["goal_distance"]], dim = 0).to(device).float()
    positive_rewind_mask = torch.cat([openx_data["rewind_mask"], extra_data["rewind_mask"]], dim = 0).to(device).float()

    negative_video_array_1 = torch.roll(positive_video_array, extra_len, 0)
    negative_text_array_1 = positive_text_array.clone()
    negative_progress_1 = torch.zeros_like(positive_progress)
    negative_goal_distance_1 = torch.zeros_like(positive_goal_distance)
    negative_rewind_mask_1 = torch.zeros_like(positive_rewind_mask)

    openx_pos_video_array = torch.cat([positive_video_array[:openx_len], negative_video_array_1[:openx_len]], dim = 0)
    openx_pos_text_array = torch.cat([positive_text_array[:openx_len], negative_text_array_1[:openx_len]], dim = 0)
    openx_pos_progress = torch.cat([positive_progress[:openx_len], negative_progress_1[:openx_len]], dim = 0)
    openx_pos_goal_distance = torch.cat([positive_goal_distance[:openx_len], negative_goal_distance_1[:openx_len]], dim = 0)
    openx_pos_rewind_mask = torch.cat([positive_rewind_mask[:openx_len], negative_rewind_mask_1[:openx_len]], dim = 0)
        
    extra_pos_video_array = torch.cat([positive_video_array[openx_len:], negative_video_array_1[openx_len:]], dim = 0)
    extra_pos_text_array = torch.cat([positive_text_array[openx_len:], negative_text_array_1[openx_len:]], dim = 0)
    extra_pos_progress = torch.cat([positive_progress[openx_len:], negative_progress_1[openx_len:]], dim = 0)
    extra_pos_goal_distance = torch.cat([positive_goal_distance[openx_len:], negative_goal_distance_1[openx_len:]], dim = 0)
    extra_pos_rewind_mask = torch.cat([positive_rewind_mask[openx_len:], negative_rewind_mask_1[openx_len:]], dim = 0)

    video_array = torch.cat([openx_pos_video_array, extra_pos_video_array], dim = 0)
    text_array = torch.cat([openx_pos_text_array, extra_pos_text_array], dim = 0)
    progress = torch.cat([openx_pos_progress, extra_pos_progress], dim = 0).float()
    goal_distance = torch.cat([openx_pos_goal_distance, extra_pos_goal_distance], dim = 0).float()
    rewind_mask = torch.cat([openx_pos_rewind_mask, extra_pos_rewind_mask], dim = 0).float()

    openx_len = len(openx_pos_video_array)
    extra_len = len(extra_pos_video_array)

    video_embedding = video_array

    # Binary classification targets
    compressed_extra_class_label = extra_data["class_label"][:, 0].float()
    openx_target = torch.cat([torch.ones(openx_len // 2), torch.zeros(openx_len // 2)], dim=0).to(device)
    extra_target = torch.cat([compressed_extra_class_label, torch.zeros(extra_len // 2)], dim=0).to(device)
    positive_sequence_mask = torch.cat([openx_target, extra_target], dim=0).bool()

    # Get predictions from classifier
    progress_pred = rewind_model(video_embedding, text_array)

    openx_progress_pred = progress_pred[:openx_len]
    extra_progress_pred = progress_pred[openx_len:]

    openx_progress_target = progress[:openx_len]
    extra_progress_target = progress[openx_len:]

    valid_openx_progress_pred = openx_progress_pred[openx_target.bool()]
    valid_openx_progress_target = openx_progress_target[openx_target.bool()]

    valid_extra_progress_pred = extra_progress_pred[extra_target.bool()]
    valid_extra_progress_target = extra_progress_target[extra_target.bool()]

    rest_openx_progress_pred = openx_progress_pred[~openx_target.bool()]
    rest_openx_progress_target = openx_progress_target[~openx_target.bool()]

    rest_extra_progress_pred = extra_progress_pred[~extra_target.bool()]
    rest_extra_progress_target = extra_progress_target[~extra_target.bool()]

    openx_progress_loss = mse_loss(valid_openx_progress_pred[:,1:].squeeze(-1), valid_openx_progress_target[:,1:])
    extra_progress_loss = mse_loss(valid_extra_progress_pred[:,1:].squeeze(-1), valid_extra_progress_target[:,1:])
    rest_openx_progress_loss = mse_loss(rest_openx_progress_pred[:,1:].squeeze(-1), rest_openx_progress_target[:,1:])
    rest_extra_progress_loss = mse_loss(rest_extra_progress_pred[:,1:].squeeze(-1), rest_extra_progress_target[:,1:])

    total_len = len(openx_progress_pred) + len(extra_progress_pred) + len(rest_openx_progress_pred) + len(rest_extra_progress_pred)

    progress_loss = openx_progress_loss * len(openx_progress_pred) / total_len \
                    + extra_progress_loss * len(extra_progress_pred) / total_len \
                    + rest_openx_progress_loss * len(rest_openx_progress_pred) / total_len \
                    + rest_extra_progress_loss * len(rest_extra_progress_pred) / total_len

    directional_loss = progress_loss.new_tensor(0.0)
    directional_violation_rate = progress_loss.new_tensor(0.0)
    away_step_rate = progress_loss.new_tensor(0.0)
    if args.lambda_dir > 0:
        directional_loss, directional_violation_rate, away_step_rate = compute_directional_penalty(
            reward_predictions=progress_pred[positive_sequence_mask],
            goal_distances=goal_distance[positive_sequence_mask],
            tau_away=args.tau_away,
            margin=args.margin,
        )

    (
        forward_delta_mean,
        forward_decrease_rate,
        rewind_delta_mean,
        rewind_non_decreasing_rate,
        rewind_step_rate,
    ) = _summarize_progress_deltas(
        progress_targets=progress[positive_sequence_mask],
        rewind_mask=rewind_mask[positive_sequence_mask],
    )

    loss = (args.lambda_prog * progress_loss) + (args.lambda_dir * directional_loss)

    loss.backward()
    if args.clip_grad:
        torch.nn.utils.clip_grad_norm_(rewind_model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    # Log all metrics
    wandb_log = {
        "train/loss": loss.item(),
        "train/progress_loss": progress_loss.item(),
        "train/directional_loss": directional_loss.item(),
        "train/directional_violation_rate": directional_violation_rate.item(),
        "train/away_step_rate": away_step_rate.item(),
        "train/progress_target_mean": progress[positive_sequence_mask].mean().item(),
        "train/progress_target_start": progress[positive_sequence_mask][:, 0].mean().item(),
        "train/progress_target_end": progress[positive_sequence_mask][:, -1].mean().item(),
        "train/goal_distance_mean": goal_distance[positive_sequence_mask].mean().item(),
        "train/forward_progress_delta_mean": forward_delta_mean.item(),
        "train/forward_progress_decrease_rate": forward_decrease_rate.item(),
        "train/rewind_progress_delta_mean": rewind_delta_mean.item(),
        "train/rewind_progress_non_decreasing_rate": rewind_non_decreasing_rate.item(),
        "train/rewind_step_rate": rewind_step_rate.item(),
        "lr": optimizer.param_groups[0]["lr"],
    }
    wandb.log(wandb_log)
    return loss.item()
