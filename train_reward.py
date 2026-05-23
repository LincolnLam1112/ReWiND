import os
import json
import h5py
import wandb
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.eval_utils import compute_metrics_multi
from model import ReWiNDTransformer
from dataset import ReWiNDVideoDataset

from utils.update_utils import train_step_fn, CosineWithMinLRScheduler
from utils.eval_confusion_matrix import plot_confusion_matrix
from utils.vla_eval import compute_vla_eval

os.environ["TOKENIZERS_PARALLELISM"] = "False"



def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_dim = 768
    text_dim = 384

    # TODO: Set your own WandB entity and project name
    WANDB_ENTITY_NAME = args.wandb_entity
    WANDB_PROJECT_NAME = args.wandb_project

    experiment_name = "ReWiND_Release_" + str(args.extra_data_type) + "_" + args.progress_target_type

    group_name = "ReWind_Release_" + args.extra_data_type + "_" + args.progress_target_type
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group=group_name,
        config=args,
        name=experiment_name,
    )

    # vla_replica overrides skip_metaworld_eval (it has its own eval path).
    use_metaworld_eval = (args.extra_data_type == "metaworld") and (not args.skip_metaworld_eval)

    if use_metaworld_eval:
        h5_train_eval_file = os.path.join(args.h5_folder_path, "metaworld_embeddings_train.h5")
        h5_eval_file = os.path.join(args.h5_folder_path, "metaworld_embeddings_eval.h5")
        h5_train_eval_file = h5py.File(h5_train_eval_file, "r")
        h5_eval_file = h5py.File(h5_eval_file, "r")
        h5_close_success_file = "datasets/metaworld_dino_embeddings_eval_close_succ.h5"
        h5_all_fail_file = "datasets/metaworld_dino_embeddings_eval_all_fail.h5"
        h5_close_success_file = h5py.File(h5_close_success_file, "r")
        h5_all_fail_file = h5py.File(h5_all_fail_file, "r")
        task_list = "utils/new_task_v2.json"
        task_list = json.load(open(task_list, "r"))
    else:
        h5_train_eval_file = None
        h5_eval_file = None
        h5_close_success_file = None
        h5_all_fail_file = None
        task_list = None

    openx_h5_file = h5py.File(args.openx_embedding_path, "r")
    openx_dataset = ReWiNDVideoDataset(args, openx_h5_file, sample_neg=False)

    # Resolve extra (target-domain) train source.
    if args.extra_data_type == "vla_replica":
        if not args.extra_train_h5_path:
            raise ValueError(
                "--extra_train_h5_path is required when --extra_data_type=vla_replica"
            )
        extra_train_h5 = h5py.File(args.extra_train_h5_path, "r")
        extra_dataset = ReWiNDVideoDataset(args, extra_train_h5, sample_neg=True)
    elif use_metaworld_eval:
        extra_dataset = ReWiNDVideoDataset(args, h5_train_eval_file, sample_neg=True)
    else:
        extra_dataset = None

    if extra_dataset is None:
        openx_batch_size = args.batch_size
        extra_batch_size = 0
    else:
        openx_batch_size = int(round(args.batch_size * (1 - args.extra_data_ratio)))
        extra_batch_size = int(round(args.batch_size * args.extra_data_ratio))

    openx_dataloader = DataLoader(openx_dataset, batch_size=openx_batch_size, shuffle=True, num_workers=int(args.worker * 4), drop_last=True, pin_memory=False)
    extra_dataloader = None if extra_dataset is None else DataLoader(extra_dataset, batch_size=extra_batch_size, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=False)

    rewind_model = ReWiNDTransformer(
        args=args,
        video_dim=video_dim,  # Original video embedding dimension
        text_dim=text_dim,   # Original text embedding dimension
        hidden_dim=512  # Common dimension for transformer processing
    ).to(device)


    print(rewind_model)
    base_optimizer = torch.optim.Adam(rewind_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineWithMinLRScheduler(base_optimizer, max_steps=300000, max_lr=args.lr, min_lr=1e-5)

    print("Starting training")

    for epoch in range(args.epochs):

        rewind_model.train()

        if extra_dataloader is None:
            training_loader = ((b, None) for b in openx_dataloader)
            total_batches = len(openx_dataloader)
        else:
            training_loader = zip(openx_dataloader, extra_dataloader)
            total_batches = min(len(openx_dataloader), len(extra_dataloader))

        for batch in tqdm(training_loader, total=total_batches, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            train_step_fn(
                args=args,
                batch=batch,
                rewind_model=rewind_model,
                optimizer=base_optimizer,
                scheduler=scheduler
            )

        rewind_model.eval()
        # with torch.no_grad():
            # if args.extra_data_type == "metaworld":

                # plot_confusion_matrix(h5_file = h5_train_eval_file, set = "train", rewind_model = rewind_model, args = args, epoch = epoch, run_name = experiment_name)
                # plot_confusion_matrix(h5_file = h5_eval_file, set = "eval", rewind_model = rewind_model, args = args, epoch = epoch, run_name = experiment_name)
        
        if args.vla_eval_seen_h5_path:
            compute_vla_eval(
                args=args,
                rewind_model=rewind_model,
                eval_h5_path=args.vla_eval_seen_h5_path,
                epoch=epoch,
                log_prefix="eval_seen",
            )
        if args.vla_eval_unseen_h5_path:
            compute_vla_eval(
                args=args,
                rewind_model=rewind_model,
                eval_h5_path=args.vla_eval_unseen_h5_path,
                epoch=epoch,
                log_prefix="eval_unseen",
                negative_pool_h5_path=args.vla_eval_seen_h5_path or None,
            )

        if use_metaworld_eval:
            if epoch <= 15:
                if (epoch + 1) % args.eval_interval == 0: # too save time, we evaluate every 5 epochs
                    compute_metrics_multi(args,
                                        rewind_model,
                                        gt_data = h5_eval_file,
                                        close_success_data=h5_close_success_file,
                                        all_fail_data=h5_all_fail_file,
                                        task_list=task_list,
                                        epoch=epoch)
            else:
                compute_metrics_multi(args,
                                    rewind_model,
                                    gt_data = h5_eval_file,
                                    close_success_data=h5_close_success_file,
                                    all_fail_data=h5_all_fail_file,
                                    task_list=task_list,
                                    epoch=epoch)
        
        # save checkpoint
        if args.progress_target_type == "dino_goal_distance":
            checkpoint_dir = "checkpoints_dino_freeze" if args.use_freeze else "checkpoints_dino"
        elif args.progress_target_type == "optical_flow":
            checkpoint_dir = "checkpoints_flow_freeze" if args.use_freeze else "checkpoints_flow"
        else:
            checkpoint_dir = "checkpoints_freeze" if args.use_freeze else "checkpoints"
        if os.path.exists(checkpoint_dir) is False:
            os.mkdir(checkpoint_dir)
        save_dict = {
            "args": args,
            "model_state_dict": rewind_model.state_dict(),
            "optimizer_state_dict": base_optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        }
        torch.save(save_dict, f"{checkpoint_dir}/rewind_{args.extra_data_type}_epoch_{epoch}.pth")
        


        rewind_model.train()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--wandb_entity', type=str, required=True, help="Your WandB entity name")
    argparser.add_argument('--wandb_project', type=str, default='rewind-reward-training', help="WandB project name")
    argparser.add_argument('--h5_folder_path', type=str, default='datasets')
    argparser.add_argument('--openx_embedding_path', type=str, default='datasets/full_openx_embeddings_v2_train.h5', help="Path to the OpenX embeddings file")
    argparser.add_argument('--extra_data_type', type=str, choices=["metaworld", "vla_replica"], default="metaworld")
    argparser.add_argument('--extra_train_h5_path', type=str, default="",
                           help="Path to extra (target-domain) train h5 mixed into each batch. "
                                "Required when --extra_data_type=vla_replica.")
    argparser.add_argument('--batch_size', type=int, default=1024)
    argparser.add_argument('--epochs', type=int, default=20)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--worker', type=int, default=1)
    argparser.add_argument('--rewind', action='store_true')
    argparser.add_argument('--subsample_video', action='store_true')
    argparser.add_argument('--max_length', type=int, default=16)
    argparser.add_argument('--cosine_scheduler', action='store_true')
    argparser.add_argument('--clip_grad', action='store_true')
    argparser.add_argument('--extra_data_ratio', type=float, default=0.2)
    argparser.add_argument('--eval_interval', type=int, default=1)
    argparser.add_argument('--rewind_ratio', type=float, default=0.8)
    argparser.add_argument('--pdf', action='store_true', help="Whether to save confusion matrix as PDF")
    argparser.add_argument('--use_freeze', action='store_true')
    argparser.add_argument('--freeze_ratio', type=float, default=0.4)
    argparser.add_argument('--eval_max_samples', type=int, default=-1)
    argparser.add_argument('--progress_target_type', type=str, choices=["linear", "dino_goal_distance", "optical_flow"], default="dino_goal_distance")
    argparser.add_argument('--goal_k', type=int, default=3)
    argparser.add_argument('--lambda_prog', type=float, default=1.0)
    argparser.add_argument('--lambda_dir', type=float, default=0.25)
    argparser.add_argument('--tau_away', type=float, default=0.01)
    argparser.add_argument('--margin', type=float, default=0.0)
    argparser.add_argument('--flow_missing_fallback', type=str, choices=["linear", "error"], default="linear")
    argparser.add_argument('--skip_metaworld_eval', action='store_true',
                           help="Skip loading metaworld_embeddings_{train,eval}.h5 and the "
                                "compute_metrics_multi eval pass; use openx pool as sole training data.")
    argparser.add_argument('--vla_eval_seen_h5_path', type=str, default="",
                           help="Path to eval_seen h5 (held-out trajectories from training "
                                "instructions). Empty disables this eval.")
    argparser.add_argument('--vla_eval_unseen_h5_path', type=str, default="",
                           help="Path to eval_unseen h5 (held-out instructions). "
                                "Empty disables this eval.")
    args = argparser.parse_args()
    main(args)
