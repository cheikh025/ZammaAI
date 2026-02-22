#!/usr/bin/env python3
"""CLI entrypoint for Khreibga Zero training."""

from __future__ import annotations

import argparse

import torch

from khreibga.model import get_device
from khreibga.trainer import Trainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Khreibga Zero.")

    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--games-per-iteration", type=int, default=25)
    parser.add_argument("--num-simulations", type=int, default=200)

    parser.add_argument("--self-play-parallel", action="store_true")
    parser.add_argument("--num-self-play-workers", type=int, default=4)
    parser.add_argument("--self-play-worker-device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--eval-simulations", type=int, default=200)

    parser.add_argument("--enable-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-log-dir", type=str, default="runs/khreibga")
    parser.add_argument("--tensorboard-run-name", type=str, default="run_001")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--checkpoint-out", type=str, default="checkpoints/run_001_last.pt")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Save intermediate checkpoints every N iterations (0 disables).",
    )
    parser.add_argument(
        "--no-checkpoint-history",
        action="store_true",
        help="Disable iteration-stamped snapshot files for periodic checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-in",
        type=str,
        default=None,
        help=(
            "Resume training from this checkpoint. "
            "Model weights, optimizer state, and iteration counter are restored. "
            "--num-iterations is treated as the total target; remaining = target - saved_iteration."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output.",
    )

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return get_device()
    return torch.device(device_arg)


def main() -> int:
    args = parse_args()

    cfg = TrainingConfig(
        num_iterations=args.num_iterations,
        games_per_iteration=args.games_per_iteration,
        num_simulations=args.num_simulations,
        self_play_parallel=args.self_play_parallel,
        num_self_play_workers=args.num_self_play_workers,
        self_play_worker_device=args.self_play_worker_device,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        eval_simulations=args.eval_simulations,
        enable_tensorboard=args.enable_tensorboard,
        tensorboard_log_dir=args.tensorboard_log_dir,
        tensorboard_run_name=args.tensorboard_run_name,
    )

    device = resolve_device(args.device)

    if args.checkpoint_in:
        # Resume: restore model, optimizer, and iteration counter from checkpoint.
        trainer = Trainer.from_checkpoint(args.checkpoint_in, device=device)
        # Apply CLI overrides so the user can change settings on resume.
        trainer.config.num_simulations          = args.num_simulations
        trainer.config.games_per_iteration      = args.games_per_iteration
        trainer.config.self_play_parallel       = args.self_play_parallel
        trainer.config.num_self_play_workers    = args.num_self_play_workers
        trainer.config.eval_interval            = args.eval_interval
        trainer.config.eval_games               = args.eval_games
        trainer.config.eval_simulations         = args.eval_simulations
        trainer.config.enable_tensorboard       = args.enable_tensorboard
        trainer.config.tensorboard_log_dir      = args.tensorboard_log_dir
        trainer.config.tensorboard_run_name     = args.tensorboard_run_name
        # Recreate TensorBoard writer with the (possibly updated) settings.
        trainer.tb_writer = trainer._create_tensorboard_writer()
        # Compute how many iterations remain to reach the total target.
        remaining = max(0, args.num_iterations - trainer.iteration)
        num_iterations_to_run = remaining
        if not args.quiet:
            print(
                f"[resume] loaded checkpoint: {args.checkpoint_in} "
                f"(iteration={trainer.iteration}, step={trainer.training_step})",
                flush=True,
            )
            print(
                f"[resume] target={args.num_iterations} "
                f"done={trainer.iteration} remaining={remaining}",
                flush=True,
            )
    else:
        trainer = Trainer(config=cfg, device=device)
        num_iterations_to_run = args.num_iterations

    try:
        if not args.quiet:
            print(
                "[run] config: "
                f"device={device} "
                f"iterations={num_iterations_to_run} "
                f"games/iter={args.games_per_iteration} "
                f"sims={args.num_simulations} "
                f"parallel={args.self_play_parallel} "
                f"workers={args.num_self_play_workers} "
                f"eval_interval={args.eval_interval} "
                f"checkpoint_interval={args.checkpoint_interval}",
                flush=True,
            )
        trainer.run(
            num_iterations=num_iterations_to_run,
            checkpoint_every=args.checkpoint_interval,
            checkpoint_path=args.checkpoint_out,
            checkpoint_keep_history=not args.no_checkpoint_history,
            verbose=not args.quiet,
        )
        trainer.save_checkpoint(args.checkpoint_out)
        if not args.quiet:
            print(f"[ckpt] saved final: {args.checkpoint_out}", flush=True)
    finally:
        trainer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
