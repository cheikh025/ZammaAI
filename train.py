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
    trainer = Trainer(config=cfg, device=device)
    try:
        trainer.run(num_iterations=args.num_iterations)
        trainer.save_checkpoint(args.checkpoint_out)
    finally:
        trainer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

