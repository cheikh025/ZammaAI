#!/usr/bin/env python3
"""Evaluate the latest training checkpoint against a random agent."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from khreibga.board import BLACK, WHITE
from khreibga.env import KhreibagaEnv
from khreibga.mcts import MCTS, select_action
from khreibga.model import KhreibagaNet, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a checkpoint (latest by default) and evaluate the model "
            "against a random agent."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoint files (*.pt).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional explicit checkpoint path. If omitted, latest in --checkpoint-dir is used.",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="*.pt",
        help="Glob pattern used to find checkpoints in --checkpoint-dir.",
    )
    parser.add_argument(
        "--use-current-model",
        action="store_true",
        help="Use model_state_dict instead of best_model_state_dict from checkpoint.",
    )
    parser.add_argument("--num-games", type=int, default=50)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random-agent move selection and color assignment reproducibility.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return get_device()
    return torch.device(device_arg)


def find_latest_checkpoint(checkpoint_dir: Path, pattern: str) -> Path:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    candidates = [p for p in checkpoint_dir.glob(pattern) if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint files found in {checkpoint_dir} matching pattern {pattern!r}",
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    *,
    use_current_model: bool,
) -> KhreibagaNet:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_key = "model_state_dict" if use_current_model else "best_model_state_dict"
    if state_key not in checkpoint:
        available = ", ".join(sorted(checkpoint.keys()))
        raise KeyError(
            f"{state_key!r} missing in checkpoint {checkpoint_path}. "
            f"Available keys: {available}",
        )

    model = KhreibagaNet().to(device)
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    return model


def play_one_game_vs_random(
    model: KhreibagaNet,
    *,
    model_is_black: bool,
    num_simulations: int,
    c_puct: float,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[int | None, int]:
    env = KhreibagaEnv()
    env.reset()

    mcts = MCTS(
        model,
        c_puct=c_puct,
        num_simulations=num_simulations,
        dirichlet_epsilon=0.0,
        device=device,
    )

    while not env.done:
        model_turn = (env.current_player == BLACK) if model_is_black else (env.current_player == WHITE)
        if model_turn:
            visits = mcts.search(env)
            action = select_action(visits, temperature=0.0)
        else:
            mask = env.get_action_mask()
            legal_actions = np.where(mask > 0)[0]
            if len(legal_actions) == 0:
                break
            action = int(rng.choice(legal_actions))
        env.step(int(action))

    return env.winner, env.game.move_count


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    if args.num_games <= 0:
        raise ValueError("--num-games must be > 0")
    if args.num_simulations <= 0:
        raise ValueError("--num-simulations must be > 0")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_latest_checkpoint(
            Path(args.checkpoint_dir),
            args.checkpoint_pattern,
        )

    model = load_model_from_checkpoint(
        checkpoint_path,
        device,
        use_current_model=args.use_current_model,
    )

    rng = np.random.default_rng(args.seed)
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0

    for game_idx in range(args.num_games):
        model_is_black = (game_idx % 2 == 0)
        winner, move_count = play_one_game_vs_random(
            model,
            model_is_black=model_is_black,
            num_simulations=args.num_simulations,
            c_puct=args.c_puct,
            device=device,
            rng=rng,
        )
        total_moves += int(move_count)

        if winner is None:
            draws += 1
        elif (winner == BLACK and model_is_black) or (winner == WHITE and not model_is_black):
            wins += 1
        else:
            losses += 1

    n = args.num_games
    win_rate = wins / n
    draw_rate = draws / n
    loss_rate = losses / n
    score_rate = (wins + 0.5 * draws) / n
    avg_moves = total_moves / n

    print(f"checkpoint: {checkpoint_path}")
    print(
        f"model_side: alternating (black on even games, white on odd games)",
    )
    print(f"games: {n}  sims_per_move: {args.num_simulations}  device: {device}")
    print(f"w/l/d: {wins}/{losses}/{draws}")
    print(
        f"win_rate={win_rate:.3f} draw_rate={draw_rate:.3f} "
        f"loss_rate={loss_rate:.3f} score_rate={score_rate:.3f}",
    )
    print(f"avg_half_moves={avg_moves:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
