#!/usr/bin/env python3
"""Play against a checkpointed Khreibaga model from the terminal."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch

from khreibga.board import BLACK, WHITE, display_board
from khreibga.env import KhreibagaEnv
from khreibga.mcts import MCTS, select_action
from khreibga.model import KhreibagaNet, get_device
from play_vs_random import (
    PLAYER_NAME,
    choose_human_move,
    fmt_action,
    legal_actions,
    print_play_help,
    rehash_state,
    setup_position,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play Khreibaga against a checkpoint model.",
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
        help="Explicit checkpoint path. If omitted, latest in --checkpoint-dir is used.",
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
        help="Use model_state_dict instead of best_model_state_dict.",
    )
    parser.add_argument(
        "--human",
        choices=["black", "white", "random"],
        default="white",
        help="Human side (default: white).",
    )
    parser.add_argument(
        "--first",
        choices=["black", "white"],
        default="white",
        help="Side to move first (default: white).",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=50,
        help="MCTS simulations per model move (default: 50).",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.0,
        help="MCTS exploration constant (default: 1.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for random choices (default: 0).",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Open setup mode before starting play.",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        default=1000,
        help="Safety stop after N half-moves (default: 1000).",
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


def main() -> int:
    args = parse_args()

    if args.num_simulations <= 0:
        raise ValueError("--num-simulations must be > 0")
    if args.max_ply <= 0:
        raise ValueError("--max-ply must be > 0")

    device = resolve_device(args.device)
    rng = random.Random(args.seed)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
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

    env = KhreibagaEnv()
    env.reset()
    gs = env.game
    gs.current_player = WHITE if args.first == "white" else BLACK
    rehash_state(gs)

    if args.human == "black":
        human_player = BLACK
    elif args.human == "white":
        human_player = WHITE
    else:
        human_player = rng.choice([BLACK, WHITE])

    model_player = WHITE if human_player == BLACK else BLACK

    mcts = MCTS(
        model,
        c_puct=args.c_puct,
        num_simulations=args.num_simulations,
        dirichlet_epsilon=0.0,
        device=device,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device} | Simulations/move: {args.num_simulations}")
    print(
        f"Human: {PLAYER_NAME[human_player]} | "
        f"Model: {PLAYER_NAME[model_player]} | "
        f"First: {PLAYER_NAME[gs.current_player]}",
    )
    print_play_help()

    if args.setup:
        setup_position(gs)

    ply = 0
    while not gs.done and ply < args.max_ply:
        print()
        display_board(gs.board)
        legal = legal_actions(gs)
        print(
            f"To move: {PLAYER_NAME[gs.current_player]} | "
            f"legal actions: {len(legal)} | "
            f"clock={gs.half_move_clock} | ply={gs.move_count}",
        )
        if gs.chain_piece is not None:
            print(f"Capture chain active from square {gs.chain_piece}.")

        if not legal:
            gs._check_terminal()
            break

        if gs.current_player == human_player:
            action = choose_human_move(gs, rng)
        else:
            visits = mcts.search(env)
            action = select_action(visits, temperature=0.0)
            print(f"Model plays {fmt_action(action)}")

        env.step(int(action))
        ply += 1

    print()
    display_board(gs.board)
    if gs.done:
        if gs.winner is None:
            print("Result: DRAW")
        elif gs.winner == human_player:
            print("Result: YOU WIN")
        else:
            print("Result: MODEL WINS")
    else:
        print(f"Stopped after {args.max_ply} half-moves.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
