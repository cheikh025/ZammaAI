#!/usr/bin/env python3
"""
Evaluate two checkpoints against each other with per-color breakdown.
Runs the match twice: once with BLACK moving first (normal rules),
once with WHITE moving first, so you can see whether winning is tied
to the BLACK pieces or to whoever moves first.

Usage:
    python eval_checkpoints.py --ckpt-a checkpoints/run_001_last_iter_500.pt \
                                --ckpt-b checkpoints/run_001_last_iter_400.pt \
                                --games 50 --simulations 10
"""

from __future__ import annotations

import argparse
import math

import torch

from khreibga.board import BLACK, WHITE
from khreibga.env import KhreibagaEnv
from khreibga.game import compute_zobrist_hash
from khreibga.mcts import MCTS, select_action
from khreibga.model import KhreibagaNet, get_device
from khreibga.trainer import Trainer


def load_model(path: str, device: torch.device) -> KhreibagaNet:
    trainer = Trainer.from_checkpoint(path, device=device)
    model = trainer.model
    model.eval()
    return model


def elo_diff(score_rate: float) -> float:
    if score_rate <= 0.0:
        return float("-inf")
    if score_rate >= 1.0:
        return float("inf")
    return -400.0 * math.log10((1.0 / score_rate) - 1.0)


def make_env(white_moves_first: bool = False) -> KhreibagaEnv:
    """Create a fresh env, optionally giving WHITE the first move."""
    env = KhreibagaEnv()
    env.reset()
    if white_moves_first:
        env.game.current_player = WHITE
        env.game.first_mover = WHITE
        env.game.current_hash = compute_zobrist_hash(env.game.board, WHITE)
        env.game.history = {env.game.current_hash: 1}
    return env


def evaluate(
    model_a: KhreibagaNet,
    model_b: KhreibagaNet,
    num_games: int,
    num_simulations: int,
    c_puct: float,
    device: torch.device,
    white_moves_first: bool = False,
) -> dict:
    """
    Pit model_a vs model_b for num_games games.
    Even games: A=BLACK, B=WHITE.
    Odd  games: A=WHITE, B=BLACK.
    white_moves_first: if True, WHITE makes the very first move of every game.
    Returns per-color stats for model_a.
    """
    results = {
        "as_black": {"wins": 0, "losses": 0, "draws": 0},
        "as_white": {"wins": 0, "losses": 0, "draws": 0},
    }

    for game_idx in range(num_games):
        a_is_black = (game_idx % 2 == 0)
        color_key = "as_black" if a_is_black else "as_white"

        env = make_env(white_moves_first=white_moves_first)

        mcts_a = MCTS(model_a, c_puct=c_puct, num_simulations=num_simulations,
                      dirichlet_epsilon=0.0, device=device)
        mcts_b = MCTS(model_b, c_puct=c_puct, num_simulations=num_simulations,
                      dirichlet_epsilon=0.0, device=device)

        while not env.done:
            a_turn = (env.current_player == BLACK) == a_is_black
            mcts = mcts_a if a_turn else mcts_b
            visits = mcts.search(env)
            action = select_action(visits, temperature=0)
            env.step(action)

        if env.winner is None:
            results[color_key]["draws"] += 1
        elif (env.winner == BLACK) == a_is_black:
            results[color_key]["wins"] += 1
        else:
            results[color_key]["losses"] += 1

        games_done = game_idx + 1
        w = results["as_black"]["wins"] + results["as_white"]["wins"]
        l = results["as_black"]["losses"] + results["as_white"]["losses"]
        d = results["as_black"]["draws"] + results["as_white"]["draws"]
        print(f"\r  game {games_done:>4}/{num_games}  w/l/d={w}/{l}/{d}", end="", flush=True)

    print()
    return results


def summarise(r: dict, num_games: int, name_a: str, name_b: str, first_mover: str) -> None:
    """
    r["as_black"] = stats when A played BLACK (B played WHITE)
    r["as_white"] = stats when A played WHITE (B played BLACK)
    wins/losses are from A's perspective.
    """
    half = num_games // 2
    ab = r["as_black"]   # A=BLACK, B=WHITE
    aw = r["as_white"]   # A=WHITE, B=BLACK

    total_a_wins = ab["wins"] + aw["wins"]
    total_b_wins = ab["losses"] + aw["losses"]
    total_draws  = ab["draws"] + aw["draws"]
    total_sr_a   = (total_a_wins + 0.5 * total_draws) / num_games

    W = 60
    print(f"\n{'─'*W}")
    print(f"  First mover: {first_mover}")
    print(f"{'─'*W}")
    print(f"  {'Matchup':<34}  {'A wins':>6}  {'B wins':>6}  {'draws':>5}")
    print(f"  {'─'*34}  {'─'*6}  {'─'*6}  {'─'*5}")

    # Row 1: A as BLACK vs B as WHITE
    row1 = f"A={name_a}(BLACK) vs B={name_b}(WHITE)"
    print(f"  {row1:<34}  {ab['wins']:>6}  {ab['losses']:>6}  {ab['draws']:>5}   ({half} games)")

    # Row 2: A as WHITE vs B as BLACK
    row2 = f"A={name_a}(WHITE) vs B={name_b}(BLACK)"
    print(f"  {row2:<34}  {aw['wins']:>6}  {aw['losses']:>6}  {aw['draws']:>5}   ({half} games)")

    print(f"  {'─'*34}  {'─'*6}  {'─'*6}  {'─'*5}")
    print(f"  {'TOTAL':<34}  {total_a_wins:>6}  {total_b_wins:>6}  {total_draws:>5}   ({num_games} games)")
    print(f"  score_rate(A) = {total_sr_a:.3f}   elo_diff(A) = {elo_diff(total_sr_a):.1f}")
    print(f"{'─'*W}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-color checkpoint evaluation.")
    parser.add_argument("--ckpt-a", required=True)
    parser.add_argument("--ckpt-b", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.games % 2 != 0:
        print(f"[warn] --games={args.games} is odd; rounding down to {args.games - 1}")
        args.games -= 1

    device = get_device() if args.device == "auto" else torch.device(args.device)

    print(f"[eval] device={device}  games={args.games}  sims={args.simulations}")
    print(f"[eval] A = {args.ckpt_a}")
    print(f"[eval] B = {args.ckpt_b}")

    print("[eval] loading checkpoints...", flush=True)
    model_a = load_model(args.ckpt_a, device)
    model_b = load_model(args.ckpt_b, device)

    name_a = args.ckpt_a.split("/")[-1].replace("run_001_last_", "")
    name_b = args.ckpt_b.split("/")[-1].replace("run_001_last_", "")

    # --- Round 1: normal rules, BLACK moves first ---
    print("\n[eval] Round 1 — BLACK moves first (normal rules)...", flush=True)
    results_normal = evaluate(
        model_a, model_b,
        num_games=args.games,
        num_simulations=args.simulations,
        c_puct=args.c_puct,
        device=device,
        white_moves_first=False,
    )
    summarise(results_normal, args.games, name_a, name_b, first_mover="BLACK")

    # --- Round 2: modified rules, WHITE moves first ---
    print("\n[eval] Round 2 — WHITE moves first (modified rules)...", flush=True)
    results_white_first = evaluate(
        model_a, model_b,
        num_games=args.games,
        num_simulations=args.simulations,
        c_puct=args.c_puct,
        device=device,
        white_moves_first=True,
    )
    summarise(results_white_first, args.games, name_a, name_b, first_mover="WHITE")

    # --- Verdict ---
    b_wins_normal = results_normal["as_black"]["wins"] + results_normal["as_white"]["wins"]
    first_mover_wins = results_white_first["as_white"]["wins"] + results_white_first["as_black"]["losses"]

    print("\n[verdict]")
    print(f"  Normal   (BLACK first): BLACK-piece winner? {b_wins_normal}/{args.games} games won by BLACK pieces")
    white_as_first = results_white_first["as_white"]["wins"] + results_white_first["as_black"]["losses"]
    # first mover in white_first variant is WHITE
    # when A is BLACK: first mover is WHITE=B, B wins when A loses → results_white_first as_black losses
    # when A is WHITE: first mover is WHITE=A, A wins → results_white_first as_white wins
    first_mover_w = results_white_first["as_white"]["wins"] + results_white_first["as_black"]["losses"]
    print(f"  Modified (WHITE first): First-mover winner? {first_mover_w}/{args.games} games won by first mover (WHITE)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
