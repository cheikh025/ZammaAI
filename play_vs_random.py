#!/usr/bin/env python3
"""Play Khreibga against a random agent from the terminal."""

from __future__ import annotations

import argparse
import random
from typing import Dict, List

from khreibga.board import (
    BLACK,
    BLACK_KING,
    BLACK_MAN,
    EMPTY,
    NUM_SQUARES,
    WHITE,
    WHITE_KING,
    WHITE_MAN,
    display_board,
    sq_to_rc,
)
from khreibga.game import GameState, compute_zobrist_hash

PLAYER_NAME: Dict[int, str] = {
    BLACK: "BLACK",
    WHITE: "WHITE",
}

TOKEN_TO_PIECE: Dict[str, int] = {
    ".": EMPTY,
    "bm": BLACK_MAN,
    "bk": BLACK_KING,
    "wm": WHITE_MAN,
    "wk": WHITE_KING,
}

CHAR_TO_PIECE: Dict[str, int] = {
    ".": EMPTY,
    "b": BLACK_MAN,
    "B": BLACK_KING,
    "w": WHITE_MAN,
    "W": WHITE_KING,
}

PIECE_TO_CHAR: Dict[int, str] = {v: k for k, v in CHAR_TO_PIECE.items()}


def decode_action(action: int) -> tuple[int, int]:
    """Return (src, dst) from an action index."""
    return action // NUM_SQUARES, action % NUM_SQUARES


def legal_actions(gs: GameState) -> List[int]:
    """Return legal action indices from the current mask."""
    mask = gs.get_action_mask()
    return [i for i, v in enumerate(mask) if v == 1]


def fmt_action(action: int) -> str:
    """Human-readable action string."""
    src, dst = decode_action(action)
    sr, sc = sq_to_rc(src)
    dr, dc = sq_to_rc(dst)
    return f"{action:3d}: {src:2d}({sr},{sc})->{dst:2d}({dr},{dc})"


def board_to_string(board: List[int]) -> str:
    """Compact 25-char board string for scenario reuse."""
    return "".join(PIECE_TO_CHAR[p] for p in board)


def rehash_state(gs: GameState) -> None:
    """Rebuild hash/history after manual edits, then re-evaluate terminal."""
    gs.chain_piece = None
    gs.done = False
    gs.winner = None
    gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
    gs.history = {gs.current_hash: 1}
    gs._check_terminal()


def print_setup_help() -> None:
    print("Setup commands:")
    print("  show                      - display board")
    print("  clear                     - clear board")
    print("  set <sq> <piece>          - piece in: ., bm, bk, wm, wk")
    print("  load <25chars>            - chars: . b B w W")
    print("  dump                      - print compact 25-char board")
    print("  turn <black|white>        - set side to move")
    print("  clock <n>                 - set half-move clock")
    print("  ply <n>                   - set total half-move count")
    print("  done                      - finish setup and play")
    print("  quit                      - exit")


def setup_position(gs: GameState) -> None:
    """Interactive board setup before/while playing."""
    print_setup_help()
    while True:
        raw = input("setup> ").strip()
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in {"help", "?"}:
            print_setup_help()
            continue
        if cmd in {"quit", "exit"}:
            raise SystemExit(0)
        if cmd == "show":
            display_board(gs.board)
            continue
        if cmd == "clear":
            gs.board = [EMPTY] * NUM_SQUARES
            print("Board cleared.")
            continue
        if cmd == "dump":
            print(board_to_string(gs.board))
            continue
        if cmd == "set":
            if len(parts) != 3:
                print("Usage: set <sq> <piece>")
                continue
            try:
                sq = int(parts[1])
            except ValueError:
                print("Square must be an integer in [0, 24].")
                continue
            piece_token = parts[2].lower()
            if sq < 0 or sq >= NUM_SQUARES:
                print("Square must be in [0, 24].")
                continue
            if piece_token not in TOKEN_TO_PIECE:
                print("Piece must be one of: ., bm, bk, wm, wk")
                continue
            gs.board[sq] = TOKEN_TO_PIECE[piece_token]
            print(f"Set sq {sq} to {piece_token}.")
            continue
        if cmd == "load":
            if len(parts) != 2:
                print("Usage: load <25chars>")
                continue
            s = parts[1]
            if len(s) != NUM_SQUARES:
                print("Scenario must be exactly 25 characters.")
                continue
            bad = [ch for ch in s if ch not in CHAR_TO_PIECE]
            if bad:
                print("Invalid chars. Allowed: . b B w W")
                continue
            gs.board = [CHAR_TO_PIECE[ch] for ch in s]
            print("Scenario loaded.")
            continue
        if cmd == "turn":
            if len(parts) != 2:
                print("Usage: turn <black|white>")
                continue
            side = parts[1].lower()
            if side == "black":
                gs.current_player = BLACK
            elif side == "white":
                gs.current_player = WHITE
            else:
                print("Turn must be black or white.")
                continue
            print(f"Turn set to {PLAYER_NAME[gs.current_player]}.")
            continue
        if cmd == "clock":
            if len(parts) != 2:
                print("Usage: clock <n>")
                continue
            try:
                n = int(parts[1])
            except ValueError:
                print("clock must be an integer >= 0.")
                continue
            if n < 0:
                print("clock must be >= 0.")
                continue
            gs.half_move_clock = n
            print(f"half_move_clock = {n}")
            continue
        if cmd == "ply":
            if len(parts) != 2:
                print("Usage: ply <n>")
                continue
            try:
                n = int(parts[1])
            except ValueError:
                print("ply must be an integer >= 0.")
                continue
            if n < 0:
                print("ply must be >= 0.")
                continue
            gs.move_count = n
            print(f"move_count = {n}")
            continue
        if cmd in {"done", "start"}:
            rehash_state(gs)
            print("Setup finished.")
            display_board(gs.board)
            return

        print("Unknown setup command. Type: help")


def print_play_help() -> None:
    print("Play commands:")
    print("  moves                     - list legal actions")
    print("  <action>                  - play action index (0..624)")
    print("  <src> <dst>               - play by source/target squares")
    print("  rand                      - pick a random legal move for you")
    print("  board                     - display board")
    print("  setup                     - enter scenario setup mode")
    print("  quit                      - exit")


def choose_human_move(gs: GameState, rng: random.Random) -> int:
    """Read and validate one human move from stdin."""
    legal = legal_actions(gs)
    if not legal:
        raise RuntimeError("No legal moves available for human player.")

    while True:
        raw = input("you> ").strip()
        if not raw:
            continue
        low = raw.lower()

        if low in {"help", "?"}:
            print_play_help()
            continue
        if low in {"quit", "exit"}:
            raise SystemExit(0)
        if low in {"board", "show"}:
            display_board(gs.board)
            continue
        if low in {"moves", "m"}:
            for action in legal:
                print(f"  {fmt_action(action)}")
            continue
        if low in {"setup"}:
            setup_position(gs)
            legal = legal_actions(gs)
            if not legal:
                print("No legal moves after setup.")
            continue
        if low in {"rand", "r"}:
            action = rng.choice(legal)
            print(f"You (random) play {fmt_action(action)}")
            return action

        parts = raw.split()
        if len(parts) == 1:
            try:
                action = int(parts[0])
            except ValueError:
                print("Invalid input. Type: help")
                continue
            if action in legal:
                return action
            print("Illegal action for current position.")
            continue
        if len(parts) == 2:
            try:
                src = int(parts[0])
                dst = int(parts[1])
            except ValueError:
                print("Use integers: <src> <dst>")
                continue
            if src < 0 or src >= NUM_SQUARES or dst < 0 or dst >= NUM_SQUARES:
                print("Squares must be in [0, 24].")
                continue
            action = src * NUM_SQUARES + dst
            if action in legal:
                return action
            print("Illegal action for current position.")
            continue

        print("Invalid input. Type: help")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Khreibga vs random agent.")
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
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default: 0).",
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


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    gs = GameState()
    gs.current_player = WHITE if args.first == "white" else BLACK
    rehash_state(gs)
    if args.human == "black":
        human_player = BLACK
    elif args.human == "white":
        human_player = WHITE
    else:
        human_player = rng.choice([BLACK, WHITE])

    print(f"Human: {PLAYER_NAME[human_player]} | Random seed: {args.seed}")
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
            f"clock={gs.half_move_clock} | ply={gs.move_count}"
        )
        if gs.chain_piece is not None:
            print(f"Capture chain active from square {gs.chain_piece}.")

        if not legal:
            gs._check_terminal()
            break

        if gs.current_player == human_player:
            action = choose_human_move(gs, rng)
        else:
            action = rng.choice(legal)
            print(f"Random plays {fmt_action(action)}")

        gs.step(action)
        ply += 1

    print()
    display_board(gs.board)
    if gs.done:
        if gs.winner is None:
            print("Result: DRAW")
        elif gs.winner == human_player:
            print("Result: YOU WIN")
        else:
            print("Result: RANDOM WINS")
    else:
        print(f"Stopped after {args.max_ply} half-moves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
