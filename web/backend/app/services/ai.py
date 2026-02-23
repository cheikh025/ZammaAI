"""AI move selection: random policy and MCTS + neural network policy."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np

from khreibga.game import GameState
from khreibga.env import KhreibagaEnv

# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------

_model = None
_device = None


def _load_model():
    """Load KhreibagaNet once and cache it. Falls back to untrained weights."""
    global _model, _device
    if _model is not None:
        return _model, _device

    import torch
    from khreibga.model import KhreibagaNet, get_device

    _device = get_device()
    _model = KhreibagaNet().to(_device)

    # Try to find a checkpoint relative to the repo root
    repo_root = Path(__file__).resolve().parents[4]  # ZammaAI/
    checkpoint_dir = repo_root / "checkpoints"

    if checkpoint_dir.exists():
        candidates = sorted(
            checkpoint_dir.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            latest = candidates[-1]
            try:
                ckpt = torch.load(latest, map_location=_device, weights_only=False)
                key = (
                    "best_model_state_dict"
                    if "best_model_state_dict" in ckpt
                    else "model_state_dict"
                )
                _model.load_state_dict(ckpt[key])
                print(f"[AI] Loaded checkpoint: {latest.name}")
            except Exception as exc:
                print(f"[AI] Could not load checkpoint ({exc}). Using untrained model.")
        else:
            print("[AI] No *.pt files in checkpoints/. Using untrained model.")
    else:
        print("[AI] No checkpoints/ directory found. Using untrained model.")

    _model.eval()
    return _model, _device


# ---------------------------------------------------------------------------
# Policy functions
# ---------------------------------------------------------------------------

def random_action(game: GameState) -> int:
    """Pick a uniformly random legal action."""
    mask = game.get_action_mask()
    legal = [i for i, m in enumerate(mask) if m]
    if not legal:
        raise ValueError("No legal actions available.")
    return random.choice(legal)


def mcts_action(game: GameState, simulations: int = 200) -> int:
    """Pick the best action using MCTS guided by the neural network."""
    from khreibga.mcts import MCTS, select_action

    model, device = _load_model()

    # Wrap a clone of the game in a KhreibagaEnv for MCTS
    env = KhreibagaEnv()
    env.game = game.clone()

    mcts = MCTS(
        model,
        num_simulations=simulations,
        dirichlet_epsilon=0.0,   # no noise during play
        device=device,
    )
    visits = mcts.search(env)
    return select_action(visits, temperature=0.0)


def evaluate_position(game: GameState) -> float:
    """Single NN forward pass, returns value in [-1,+1] from current player's perspective."""
    from khreibga.model import predict

    model, device = _load_model()
    obs = np.array(game.get_observation(), dtype=np.float32)
    mask = np.array(game.get_action_mask(), dtype=np.float32)

    _, values = predict(model, [obs], [mask], device=device)
    return float(values[0])


def hint_actions(game: GameState, simulations: int = 50) -> list[dict]:
    """Lightweight MCTS, returns top 3 moves with visit shares."""
    from khreibga.mcts import MCTS

    model, device = _load_model()

    env = KhreibagaEnv()
    env.game = game.clone()

    mcts = MCTS(
        model,
        num_simulations=simulations,
        dirichlet_epsilon=0.25,
        device=device,
    )
    visits = mcts.search(env)  # np.ndarray shape (625,)

    total = float(visits.sum())
    if total == 0:
        return []

    # Get indices of top 3 by visit count
    top_indices = np.argsort(visits)[::-1][:3]
    hints = []
    for action in top_indices:
        action = int(action)
        count = float(visits[action])
        if count == 0:
            break
        hints.append({
            "action": action,
            "src": action // 25,
            "dst": action % 25,
            "visit_share": round(count / total, 4),
        })
    return hints


def get_ai_action(game: GameState, mode: str, simulations: int = 200) -> int:
    """Dispatch to the correct policy based on session mode."""
    if mode == "hvr":
        return random_action(game)
    if mode in ("hvai", "aivh"):
        return mcts_action(game, simulations)
    raise ValueError(f"Mode {mode!r} has no AI policy.")
