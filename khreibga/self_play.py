"""
Self-play game generation and model evaluation for Khreibaga Zero.

Provides:
  - ``self_play_game``  : generate (obs, policy, value) training examples
                          from a single game of self-play.
  - ``evaluate_models`` : pit two models against each other to decide
                          whether the new model should replace the old one.
"""

from __future__ import annotations

import numpy as np
import torch

from khreibga.board import BLACK, WHITE
from khreibga.env import KhreibagaEnv
from khreibga.mcts import (
    MCTS,
    _get_obs,
    get_action_probs,
    select_action,
    ACTION_SPACE,
)
from khreibga.model import KhreibagaNet


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

def self_play_game(
    model: KhreibagaNet,
    num_simulations: int = 200,
    c_puct: float = 1.0,
    temp_threshold: int = 10,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    device: torch.device | None = None,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play one self-play game and return training examples.

    The current best model plays both sides.  MCTS is used to produce a
    policy target at each decision point.  After the game ends, the
    outcome is assigned retroactively to each example.

    Parameters
    ----------
    model : KhreibagaNet
        The neural network (should be in eval mode on *device*).
    num_simulations : int
        MCTS simulations per move.
    c_puct : float
        MCTS exploration constant.
    temp_threshold : int
        Number of decision points after which temperature drops to 0.
    dirichlet_alpha, dirichlet_epsilon : float
        Dirichlet noise parameters for root exploration.
    device : torch.device or None
        Device for NN inference.
    rng : np.random.Generator or None
        RNG for reproducibility.

    Returns
    -------
    list of (obs, policy, value)
        - obs    : ``(7, 5, 5)`` float32 numpy array (canonical observation).
        - policy : ``(625,)`` float32 array (MCTS visit distribution, τ=1).
        - value  : float in ``{-1, 0, +1}`` from the acting player's
                    perspective.
    """
    if rng is None:
        rng = np.random.default_rng()

    env = KhreibagaEnv()
    env.reset()

    mcts = MCTS(
        model,
        c_puct=c_puct,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        device=device,
        rng=rng,
    )

    history: list[tuple[np.ndarray, np.ndarray, int]] = []  # (obs, policy, player)
    decision_count = 0

    while not env.done:
        obs = _get_obs(env)
        player = env.current_player

        visits = mcts.search(env)

        # Store the visit distribution normalised with τ=1 as the training
        # policy target (regardless of the temperature used to pick the action).
        policy = get_action_probs(visits, temperature=1.0)

        # Temperature schedule: explore early, exploit later.
        temp = 1.0 if decision_count < temp_threshold else 0.0
        action = select_action(visits, temperature=temp, rng=rng)

        history.append((obs, policy, player))
        env.step(action)
        decision_count += 1

    # Assign terminal values retroactively.
    winner = env.winner
    examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    for obs, policy, player in history:
        if winner is None:
            z = 0.0
        elif winner == player:
            z = 1.0
        else:
            z = -1.0
        examples.append((obs, policy, z))

    return examples


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate_models(
    new_model: KhreibagaNet,
    old_model: KhreibagaNet,
    num_games: int = 50,
    num_simulations: int = 200,
    c_puct: float = 1.0,
    device: torch.device | None = None,
) -> float:
    """Pit *new_model* against *old_model* and return new model's win rate.

    Colours are alternated: even-indexed games have new_model as BLACK,
    odd-indexed games have new_model as WHITE.  No Dirichlet noise is
    applied, and actions are chosen deterministically (τ = 0).

    Parameters
    ----------
    new_model, old_model : KhreibagaNet
        Both should be in eval mode on *device*.
    num_games : int
        Total evaluation games (should be even for fair colour split).
    num_simulations : int
        MCTS simulations per move during evaluation.
    c_puct : float
        MCTS exploration constant.
    device : torch.device or None

    Returns
    -------
    float
        Win rate of *new_model* in ``[0, 1]``.
    """
    new_wins = 0

    for game_idx in range(num_games):
        new_is_black = (game_idx % 2 == 0)

        env = KhreibagaEnv()
        env.reset()

        mcts_new = MCTS(
            new_model, c_puct=c_puct, num_simulations=num_simulations,
            dirichlet_epsilon=0.0, device=device,
        )
        mcts_old = MCTS(
            old_model, c_puct=c_puct, num_simulations=num_simulations,
            dirichlet_epsilon=0.0, device=device,
        )

        while not env.done:
            is_new_turn = (env.current_player == BLACK) == new_is_black
            mcts = mcts_new if is_new_turn else mcts_old

            visits = mcts.search(env)
            action = select_action(visits, temperature=0)
            env.step(action)

        # Score the result for new_model.
        if env.winner is not None:
            if (env.winner == BLACK) == new_is_black:
                new_wins += 1

    return new_wins / num_games if num_games > 0 else 0.0
