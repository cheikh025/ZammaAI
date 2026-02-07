"""
Gymnasium-compatible RL environment wrapper for Khreibaga.

Wraps the GameState engine and provides NumPy-based observations
and action masks suitable for neural network training (AlphaZero-style).

Follows the Gymnasium API pattern:
  - reset()  -> (obs, info)
  - step(a)  -> (obs, reward, terminated, truncated, info)

Does NOT depend on the gymnasium package itself.
"""

from __future__ import annotations

import numpy as np

from khreibga.board import BLACK, WHITE, BOARD_SIZE, NUM_SQUARES, display_board
from khreibga.game import GameState, ACTION_SPACE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_to_numpy(obs_list: list) -> np.ndarray:
    """Convert the nested-list observation from GameState to a NumPy array.

    Parameters
    ----------
    obs_list : list
        A 7 x 5 x 5 nested Python list of floats returned by
        ``GameState.get_observation()``.

    Returns
    -------
    np.ndarray
        Shape ``(7, 5, 5)``, dtype ``float32``.
    """
    return np.array(obs_list, dtype=np.float32)


def _mask_to_numpy(mask_list: list) -> np.ndarray:
    """Convert the 625-element int list from GameState to a NumPy array.

    Parameters
    ----------
    mask_list : list
        A list of 625 integers (0 or 1) returned by
        ``GameState.get_action_mask()``.

    Returns
    -------
    np.ndarray
        Shape ``(625,)``, dtype ``float32``.
    """
    return np.array(mask_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# KhreibagaEnv
# ---------------------------------------------------------------------------

class KhreibagaEnv:
    """Gymnasium-style environment for Khreibga (Mauritanian Alquerque).

    Wraps the ``GameState`` engine and provides NumPy-based observations
    and action masks for neural network training.

    Observation
    -----------
    A ``(7, 5, 5)`` float32 tensor in canonical form (board is flipped when
    it is WHITE's turn so the current player always moves up).

    Planes::

        0: current player's men
        1: current player's kings
        2: opponent's men
        3: opponent's kings
        4: repetition flag (1.0 if position seen before)
        5: colour flag (1.0 if current player is BLACK, else 0.0)
        6: move count (t / 200, capped at 1.0)

    Action Space
    ------------
    625 discrete actions.  ``action = source_sq * 25 + target_sq``.
    The neural network outputs one atomic hop at a time; multi-step capture
    chains are handled by the environment maintaining turn state.

    Rewards
    -------
    From the perspective of the player who just acted:
    +1.0 for a win, -1.0 for a loss, 0.0 for a draw or ongoing game.
    """

    def __init__(self) -> None:
        self.game = GameState()
        self.action_space_n: int = ACTION_SPACE  # 625
        self.observation_shape: tuple = (7, BOARD_SIZE, BOARD_SIZE)  # (7, 5, 5)

    # ------------------------------------------------------------------
    # Gymnasium API: reset
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset the environment to the initial board position.

        Returns
        -------
        observation : np.ndarray
            Shape ``(7, 5, 5)``, dtype ``float32``.
        info : dict
            Contains ``'action_mask'``: a ``(625,)`` float32 binary mask
            of legal actions.
        """
        obs_list, mask_list = self.game.reset()
        obs = _obs_to_numpy(obs_list)
        mask = _mask_to_numpy(mask_list)
        return obs, {"action_mask": mask}

    # ------------------------------------------------------------------
    # Gymnasium API: step
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one atomic hop.

        Parameters
        ----------
        action : int
            An integer in ``[0, 625)``.  ``src = action // 25``,
            ``dst = action % 25``.

        Returns
        -------
        observation : np.ndarray
            Shape ``(7, 5, 5)``, dtype ``float32``.
        reward : float
            +1.0 win, -1.0 loss, 0.0 draw or ongoing.  From the
            perspective of the player who just acted.
        terminated : bool
            ``True`` if the game ended by rules (win, loss, draw).
        truncated : bool
            Always ``False`` -- there is no external time limit beyond the
            built-in 50-move rule.
        info : dict
            Contains ``'action_mask'``: a ``(625,)`` float32 binary mask
            of legal actions for the next player.
        """
        obs_list, mask_list, reward, done = self.game.step(action)
        obs = _obs_to_numpy(obs_list)
        mask = _mask_to_numpy(mask_list)
        terminated = done
        truncated = False
        return obs, float(reward), terminated, truncated, {"action_mask": mask}

    # ------------------------------------------------------------------
    # Action mask
    # ------------------------------------------------------------------

    def get_action_mask(self) -> np.ndarray:
        """Return a ``(625,)`` float32 binary mask of legal actions."""
        return _mask_to_numpy(self.game.get_action_mask())

    # ------------------------------------------------------------------
    # Clone (for MCTS)
    # ------------------------------------------------------------------

    def clone(self) -> KhreibagaEnv:
        """Return a deep copy of this environment (for MCTS tree search).

        The cloned environment is fully independent: modifying it does not
        affect the original.
        """
        new_env = KhreibagaEnv.__new__(KhreibagaEnv)
        new_env.game = self.game.clone()
        new_env.action_space_n = self.action_space_n
        new_env.observation_shape = self.observation_shape
        return new_env

    # ------------------------------------------------------------------
    # Properties (delegated to GameState)
    # ------------------------------------------------------------------

    @property
    def current_player(self) -> int:
        """The player whose turn it is: ``BLACK`` (1) or ``WHITE`` (2)."""
        return self.game.current_player

    @property
    def done(self) -> bool:
        """``True`` if the game has ended."""
        return self.game.done

    @property
    def winner(self) -> int | None:
        """The winner: ``BLACK`` (1), ``WHITE`` (2), or ``None`` (draw / ongoing)."""
        return self.game.winner

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Return a text representation of the current board state.

        Also prints the board to stdout via ``display_board``.
        """
        return display_board(self.game.board)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"KhreibagaEnv({self.game!r})"


# ---------------------------------------------------------------------------
# Smoke-test utility
# ---------------------------------------------------------------------------

def play_random_game(seed: int | None = None) -> tuple[int | None, int]:
    """Play a full game with uniformly random legal moves.

    Useful for smoke-testing the environment.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.  If ``None``, the default
        NumPy RNG is used.

    Returns
    -------
    winner : int or None
        ``BLACK`` (1), ``WHITE`` (2), or ``None`` (draw).
    move_count : int
        Total number of atomic hops (half-moves) played.
    """
    rng = np.random.default_rng(seed)
    env = KhreibagaEnv()
    obs, info = env.reset()

    while not env.done:
        mask = info["action_mask"]
        legal_actions = np.where(mask > 0)[0]
        if len(legal_actions) == 0:
            # Should not happen if the game engine is correct, but guard
            # against infinite loops during development.
            break
        action = rng.choice(legal_actions)
        obs, reward, terminated, truncated, info = env.step(int(action))

    return env.winner, env.game.move_count


# ---------------------------------------------------------------------------
# Main: run a quick smoke test when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running smoke test: play_random_game(seed=0) ...")
    winner, moves = play_random_game(seed=0)
    winner_str = {BLACK: "BLACK", WHITE: "WHITE", None: "DRAW"}[winner]
    print(f"Result: {winner_str} after {moves} moves.")

    print("\nFinal board:")
    env = KhreibagaEnv()
    rng = np.random.default_rng(0)
    obs, info = env.reset()
    while not env.done:
        mask = info["action_mask"]
        legal_actions = np.where(mask > 0)[0]
        if len(legal_actions) == 0:
            break
        action = rng.choice(legal_actions)
        obs, reward, terminated, truncated, info = env.step(int(action))
    env.render()
