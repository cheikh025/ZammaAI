"""
Canonical state encoder for Khreibaga Zero.

Converts raw game state (board, player, history, move count) into a
(7, 5, 5) float32 NumPy tensor suitable for PyTorch channel-first input.

Plane layout
------------
  0 : Current player's Men      (1.0 where present)
  1 : Current player's Kings    (1.0 where present)
  2 : Opponent's Men             (1.0 where present)
  3 : Opponent's Kings           (1.0 where present)
  4 : Repetitions                (1.0 if board state has appeared before)
  5 : Colour                     (all 1.0 if current player is BLACK, else 0.0)
  6 : Move Count                 (t / MAX_STEPS, capped at 1.0)

When the current player is WHITE the board is flipped via (r, c) -> (4-r, 4-c)
so the network always sees the current player "moving up".
"""

from __future__ import annotations

import numpy as np

from khreibga.board import (
    BLACK,
    BLACK_KING,
    BLACK_MAN,
    BOARD_SIZE,
    EMPTY,
    NUM_SQUARES,
    WHITE,
    WHITE_KING,
    WHITE_MAN,
    sq_to_rc,
)

# Pre-computed lookup: for each flat square index 0..24 give (row, col).
# Used to vectorise the board -> planes mapping without per-square function calls.
_SQ_TO_RC = np.array([sq_to_rc(sq) for sq in range(NUM_SQUARES)], dtype=np.intp)

# Flipped coordinates: (4 - r, 4 - c) for the WHITE canonical transform.
_SQ_TO_RC_FLIPPED = (BOARD_SIZE - 1) - _SQ_TO_RC


def encode_observation(
    board: list[int],
    current_player: int,
    history: dict[int, int],
    current_hash: int,
    move_count: int,
    max_steps: int = 200,
) -> np.ndarray:
    """Encode a Khreibaga game state as a (7, 5, 5) float32 NumPy array.

    Parameters
    ----------
    board : list[int]
        Length-25 flat board array.  Each element is one of
        EMPTY (0), BLACK_MAN (1), BLACK_KING (2), WHITE_MAN (3), WHITE_KING (4).
    current_player : int
        BLACK (1) or WHITE (2).
    history : dict[int, int]
        Mapping from Zobrist hash to occurrence count.
    current_hash : int
        Zobrist hash of the *current* position.
    move_count : int
        Number of half-moves played so far.
    max_steps : int, optional
        Normalisation constant for the move-count plane (default 200).

    Returns
    -------
    np.ndarray
        Shape (7, 5, 5), dtype float32.
    """
    obs = np.zeros((7, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Determine piece codes for "my" and "opponent" relative to current player.
    if current_player == BLACK:
        my_man, my_king = BLACK_MAN, BLACK_KING
        opp_man, opp_king = WHITE_MAN, WHITE_KING
        coords = _SQ_TO_RC          # no flip
    else:
        my_man, my_king = WHITE_MAN, WHITE_KING
        opp_man, opp_king = BLACK_MAN, BLACK_KING
        coords = _SQ_TO_RC_FLIPPED  # flip for canonical form

    # Convert flat board to a NumPy array for vectorised comparisons.
    board_arr = np.array(board, dtype=np.intp)

    # Row and column indices in the *observation* tensor for each flat square.
    rows = coords[:, 0]
    cols = coords[:, 1]

    # Planes 0-3: piece occupancy
    obs[0, rows, cols] = (board_arr == my_man).astype(np.float32)
    obs[1, rows, cols] = (board_arr == my_king).astype(np.float32)
    obs[2, rows, cols] = (board_arr == opp_man).astype(np.float32)
    obs[3, rows, cols] = (board_arr == opp_king).astype(np.float32)

    # Plane 4: repetition -- 1.0 if this exact position has been seen before
    if history.get(current_hash, 0) > 1:
        obs[4, :, :] = 1.0

    # Plane 5: colour -- 1.0 everywhere when current player is BLACK
    if current_player == BLACK:
        obs[5, :, :] = 1.0

    # Plane 6: normalised move count, capped at 1.0
    obs[6, :, :] = min(move_count / max_steps, 1.0)

    return obs


def encode_action_mask(mask: list[int]) -> np.ndarray:
    """Convert a 625-element action mask to a (625,) float32 NumPy array.

    Parameters
    ----------
    mask : list[int]
        Length-625 list of 0s and 1s indicating legal actions.

    Returns
    -------
    np.ndarray
        Shape (625,), dtype float32.
    """
    return np.array(mask, dtype=np.float32)
