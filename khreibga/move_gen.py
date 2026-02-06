"""
Simple (non-capture) move generator for Khreibaga.

Provides functions to enumerate all legal non-capturing moves for men and
kings, respecting directionality rules and the flying-king mechanic.

Capture moves are handled separately by Agent 3's capture generator.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from khreibga.board import (
    ADJACENCY,
    BLACK,
    BLACK_KING,
    BLACK_MAN,
    EMPTY,
    NUM_SQUARES,
    WHITE,
    WHITE_KING,
    WHITE_MAN,
)

# ---------------------------------------------------------------------------
# Forward direction tables
# ---------------------------------------------------------------------------
# Black (player 1) advances in the Row+ direction (dr > 0).
# White (player 2) advances in the Row- direction (dr < 0).
#
# For simple (non-capture) moves, men may only move forward or
# forward-diagonal -- never sideways or backward.

FORWARD_DIRS: Dict[int, List[Tuple[int, int]]] = {
    BLACK: [(dr, dc) for dr, dc in [(1, 0), (1, 1), (1, -1)]],
    WHITE: [(dr, dc) for dr, dc in [(-1, 0), (-1, 1), (-1, -1)]],
}

# ---------------------------------------------------------------------------
# Piece ownership helpers
# ---------------------------------------------------------------------------

_PLAYER_PIECES: Dict[int, Tuple[int, int]] = {
    BLACK: (BLACK_MAN, BLACK_KING),
    WHITE: (WHITE_MAN, WHITE_KING),
}

_PLAYER_MEN: Dict[int, int] = {
    BLACK: BLACK_MAN,
    WHITE: WHITE_MAN,
}

_PLAYER_KINGS: Dict[int, int] = {
    BLACK: BLACK_KING,
    WHITE: WHITE_KING,
}


def is_player_piece(board: List[int], sq: int, player: int) -> bool:
    """Return True if the piece at *sq* belongs to *player*."""
    return board[sq] in _PLAYER_PIECES[player]


def get_player_pieces(board: List[int], player: int) -> List[int]:
    """Return a list of square indices where *player* has pieces."""
    man, king = _PLAYER_PIECES[player]
    return [sq for sq in range(NUM_SQUARES) if board[sq] == man or board[sq] == king]


# ---------------------------------------------------------------------------
# Man simple moves
# ---------------------------------------------------------------------------

def get_man_simple_moves(
    board: List[int], sq: int, player: int
) -> List[Tuple[int, int]]:
    """Return simple (non-capture) moves for a man at *sq*.

    Men move exactly one step in a forward or forward-diagonal direction
    into an empty adjacent square.  Only directions present in the
    adjacency graph for *sq* are considered (diagonals are absent from
    squares where ``(r+c)`` is odd).

    Returns a list of ``(source, destination)`` tuples.
    """
    moves: List[Tuple[int, int]] = []
    sq_adj = ADJACENCY[sq]  # dict[direction -> ray]

    for direction in FORWARD_DIRS[player]:
        ray = sq_adj.get(direction)
        if ray is not None and board[ray[0]] == EMPTY:
            moves.append((sq, ray[0]))

    return moves


# ---------------------------------------------------------------------------
# King simple moves  (flying king)
# ---------------------------------------------------------------------------

def get_king_simple_moves(
    board: List[int], sq: int
) -> List[Tuple[int, int]]:
    """Return simple (non-capture) moves for a king at *sq*.

    Kings are "flying" -- they may move any distance along a clear line
    (orthogonal or diagonal) in any direction.  The entire path up to the
    destination must be empty; the king stops before the first obstruction.

    Returns a list of ``(source, destination)`` tuples.
    """
    moves: List[Tuple[int, int]] = []

    for _direction, ray in ADJACENCY[sq].items():
        for target_sq in ray:
            if board[target_sq] == EMPTY:
                moves.append((sq, target_sq))
            else:
                # Path is blocked -- no further squares along this ray.
                break

    return moves


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------

def get_simple_moves(
    board: List[int], player: int
) -> List[Tuple[int, int]]:
    """Return **all** simple (non-capture) moves for *player*.

    Iterates over every piece belonging to *player* and collects moves
    according to piece type:

    * **Men** -- one step forward / forward-diagonal into an empty square.
    * **Kings** -- any distance along a clear line in any direction.

    Returns a list of ``(source, destination)`` tuples.

    Note: This function does **not** check whether captures are available.
    The caller (legal-move generator / action-mask builder) is responsible
    for suppressing simple moves when compulsory captures exist.
    """
    man = _PLAYER_MEN[player]
    king = _PLAYER_KINGS[player]
    moves: List[Tuple[int, int]] = []

    for sq in range(NUM_SQUARES):
        piece = board[sq]
        if piece == man:
            moves.extend(get_man_simple_moves(board, sq, player))
        elif piece == king:
            moves.extend(get_king_simple_moves(board, sq))

    return moves
