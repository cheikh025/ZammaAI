"""
Khreibaga (Mauritanian Alquerque) board representation and geometry.

Defines the 5x5 board topology, adjacency structure with orthogonal and
diagonal connectivity, coordinate helpers, initial setup, and display.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOARD_SIZE: int = 5
NUM_SQUARES: int = BOARD_SIZE * BOARD_SIZE  # 25

# Piece types
EMPTY: int = 0
BLACK_MAN: int = 1
BLACK_KING: int = 2
WHITE_MAN: int = 3
WHITE_KING: int = 4

# Players
BLACK: int = 1
WHITE: int = 2

# All eight compass directions as (delta_row, delta_col)
ORTHOGONAL_DIRS: List[Tuple[int, int]] = [
    (-1, 0),  # south
    (1, 0),   # north
    (0, -1),  # west
    (0, 1),   # east
]

DIAGONAL_DIRS: List[Tuple[int, int]] = [
    (-1, -1),  # south-west
    (-1, 1),   # south-east
    (1, -1),   # north-west
    (1, 1),    # north-east
]

ALL_DIRS: List[Tuple[int, int]] = ORTHOGONAL_DIRS + DIAGONAL_DIRS

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def sq_to_rc(sq: int) -> Tuple[int, int]:
    """Convert a flat square index (0..24) to (row, col)."""
    return divmod(sq, BOARD_SIZE)


def rc_to_sq(r: int, c: int) -> int:
    """Convert (row, col) to a flat square index."""
    return r * BOARD_SIZE + c


# ---------------------------------------------------------------------------
# Adjacency / ray structure  (precomputed at module load time)
# ---------------------------------------------------------------------------

def _build_adjacency() -> List[Dict[Tuple[int, int], List[int]]]:
    """
    Build the master adjacency table.

    ADJACENCY[sq] is a dict mapping each valid direction (dr, dc) to a list
    of squares along that ray, ordered by increasing distance from *sq*.

    Orthogonal directions are available for every square.
    Diagonal directions are available only when (row + col) is even.
    """
    adj: List[Dict[Tuple[int, int], List[int]]] = [{} for _ in range(NUM_SQUARES)]

    for sq in range(NUM_SQUARES):
        r, c = sq_to_rc(sq)
        has_diag = (r + c) % 2 == 0

        directions = ORTHOGONAL_DIRS + (DIAGONAL_DIRS if has_diag else [])

        for dr, dc in directions:
            ray: List[int] = []
            cr, cc = r + dr, c + dc
            while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                ray.append(rc_to_sq(cr, cc))
                # Continue along this direction only if connectivity is
                # maintained at *each* intermediate square.  For orthogonal
                # directions this is always fine.  For diagonal directions
                # the next square must also support diagonals AND the
                # connecting parity must hold.
                #
                # On the Alquerque board, diagonal connectivity follows a
                # strict rule: a diagonal edge exists between (r1,c1) and
                # (r2,c2) iff BOTH endpoints have (r+c) even.  Since
                # stepping diagonally by (dr,dc) always changes (r+c) by
                # an even amount (dr+dc is even for diagonal steps), every
                # square along a diagonal ray also satisfies (r+c) even.
                # Therefore we can always extend the ray.
                cr += dr
                cc += dc

            if ray:
                adj[sq][(dr, dc)] = ray

    return adj


ADJACENCY: List[Dict[Tuple[int, int], List[int]]] = _build_adjacency()


def get_neighbors(sq: int) -> List[int]:
    """Return all squares immediately adjacent to *sq* (one step away)."""
    return [ray[0] for ray in ADJACENCY[sq].values()]


def get_ray(sq: int, direction: Tuple[int, int]) -> List[int]:
    """
    Return the list of squares along the ray from *sq* in *direction*,
    **excluding** *sq* itself, ordered by increasing distance.

    Returns an empty list if the direction is not valid for this square.
    """
    return list(ADJACENCY[sq].get(direction, []))


# ---------------------------------------------------------------------------
# Initial board setup
# ---------------------------------------------------------------------------

def initial_board() -> List[int]:
    """
    Return a list of 25 ints representing the Khreibaga starting position.

    Black (Player 1): rows 0-1 fully + indices 10, 11  (12 pieces).
    White (Player 2): rows 3-4 fully + indices 13, 14  (12 pieces).
    Centre (index 12) is EMPTY.
    """
    board = [EMPTY] * NUM_SQUARES

    # Black men ----------------------------------------------------------
    # Row 0: indices 0..4
    for sq in range(0, 5):
        board[sq] = BLACK_MAN
    # Row 1: indices 5..9
    for sq in range(5, 10):
        board[sq] = BLACK_MAN
    # Left side of Row 2: indices 10, 11
    board[10] = BLACK_MAN
    board[11] = BLACK_MAN

    # White men ----------------------------------------------------------
    # Right side of Row 2: indices 13, 14
    board[13] = WHITE_MAN
    board[14] = WHITE_MAN
    # Row 3: indices 15..19
    for sq in range(15, 20):
        board[sq] = WHITE_MAN
    # Row 4: indices 20..24
    for sq in range(20, 25):
        board[sq] = WHITE_MAN

    return board


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_PIECE_CHAR = {
    EMPTY: ".",
    BLACK_MAN: "b",
    BLACK_KING: "B",
    WHITE_MAN: "w",
    WHITE_KING: "W",
}


def display_board(board: List[int]) -> str:
    """
    Return a human-readable text representation of the board.

    Row 4 is printed at the top, row 0 at the bottom (matches visual
    orientation with Black starting at the bottom).

    Example output (initial position):

        4 | w  w  w  w  w
        3 | w  w  w  w  w
        2 | b  b  .  w  w
        1 | b  b  b  b  b
        0 | b  b  b  b  b
          +--------------
            0  1  2  3  4
    """
    lines: List[str] = []
    for r in range(BOARD_SIZE - 1, -1, -1):
        row_chars = [_PIECE_CHAR[board[rc_to_sq(r, c)]] for c in range(BOARD_SIZE)]
        lines.append(f"  {r} | {'  '.join(row_chars)}")
    lines.append("    +--------------")
    lines.append("      0  1  2  3  4")
    text = "\n".join(lines)
    print(text)
    return text
