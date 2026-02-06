"""
Khreibaga capture logic.

Implements compulsory capture detection, multi-step chain enumeration,
the majority capture rule, action mask generation, and hop execution.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from khreibga.board import (
    ADJACENCY,
    BLACK,
    BLACK_KING,
    BLACK_MAN,
    BOARD_SIZE,
    EMPTY,
    NUM_SQUARES,
    WHITE,
    WHITE_KING,
    WHITE_MAN,
    rc_to_sq,
    sq_to_rc,
)

# ---------------------------------------------------------------------------
# Helpers (mirror what move_gen provides, so captures.py is self-contained
# during early development; also importable from move_gen when available)
# ---------------------------------------------------------------------------

try:
    from khreibga.move_gen import get_player_pieces, get_simple_moves, is_player_piece
except ImportError:  # pragma: no cover – move_gen may not exist yet
    pass


def _is_player_piece(board: List[int], sq: int, player: int) -> bool:
    """Return True if *sq* holds a piece belonging to *player*."""
    piece = board[sq]
    if player == BLACK:
        return piece in (BLACK_MAN, BLACK_KING)
    return piece in (WHITE_MAN, WHITE_KING)


def _is_enemy_piece(board: List[int], sq: int, player: int) -> bool:
    """Return True if *sq* holds a piece belonging to the opponent."""
    piece = board[sq]
    if player == BLACK:
        return piece in (WHITE_MAN, WHITE_KING)
    return piece in (BLACK_MAN, BLACK_KING)


def _get_player_pieces(board: List[int], player: int) -> List[int]:
    """Return a list of square indices occupied by *player*'s pieces."""
    return [sq for sq in range(NUM_SQUARES) if _is_player_piece(board, sq, player)]


def _is_king(board: List[int], sq: int) -> bool:
    return board[sq] in (BLACK_KING, WHITE_KING)


def _is_man(board: List[int], sq: int) -> bool:
    return board[sq] in (BLACK_MAN, WHITE_MAN)


# ---------------------------------------------------------------------------
# Man captures (DFS)
# ---------------------------------------------------------------------------

def find_man_captures(
    board: List[int],
    sq: int,
    player: int,
    captured: Optional[Set[int]] = None,
    _origin: Optional[int] = None,
) -> List[List[Tuple[int, int]]]:
    """
    Find ALL maximal capture chains for a man starting at *sq*.

    Parameters
    ----------
    board : current board state (not mutated).
    sq    : square the man currently sits on (during recursion this is the
            landing square of the previous hop).
    player: BLACK or WHITE.
    captured: set of squares whose pieces have been captured earlier in this
              chain (they are effectively empty).
    _origin: (internal) the square the man occupied at the very start of
             this chain.  Treated as virtually empty because the man has
             left it.  Set automatically on the first call.

    Returns
    -------
    A list of chains. Each chain is a list of (src, dst) hops.
    An empty list means no capture is possible from *sq*.
    """
    if captured is None:
        captured = set()
    if _origin is None:
        _origin = sq

    chains: List[List[Tuple[int, int]]] = []

    for direction, ray in ADJACENCY[sq].items():
        if len(ray) < 2:
            continue

        enemy_sq = ray[0]
        land_sq = ray[1]

        # The enemy square must contain an opponent piece (and not already
        # captured in this chain).
        if enemy_sq in captured:
            continue
        if not _is_enemy_piece(board, enemy_sq, player):
            continue

        # The landing square must be effectively empty:
        #   - actually empty on the board, OR
        #   - its piece was captured earlier in this chain (immediate
        #     removal), OR
        #   - it is the origin square (the man has left it).
        land_piece = board[land_sq]
        land_ok = (
            land_piece == EMPTY
            or land_sq in captured
            or land_sq == _origin
        )
        if not land_ok:
            continue

        new_captured = captured | {enemy_sq}
        sub_chains = find_man_captures(
            board, land_sq, player, new_captured, _origin
        )

        if sub_chains:
            for sc in sub_chains:
                chains.append([(sq, land_sq)] + sc)
        else:
            # Terminal: no further capture from land_sq.
            chains.append([(sq, land_sq)])

    # If no chain was found at all, return empty (base case).
    return chains


# ---------------------------------------------------------------------------
# King captures (DFS) – flying king / long leap
# ---------------------------------------------------------------------------

def find_king_captures(
    board: List[int],
    sq: int,
    player: int,
    captured: Optional[Set[int]] = None,
    _origin: Optional[int] = None,
) -> List[List[Tuple[int, int]]]:
    """
    Find ALL maximal capture chains for a king starting at *sq*.

    The flying king can slide along a ray, jump over exactly one enemy piece,
    and land on any empty square beyond that enemy (before hitting another
    piece or the board edge).  The path from the king to the enemy must be
    clear (all empty or previously captured).

    Parameters
    ----------
    _origin : (internal) the square the king occupied at the very start
              of this chain.  Treated as virtually empty because the king
              has left it.

    Returns a list of chains, each chain being a list of (src, dst) hops.
    """
    if captured is None:
        captured = set()
    if _origin is None:
        _origin = sq

    chains: List[List[Tuple[int, int]]] = []

    for direction, ray in ADJACENCY[sq].items():
        # Walk along the ray looking for the first piece.
        enemy_idx: Optional[int] = None  # index into ray
        for i, ray_sq in enumerate(ray):
            cell = board[ray_sq]
            # The square is clear if it is empty, previously captured,
            # or is the origin (the king left that square).
            if ray_sq in captured or cell == EMPTY or ray_sq == _origin:
                continue
            if _is_enemy_piece(board, ray_sq, player):
                enemy_idx = i
                break
            else:
                # Friendly piece (or own piece at origin already handled)
                # blocks the ray.
                break

        if enemy_idx is None:
            continue

        enemy_sq = ray[enemy_idx]
        if enemy_sq in captured:
            continue

        # Enumerate landing squares beyond the enemy.
        for j in range(enemy_idx + 1, len(ray)):
            land_sq = ray[j]
            land_cell = board[land_sq]
            if (
                land_cell != EMPTY
                and land_sq not in captured
                and land_sq != _origin
            ):
                # Blocked by another piece.
                break

            new_captured = captured | {enemy_sq}
            sub_chains = find_king_captures(
                board, land_sq, player, new_captured, _origin
            )

            if sub_chains:
                for sc in sub_chains:
                    chains.append([(sq, land_sq)] + sc)
            else:
                chains.append([(sq, land_sq)])

    return chains


# ---------------------------------------------------------------------------
# Aggregate capture enumeration
# ---------------------------------------------------------------------------

def find_all_capture_chains(
    board: List[int],
    player: int,
) -> List[List[Tuple[int, int]]]:
    """
    Return every maximal capture chain available to *player*.

    Each chain is a list of (src, dst) hops.
    """
    all_chains: List[List[Tuple[int, int]]] = []

    for sq in _get_player_pieces(board, player):
        if _is_king(board, sq):
            chains = find_king_captures(board, sq, player)
        else:
            chains = find_man_captures(board, sq, player)
        all_chains.extend(chains)

    return all_chains


# ---------------------------------------------------------------------------
# Piece-local capture enumeration
# ---------------------------------------------------------------------------

def find_piece_capture_chains(
    board: List[int],
    player: int,
    sq: int,
) -> List[List[Tuple[int, int]]]:
    """
    Return capture chains for exactly one piece at *sq*.

    Used for mid-chain continuation where only the active piece is allowed to
    move; global majority filtering across other pieces must not interfere.
    """
    if not _is_player_piece(board, sq, player):
        return []
    if _is_king(board, sq):
        return find_king_captures(board, sq, player)
    return find_man_captures(board, sq, player)


# ---------------------------------------------------------------------------
# Majority rule
# ---------------------------------------------------------------------------

def apply_majority_rule(
    chains: List[List[Tuple[int, int]]],
) -> List[List[Tuple[int, int]]]:
    """
    Keep only chains that capture the maximum number of pieces.

    The number of captures equals `len(chain)` (one hop per capture).
    """
    if not chains:
        return []

    max_len = max(len(c) for c in chains)
    return [c for c in chains if len(c) == max_len]


# ---------------------------------------------------------------------------
# First-hops (atomic actions exposed to NN)
# ---------------------------------------------------------------------------

def get_capture_first_hops(
    board: List[int],
    player: int,
) -> List[Tuple[int, int]]:
    """
    Return a list of (src, dst) pairs representing the first hop of every
    legal capture sequence after the majority rule has been applied.

    Duplicates are removed (multiple chains may share the same first hop).
    """
    chains = find_all_capture_chains(board, player)
    best = apply_majority_rule(chains)

    seen: Set[Tuple[int, int]] = set()
    hops: List[Tuple[int, int]] = []
    for chain in best:
        first = chain[0]
        if first not in seen:
            seen.add(first)
            hops.append(first)
    return hops


def get_piece_capture_first_hops(
    board: List[int],
    player: int,
    sq: int,
) -> List[Tuple[int, int]]:
    """
    Return legal first hops for a specific piece at *sq* after piece-local
    majority filtering.

    This is used while a chain is already in progress and the active piece is
    locked for continuation.
    """
    chains = find_piece_capture_chains(board, player, sq)
    best = apply_majority_rule(chains)

    seen: Set[Tuple[int, int]] = set()
    hops: List[Tuple[int, int]] = []
    for chain in best:
        first = chain[0]
        if first not in seen:
            seen.add(first)
            hops.append(first)
    return hops


# ---------------------------------------------------------------------------
# Action mask (Algorithm 1 from spec)
# ---------------------------------------------------------------------------

def get_action_mask(board: List[int], player: int) -> List[int]:
    """
    Return a list of 625 ints (0 or 1).

    If captures exist, mask the first hops of maximum-length capture chains.
    Otherwise, mask simple (non-capturing) moves.

    Index formula: ``src * 25 + dst``.
    """
    mask = [0] * (NUM_SQUARES * NUM_SQUARES)

    capture_hops = get_capture_first_hops(board, player)
    if capture_hops:
        for src, dst in capture_hops:
            mask[src * NUM_SQUARES + dst] = 1
        return mask

    # No captures – fall back to simple moves.
    try:
        from khreibga.move_gen import get_simple_moves
        simple = get_simple_moves(board, player)
    except ImportError:
        # move_gen not available yet – return all-zero mask.
        simple = []

    for src, dst in simple:
        mask[src * NUM_SQUARES + dst] = 1

    return mask


# ---------------------------------------------------------------------------
# Execute a single atomic hop
# ---------------------------------------------------------------------------

def execute_hop(
    board: List[int],
    src: int,
    dst: int,
    player: int,
) -> Tuple[List[int], Optional[int]]:
    """
    Execute a single atomic hop on a *copy* of the board.

    For a capturing hop the captured piece(s) along the straight-line path
    from *src* to *dst* are removed.  For a simple (non-capturing) move the
    piece just slides.

    Returns
    -------
    (new_board, captured_sq)
        *captured_sq* is the index of the enemy piece that was jumped, or
        ``None`` for a simple (non-capturing) move.
    """
    new_board = list(board)
    piece = new_board[src]

    # Determine the direction from src to dst.
    sr, sc = sq_to_rc(src)
    dr, dc = sq_to_rc(dst)

    row_diff = dr - sr
    col_diff = dc - sc

    # Normalise to a unit direction.
    def _sign(x: int) -> int:
        return (x > 0) - (x < 0)

    direction = (_sign(row_diff), _sign(col_diff))

    # Walk from src toward dst along the ray, looking for an enemy.
    captured_sq: Optional[int] = None
    r, c = sr + direction[0], sc + direction[1]
    while (r, c) != (dr, dc):
        sq = rc_to_sq(r, c)
        if _is_enemy_piece(new_board, sq, player):
            captured_sq = sq
            new_board[sq] = EMPTY
            break
        r += direction[0]
        c += direction[1]

    # Move the piece.
    new_board[dst] = piece
    new_board[src] = EMPTY

    # Promotion check: a man that STOPS on the opponent's back rank promotes.
    # (If further captures are available the caller should NOT promote yet;
    # however, execute_hop itself is atomic and unaware of continuation.
    # The caller – the game environment – is responsible for deferring
    # promotion when the chain continues.  We promote here unconditionally;
    # the environment must undo this if the chain continues.)
    # Actually, to keep this function simple and let the environment handle
    # promotion, we do NOT promote here.  The environment will check after
    # the chain completes.

    return new_board, captured_sq
