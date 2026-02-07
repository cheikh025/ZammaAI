"""
Tests for khreibga.move_gen -- simple (non-capture) move generation.

Covers:
  - Initial position move counts
  - Man directionality (forward only, no backward)
  - King flying movement (long range along clear lines)
  - King blocked by friendly or enemy pieces
  - Edge and corner cases
  - Single man on an empty board
  - FORWARD_DIRS correctness
  - Helper utilities (is_player_piece, get_player_pieces)
"""

from __future__ import annotations

import pytest

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
    initial_board,
    rc_to_sq,
    sq_to_rc,
)
from khreibga.move_gen import (
    FORWARD_DIRS,
    get_king_simple_moves,
    get_man_simple_moves,
    get_player_pieces,
    get_simple_moves,
    is_player_piece,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def empty_board() -> list[int]:
    """Return a completely empty board."""
    return [EMPTY] * NUM_SQUARES


def board_with(pieces: dict[int, int]) -> list[int]:
    """Return an empty board with specific pieces placed.

    *pieces* maps square-index -> piece-type.
    """
    board = empty_board()
    for sq, piece in pieces.items():
        board[sq] = piece
    return board


# ===================================================================
# FORWARD_DIRS sanity
# ===================================================================

class TestForwardDirs:
    """Verify the FORWARD_DIRS table is consistent with the spec."""

    def test_black_forward_dirs_all_positive_dr(self):
        for dr, _dc in FORWARD_DIRS[BLACK]:
            assert dr > 0, f"Black forward dir has non-positive dr: ({dr}, {_dc})"

    def test_white_forward_dirs_all_negative_dr(self):
        for dr, _dc in FORWARD_DIRS[WHITE]:
            assert dr < 0, f"White forward dir has non-positive dr: ({dr}, {_dc})"

    def test_black_has_three_forward_dirs(self):
        assert len(FORWARD_DIRS[BLACK]) == 3  # (1,0), (1,1), (1,-1)

    def test_white_has_three_forward_dirs(self):
        assert len(FORWARD_DIRS[WHITE]) == 3  # (-1,0), (-1,1), (-1,-1)

    def test_no_sideways_in_forward(self):
        """Pure sideways (dr==0) must not appear in FORWARD_DIRS."""
        for player in (BLACK, WHITE):
            for dr, _dc in FORWARD_DIRS[player]:
                assert dr != 0


# ===================================================================
# is_player_piece / get_player_pieces
# ===================================================================

class TestHelpers:

    def test_is_player_piece_black_man(self):
        board = board_with({0: BLACK_MAN})
        assert is_player_piece(board, 0, BLACK)
        assert not is_player_piece(board, 0, WHITE)

    def test_is_player_piece_white_king(self):
        board = board_with({12: WHITE_KING})
        assert is_player_piece(board, 12, WHITE)
        assert not is_player_piece(board, 12, BLACK)

    def test_is_player_piece_empty(self):
        board = empty_board()
        assert not is_player_piece(board, 0, BLACK)
        assert not is_player_piece(board, 0, WHITE)

    def test_get_player_pieces_initial(self):
        board = initial_board()
        black_pieces = get_player_pieces(board, BLACK)
        white_pieces = get_player_pieces(board, WHITE)
        assert len(black_pieces) == 12
        assert len(white_pieces) == 12
        # No overlap
        assert set(black_pieces).isdisjoint(set(white_pieces))

    def test_get_player_pieces_empty_board(self):
        board = empty_board()
        assert get_player_pieces(board, BLACK) == []
        assert get_player_pieces(board, WHITE) == []

    def test_get_player_pieces_includes_kings(self):
        board = board_with({0: BLACK_MAN, 12: BLACK_KING})
        pieces = get_player_pieces(board, BLACK)
        assert 0 in pieces
        assert 12 in pieces


# ===================================================================
# Man directionality
# ===================================================================

class TestManDirectionality:
    """Men can only move forward and forward-diagonal (never sideways/back)."""

    def test_black_man_center_moves_forward_only(self):
        """Black man at (2,2) on empty board: should move to row 3 only."""
        sq = rc_to_sq(2, 2)  # index 12, (r+c)=4 even -> has diagonals
        board = board_with({sq: BLACK_MAN})
        moves = get_man_simple_moves(board, sq, BLACK)

        dsts = [dst for _src, dst in moves]
        for dst in dsts:
            dr, dc = sq_to_rc(dst)
            assert dr == 3, (
                f"Black man at (2,2) reached ({dr},{dc}) which is not row 3"
            )

        # Expect three forward destinations: (3,2), (3,1), (3,3)
        expected = {rc_to_sq(3, 2), rc_to_sq(3, 1), rc_to_sq(3, 3)}
        assert set(dsts) == expected

    def test_black_man_no_backward(self):
        """Black man at (2,2) must NOT reach row 1."""
        sq = rc_to_sq(2, 2)
        board = board_with({sq: BLACK_MAN})
        moves = get_man_simple_moves(board, sq, BLACK)
        dsts = {dst for _src, dst in moves}
        backward = {rc_to_sq(1, 1), rc_to_sq(1, 2), rc_to_sq(1, 3)}
        assert dsts.isdisjoint(backward), "Man moved backward!"

    def test_white_man_center_moves_forward_only(self):
        """White man at (2,2) should move to row 1 only."""
        sq = rc_to_sq(2, 2)
        board = board_with({sq: WHITE_MAN})
        moves = get_man_simple_moves(board, sq, WHITE)

        dsts = [dst for _src, dst in moves]
        for dst in dsts:
            dr, dc = sq_to_rc(dst)
            assert dr == 1, (
                f"White man at (2,2) reached ({dr},{dc}) which is not row 1"
            )

        expected = {rc_to_sq(1, 2), rc_to_sq(1, 1), rc_to_sq(1, 3)}
        assert set(dsts) == expected

    def test_white_man_no_backward(self):
        """White man at (2,2) must NOT reach row 3."""
        sq = rc_to_sq(2, 2)
        board = board_with({sq: WHITE_MAN})
        moves = get_man_simple_moves(board, sq, WHITE)
        dsts = {dst for _src, dst in moves}
        backward = {rc_to_sq(3, 1), rc_to_sq(3, 2), rc_to_sq(3, 3)}
        assert dsts.isdisjoint(backward), "Man moved backward!"

    def test_man_no_sideways(self):
        """A man must not move pure sideways (dr == 0)."""
        sq = rc_to_sq(2, 2)
        board = board_with({sq: BLACK_MAN})
        moves = get_man_simple_moves(board, sq, BLACK)
        dsts = {dst for _src, dst in moves}
        sideways = {rc_to_sq(2, 1), rc_to_sq(2, 3)}
        assert dsts.isdisjoint(sideways), "Man moved sideways!"

    def test_man_blocked_by_friendly(self):
        """Man cannot move to a square occupied by a friendly piece."""
        sq = rc_to_sq(2, 2)
        blocker = rc_to_sq(3, 2)
        board = board_with({sq: BLACK_MAN, blocker: BLACK_MAN})
        moves = get_man_simple_moves(board, sq, BLACK)
        dsts = {dst for _src, dst in moves}
        assert blocker not in dsts

    def test_man_blocked_by_enemy(self):
        """Man cannot move (simple) to a square occupied by an enemy piece."""
        sq = rc_to_sq(2, 2)
        blocker = rc_to_sq(3, 2)
        board = board_with({sq: BLACK_MAN, blocker: WHITE_MAN})
        moves = get_man_simple_moves(board, sq, BLACK)
        dsts = {dst for _src, dst in moves}
        assert blocker not in dsts

    def test_man_on_odd_parity_square_no_diagonals(self):
        """On a square where (r+c) is odd, no diagonal moves exist."""
        sq = rc_to_sq(2, 1)  # (r+c)=3 odd -> no diagonals
        board = board_with({sq: BLACK_MAN})
        moves = get_man_simple_moves(board, sq, BLACK)
        dsts = {dst for _src, dst in moves}
        # Only orthogonal forward: (3,1)
        assert dsts == {rc_to_sq(3, 1)}

    def test_man_on_even_parity_square_has_diagonals(self):
        """On a square where (r+c) is even, diagonal forward moves exist."""
        sq = rc_to_sq(2, 0)  # (r+c)=2 even -> has diagonals
        board = board_with({sq: BLACK_MAN})
        moves = get_man_simple_moves(board, sq, BLACK)
        dsts = {dst for _src, dst in moves}
        # Forward orthogonal (3,0) + forward-diagonal (3,1)
        # (3,-1) is off-board, so only two moves
        expected = {rc_to_sq(3, 0), rc_to_sq(3, 1)}
        assert dsts == expected


# ===================================================================
# King flying movement
# ===================================================================

class TestKingFlying:
    """Flying kings can move any distance along clear lines in any direction."""

    def test_king_at_corner_00(self):
        """King at (0,0) -- (r+c)=0 even, has diagonals.

        Expected reachable squares:
          Orthogonal (1,0): (1,0),(2,0),(3,0),(4,0)
          Orthogonal (0,1): (0,1),(0,2),(0,3),(0,4)
          Diagonal (1,1):   (1,1),(2,2),(3,3),(4,4)
        No (-1,*) or (*,-1) directions exist from corner (0,0).
        """
        sq = rc_to_sq(0, 0)
        board = board_with({sq: BLACK_KING})
        moves = get_king_simple_moves(board, sq)
        dsts = {dst for _src, dst in moves}

        expected = set()
        # (1,0) ray
        for r in range(1, 5):
            expected.add(rc_to_sq(r, 0))
        # (0,1) ray
        for c in range(1, 5):
            expected.add(rc_to_sq(0, c))
        # (1,1) diagonal ray
        for d in range(1, 5):
            expected.add(rc_to_sq(d, d))

        assert dsts == expected
        assert len(dsts) == 12  # 4 + 4 + 4

    def test_king_at_corner_44(self):
        """King at (4,4): rays go (-1,0), (0,-1), (-1,-1)."""
        sq = rc_to_sq(4, 4)
        board = board_with({sq: WHITE_KING})
        moves = get_king_simple_moves(board, sq)
        dsts = {dst for _src, dst in moves}

        expected = set()
        for r in range(3, -1, -1):
            expected.add(rc_to_sq(r, 4))
        for c in range(3, -1, -1):
            expected.add(rc_to_sq(4, c))
        for d in range(1, 5):
            expected.add(rc_to_sq(4 - d, 4 - d))

        assert dsts == expected
        assert len(dsts) == 12

    def test_king_at_center(self):
        """King at (2,2) -- has 8 directions (even parity).

        All rays are clear on an empty board.
        """
        sq = rc_to_sq(2, 2)
        board = board_with({sq: BLACK_KING})
        moves = get_king_simple_moves(board, sq)
        dsts = {dst for _src, dst in moves}

        # Should reach every other square on the board that is reachable
        # via the adjacency rays.  Let's compute it from ADJACENCY directly.
        expected = set()
        for _direction, ray in ADJACENCY[sq].items():
            for target in ray:
                expected.add(target)

        assert dsts == expected

    def test_king_at_odd_parity_no_diagonals(self):
        """King at (1,0) -- (r+c)=1 odd, no diagonal connections.

        Only orthogonal rays.
        """
        sq = rc_to_sq(1, 0)
        board = board_with({sq: BLACK_KING})
        moves = get_king_simple_moves(board, sq)
        dsts = {dst for _src, dst in moves}

        expected = set()
        # Up: (2,0),(3,0),(4,0)
        for r in range(2, 5):
            expected.add(rc_to_sq(r, 0))
        # Down: (0,0)
        expected.add(rc_to_sq(0, 0))
        # Right: (1,1),(1,2),(1,3),(1,4)
        for c in range(1, 5):
            expected.add(rc_to_sq(1, c))
        # Left: nothing (col 0 already)

        assert dsts == expected


# ===================================================================
# King blocked by pieces
# ===================================================================

class TestKingBlocked:
    """Kings cannot pass through occupied squares."""

    def test_king_blocked_by_friendly_orthogonal(self):
        """King at (0,0), friendly piece at (2,0).

        Along (1,0) ray: can reach (1,0), but NOT (2,0),(3,0),(4,0).
        Other rays unaffected.
        """
        king_sq = rc_to_sq(0, 0)
        blocker_sq = rc_to_sq(2, 0)
        board = board_with({king_sq: BLACK_KING, blocker_sq: BLACK_MAN})
        moves = get_king_simple_moves(board, king_sq)
        dsts = {dst for _src, dst in moves}

        # (1,0) reachable
        assert rc_to_sq(1, 0) in dsts
        # (2,0) and beyond not reachable (blocked)
        assert rc_to_sq(2, 0) not in dsts
        assert rc_to_sq(3, 0) not in dsts
        assert rc_to_sq(4, 0) not in dsts

    def test_king_blocked_by_enemy_orthogonal(self):
        """King at (0,0), enemy piece at (2,0).

        Simple moves cannot jump over or land on enemy pieces.
        Same result as friendly blocker for simple moves.
        """
        king_sq = rc_to_sq(0, 0)
        blocker_sq = rc_to_sq(2, 0)
        board = board_with({king_sq: BLACK_KING, blocker_sq: WHITE_MAN})
        moves = get_king_simple_moves(board, king_sq)
        dsts = {dst for _src, dst in moves}

        assert rc_to_sq(1, 0) in dsts
        assert rc_to_sq(2, 0) not in dsts
        assert rc_to_sq(3, 0) not in dsts
        assert rc_to_sq(4, 0) not in dsts

    def test_king_blocked_by_friendly_diagonal(self):
        """King at (0,0), friendly piece at (2,2).

        Along (1,1) diagonal: can reach (1,1), but NOT (2,2),(3,3),(4,4).
        """
        king_sq = rc_to_sq(0, 0)
        blocker_sq = rc_to_sq(2, 2)
        board = board_with({king_sq: BLACK_KING, blocker_sq: BLACK_MAN})
        moves = get_king_simple_moves(board, king_sq)
        dsts = {dst for _src, dst in moves}

        assert rc_to_sq(1, 1) in dsts
        assert rc_to_sq(2, 2) not in dsts
        assert rc_to_sq(3, 3) not in dsts
        assert rc_to_sq(4, 4) not in dsts

    def test_king_blocked_at_distance_1(self):
        """King at (0,0), blocker right next to it at (1,0).

        The entire (1,0) ray is blocked -- no moves in that direction.
        """
        king_sq = rc_to_sq(0, 0)
        blocker_sq = rc_to_sq(1, 0)
        board = board_with({king_sq: BLACK_KING, blocker_sq: WHITE_MAN})
        moves = get_king_simple_moves(board, king_sq)
        dsts = {dst for _src, dst in moves}

        # None of the (1,0) ray is reachable
        for r in range(1, 5):
            assert rc_to_sq(r, 0) not in dsts

    def test_king_surrounded(self):
        """King completely surrounded by pieces has no simple moves."""
        king_sq = rc_to_sq(2, 2)
        board = board_with({king_sq: BLACK_KING})
        # Block every immediate neighbor
        for nbr in ADJACENCY[king_sq]:
            ray = ADJACENCY[king_sq][nbr]
            if ray:
                board[ray[0]] = WHITE_MAN
        moves = get_king_simple_moves(board, king_sq)
        assert len(moves) == 0


# ===================================================================
# Edge / corner cases
# ===================================================================

class TestEdgeCases:
    """Pieces at board edges have fewer available directions."""

    def test_man_bottom_left_corner(self):
        """Black man at (0,0): forward is (1,0), forward-diag (1,1).

        (0,0) has (r+c)=0 even, so diagonals exist.
        Expect moves to (1,0) and (1,1).
        """
        sq = rc_to_sq(0, 0)
        board = board_with({sq: BLACK_MAN})
        moves = get_man_simple_moves(board, sq, BLACK)
        dsts = {dst for _src, dst in moves}
        expected = {rc_to_sq(1, 0), rc_to_sq(1, 1)}
        assert dsts == expected

    def test_man_top_row_cannot_advance(self):
        """Black man at (4,2): row 4 is the last row.

        Forward (dr=1) goes off-board => no simple moves.
        """
        sq = rc_to_sq(4, 2)
        board = board_with({sq: BLACK_MAN})
        moves = get_man_simple_moves(board, sq, BLACK)
        assert len(moves) == 0

    def test_white_man_bottom_row_cannot_advance(self):
        """White man at (0,2): row 0 is the last row for white.

        Forward (dr=-1) goes off-board => no simple moves.
        """
        sq = rc_to_sq(0, 2)
        board = board_with({sq: WHITE_MAN})
        moves = get_man_simple_moves(board, sq, WHITE)
        assert len(moves) == 0

    def test_man_right_edge(self):
        """Black man at (0,4): (r+c)=4 even, has diagonals.

        Forward (1,0) -> (1,4) OK
        Forward-diag (1,1) -> (1,5) off-board
        Forward-diag (1,-1) -> (1,3) OK
        """
        sq = rc_to_sq(0, 4)
        board = board_with({sq: BLACK_MAN})
        moves = get_man_simple_moves(board, sq, BLACK)
        dsts = {dst for _src, dst in moves}
        expected = {rc_to_sq(1, 4), rc_to_sq(1, 3)}
        assert dsts == expected

    def test_man_left_edge(self):
        """Black man at (0,0): (r+c)=0 even.

        Forward-diag (1,-1) -> (1,-1) off-board.
        Expect (1,0) and (1,1).
        """
        sq = rc_to_sq(0, 0)
        board = board_with({sq: BLACK_MAN})
        moves = get_man_simple_moves(board, sq, BLACK)
        dsts = {dst for _src, dst in moves}
        assert dsts == {rc_to_sq(1, 0), rc_to_sq(1, 1)}

    def test_king_corner_04(self):
        """King at (0,4): (r+c)=4 even, diagonals available.

        Rays: (1,0), (0,-1), (1,-1).
        No (-1,*) or (*,+1) directions from this corner.
        """
        sq = rc_to_sq(0, 4)
        board = board_with({sq: BLACK_KING})
        moves = get_king_simple_moves(board, sq)
        dsts = {dst for _src, dst in moves}

        expected = set()
        # (1,0): (1,4),(2,4),(3,4),(4,4)
        for r in range(1, 5):
            expected.add(rc_to_sq(r, 4))
        # (0,-1): (0,3),(0,2),(0,1),(0,0)
        for c in range(3, -1, -1):
            expected.add(rc_to_sq(0, c))
        # (1,-1): (1,3),(2,2),(3,1),(4,0)
        for d in range(1, 5):
            expected.add(rc_to_sq(0 + d, 4 - d))

        assert dsts == expected


# ===================================================================
# Single man on an empty board  (exhaustive move set)
# ===================================================================

class TestSingleManEmptyBoard:
    """Verify exact move sets for isolated men."""

    def test_black_man_at_22(self):
        """Black man at (2,2), even parity.

        Forward: (3,2)
        Forward-diag: (3,1), (3,3)
        """
        sq = rc_to_sq(2, 2)
        board = board_with({sq: BLACK_MAN})
        moves = get_simple_moves(board, BLACK)
        assert set(moves) == {
            (sq, rc_to_sq(3, 2)),
            (sq, rc_to_sq(3, 1)),
            (sq, rc_to_sq(3, 3)),
        }

    def test_white_man_at_22(self):
        """White man at (2,2), even parity.

        Forward: (1,2)
        Forward-diag: (1,1), (1,3)
        """
        sq = rc_to_sq(2, 2)
        board = board_with({sq: WHITE_MAN})
        moves = get_simple_moves(board, WHITE)
        assert set(moves) == {
            (sq, rc_to_sq(1, 2)),
            (sq, rc_to_sq(1, 1)),
            (sq, rc_to_sq(1, 3)),
        }

    def test_black_man_at_21_odd_parity(self):
        """Black man at (2,1), (r+c)=3 odd => no diagonals.

        Only forward orthogonal: (3,1).
        """
        sq = rc_to_sq(2, 1)
        board = board_with({sq: BLACK_MAN})
        moves = get_simple_moves(board, BLACK)
        assert set(moves) == {(sq, rc_to_sq(3, 1))}

    def test_black_man_at_00(self):
        """Black man at (0,0), even parity.

        Forward: (1,0)
        Forward-diag (1,1): (1,1) -- available because even parity
        Forward-diag (1,-1): off board
        """
        sq = rc_to_sq(0, 0)
        board = board_with({sq: BLACK_MAN})
        moves = get_simple_moves(board, BLACK)
        assert set(moves) == {
            (sq, rc_to_sq(1, 0)),
            (sq, rc_to_sq(1, 1)),
        }


# ===================================================================
# Initial position -- limited forward moves
# ===================================================================

class TestInitialPosition:
    """In the starting position, most pieces are blocked by friendly pieces."""

    def test_black_initial_move_count(self):
        """Black has limited moves: only row-2 pieces and some row-1 pieces
        can advance forward into empty squares.

        Initial layout:
          Row 4: W W W W W
          Row 3: W W W W W
          Row 2: B B . W W     (indices 10,11 = Black; 12 = empty; 13,14 = White)
          Row 1: B B B B B
          Row 0: B B B B B

        Black forward = dr > 0 (toward higher rows).

        Row 0 pieces (sq 0..4): all blocked by row 1 (full of black).
        Row 1 pieces (sq 5..9):
          - sq 5 = (1,0): forward (2,0)=sq10 is BLACK => blocked.
                          (r+c)=1 odd => no diag. => 0 moves.
          - sq 6 = (1,1): forward (2,1)=sq11 is BLACK => blocked.
                          (r+c)=2 even => diag (1,1)->(2,2)=sq12 EMPTY! => 1 move.
                          diag (1,-1)->(2,0)=sq10 is BLACK => blocked.
          - sq 7 = (1,2): forward (2,2)=sq12 is EMPTY => 1 move.
                          (r+c)=3 odd => no diag. => total 1 move.
          - sq 8 = (1,3): forward (2,3)=sq13 is WHITE => blocked.
                          (r+c)=4 even => diag (1,1)->(2,4)=sq14 is WHITE => blocked.
                          diag (1,-1)->(2,2)=sq12 EMPTY => 1 move.
          - sq 9 = (1,4): forward (2,4)=sq14 is WHITE => blocked.
                          (r+c)=5 odd => no diag. => 0 moves.
        Row 2:
          - sq 10 = (2,0): forward (3,0)=sq15 is WHITE => blocked.
                           (r+c)=2 even => diag (1,1)->(3,1)=sq16 WHITE => blocked.
                           diag (1,-1)->(3,-1) off-board. => 0 moves.
          - sq 11 = (2,1): forward (3,1)=sq16 is WHITE => blocked.
                           (r+c)=3 odd => no diag. => 0 moves.

        Total black moves: sq6->sq12, sq7->sq12, sq8->sq12 = 3 moves.
        """
        board = initial_board()
        moves = get_simple_moves(board, BLACK)
        assert len(moves) == 3

        # All moves should target the center (sq 12)
        center = rc_to_sq(2, 2)
        for src, dst in moves:
            assert dst == center, f"Expected destination {center}, got {dst}"

        # Sources should be sq 6, 7, 8
        sources = sorted(src for src, _dst in moves)
        assert sources == [rc_to_sq(1, 1), rc_to_sq(1, 2), rc_to_sq(1, 3)]

    def test_white_initial_move_count(self):
        """White has the symmetric situation -- 3 moves, all targeting center.

        Row 4 pieces: blocked by row 3.
        Row 3 pieces:
          - sq 15 = (3,0): forward(dr=-1)->(2,0)=sq10 BLACK => blocked.
                           (r+c)=3 odd => no diag => 0 moves.
          - sq 16 = (3,1): forward->(2,1)=sq11 BLACK => blocked.
                           (r+c)=4 even => diag(-1,1)->(2,2)=sq12 EMPTY => 1 move.
                           diag(-1,-1)->(2,0)=sq10 BLACK => blocked.
          - sq 17 = (3,2): forward->(2,2)=sq12 EMPTY => 1 move.
                           (r+c)=5 odd => no diag => total 1.
          - sq 18 = (3,3): forward->(2,3)=sq13 WHITE => blocked.
                           (r+c)=6 even => diag(-1,-1)->(2,2)=sq12 EMPTY => 1 move.
                           diag(-1,1)->(2,4)=sq14 WHITE => blocked.
          - sq 19 = (3,4): forward->(2,4)=sq14 WHITE => blocked.
                           (r+c)=7 odd => no diag => 0 moves.
        Row 2 white pieces:
          - sq 13 = (2,3): forward(dr=-1)->(1,3)=sq8 BLACK => blocked.
                           (r+c)=5 odd => no diag => 0 moves.
          - sq 14 = (2,4): forward->(1,4)=sq9 BLACK => blocked.
                           (r+c)=6 even => diag(-1,-1)->(1,3)=sq8 BLACK => blocked.
                           diag(-1,1)->(1,5) off-board => 0 moves.

        Total white moves: sq16->12, sq17->12, sq18->12 = 3 moves.
        """
        board = initial_board()
        moves = get_simple_moves(board, WHITE)
        assert len(moves) == 3

        center = rc_to_sq(2, 2)
        for src, dst in moves:
            assert dst == center

        sources = sorted(src for src, _dst in moves)
        assert sources == [rc_to_sq(3, 1), rc_to_sq(3, 2), rc_to_sq(3, 3)]


# ===================================================================
# Mixed scenarios
# ===================================================================

class TestMixedScenarios:
    """Combinations of men and kings on the board."""

    def test_man_and_king_same_player(self):
        """Both a man and king for the same player produce their own moves."""
        man_sq = rc_to_sq(1, 1)
        king_sq = rc_to_sq(3, 3)
        board = board_with({man_sq: BLACK_MAN, king_sq: BLACK_KING})
        moves = get_simple_moves(board, BLACK)

        man_moves = [(s, d) for s, d in moves if s == man_sq]
        king_moves = [(s, d) for s, d in moves if s == king_sq]

        assert len(man_moves) > 0
        assert len(king_moves) > 0

    def test_no_moves_when_no_pieces(self):
        """Player with no pieces on the board has no moves."""
        board = empty_board()
        assert get_simple_moves(board, BLACK) == []
        assert get_simple_moves(board, WHITE) == []

    def test_king_does_not_belong_to_wrong_player(self):
        """get_simple_moves for BLACK should ignore WHITE_KING."""
        sq = rc_to_sq(2, 2)
        board = board_with({sq: WHITE_KING})
        moves = get_simple_moves(board, BLACK)
        assert moves == []

    def test_multiple_kings(self):
        """Multiple kings each contribute their own moves."""
        k1 = rc_to_sq(0, 0)
        k2 = rc_to_sq(4, 4)
        board = board_with({k1: BLACK_KING, k2: BLACK_KING})
        moves = get_simple_moves(board, BLACK)

        sources = {s for s, _d in moves}
        assert k1 in sources
        assert k2 in sources

    def test_king_and_man_interact(self):
        """A man blocks a king's ray.

        King at (0,0), black man at (3,0).  King on (1,0) ray reaches
        (1,0) and (2,0) but not (3,0) or (4,0).
        """
        king_sq = rc_to_sq(0, 0)
        man_sq = rc_to_sq(3, 0)
        board = board_with({king_sq: BLACK_KING, man_sq: BLACK_MAN})
        moves = get_simple_moves(board, BLACK)

        king_dsts = {d for s, d in moves if s == king_sq}
        assert rc_to_sq(1, 0) in king_dsts
        assert rc_to_sq(2, 0) in king_dsts
        assert rc_to_sq(3, 0) not in king_dsts
        assert rc_to_sq(4, 0) not in king_dsts

        # The man at (3,0) should also have its own moves
        man_dsts = {d for s, d in moves if s == man_sq}
        assert rc_to_sq(4, 0) in man_dsts  # forward move


# ===================================================================
# Regression / sanity
# ===================================================================

class TestSanity:
    """General sanity checks on move generation."""

    def test_no_self_moves(self):
        """No move should have source == destination."""
        board = initial_board()
        for player in (BLACK, WHITE):
            moves = get_simple_moves(board, player)
            for src, dst in moves:
                assert src != dst

    def test_all_destinations_empty(self):
        """Every destination of a simple move must be EMPTY."""
        board = initial_board()
        for player in (BLACK, WHITE):
            moves = get_simple_moves(board, player)
            for _src, dst in moves:
                assert board[dst] == EMPTY, (
                    f"Move to non-empty square {dst} (contents={board[dst]})"
                )

    def test_all_sources_are_player_pieces(self):
        """Every source of a move must belong to the current player."""
        board = initial_board()
        for player in (BLACK, WHITE):
            moves = get_simple_moves(board, player)
            for src, _dst in moves:
                assert is_player_piece(board, src, player)

    def test_move_destinations_in_adjacency(self):
        """Every (src, dst) must correspond to a valid adjacency ray entry."""
        board = initial_board()
        for player in (BLACK, WHITE):
            moves = get_simple_moves(board, player)
            for src, dst in moves:
                # dst must appear somewhere in ADJACENCY[src]
                all_reachable = set()
                for ray in ADJACENCY[src].values():
                    all_reachable.update(ray)
                assert dst in all_reachable, (
                    f"Move ({src}->{dst}) not in adjacency graph"
                )


# ===================================================================
# Additional edge cases
# ===================================================================

class TestAdditionalMoveEdges:
    """Extra checks for ownership filtering and partial blocking."""

    def test_get_player_pieces_returns_sorted_indices(self):
        board = board_with({
            rc_to_sq(4, 4): BLACK_KING,
            rc_to_sq(0, 0): BLACK_MAN,
            rc_to_sq(2, 2): BLACK_MAN,
        })
        assert get_player_pieces(board, BLACK) == [0, 12, 24]

    def test_get_simple_moves_ignores_enemy_men(self):
        board = board_with({rc_to_sq(2, 2): WHITE_MAN})
        assert get_simple_moves(board, BLACK) == []

    def test_king_blocked_on_one_ray_still_moves_on_others(self):
        king_sq = rc_to_sq(2, 2)
        blocker_sq = rc_to_sq(3, 2)
        board = board_with({king_sq: BLACK_KING, blocker_sq: BLACK_MAN})

        moves = get_king_simple_moves(board, king_sq)
        dsts = {dst for _src, dst in moves}

        assert rc_to_sq(3, 2) not in dsts
        assert rc_to_sq(4, 2) not in dsts
        assert rc_to_sq(2, 3) in dsts
        assert rc_to_sq(1, 2) in dsts

    def test_man_with_blocked_forward_can_still_use_open_diagonals(self):
        sq = rc_to_sq(2, 2)
        forward_blocker = rc_to_sq(3, 2)
        board = board_with({
            sq: BLACK_MAN,
            forward_blocker: WHITE_MAN,
        })

        moves = get_man_simple_moves(board, sq, BLACK)
        assert set(moves) == {
            (sq, rc_to_sq(3, 1)),
            (sq, rc_to_sq(3, 3)),
        }
