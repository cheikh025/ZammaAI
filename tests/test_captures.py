"""
Tests for khreibga.captures – capture logic, majority rule, action mask.
"""

from __future__ import annotations

import pytest

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
    rc_to_sq,
    sq_to_rc,
)
from khreibga.captures import (
    apply_majority_rule,
    execute_hop,
    find_all_capture_chains,
    find_piece_capture_chains,
    find_king_captures,
    find_man_captures,
    get_action_mask,
    get_capture_first_hops,
    get_piece_capture_first_hops,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def empty_board() -> list[int]:
    """Return a completely empty 5x5 board."""
    return [EMPTY] * NUM_SQUARES


# ---------------------------------------------------------------------------
# Man single capture
# ---------------------------------------------------------------------------

class TestManSingleCapture:
    """Black man at 12 (2,2), white man at 7 (1,2), sq 2 (0,2) empty.
    Capture 12 -> 2 should be found."""

    def test_single_capture_found(self):
        board = empty_board()
        board[12] = BLACK_MAN  # (2,2)
        board[7] = WHITE_MAN   # (1,2)
        # sq 2 at (0,2) is already EMPTY.

        chains = find_man_captures(board, 12, BLACK)
        assert len(chains) >= 1

        # Extract (src, dst) pairs – there should be a chain starting 12->2.
        first_hops = {c[0] for c in chains}
        assert (12, 2) in first_hops

    def test_single_capture_chain_length(self):
        board = empty_board()
        board[12] = BLACK_MAN
        board[7] = WHITE_MAN

        chains = find_man_captures(board, 12, BLACK)
        # Only one capture possible so every chain has length 1.
        for chain in chains:
            assert len(chain) == 1


# ---------------------------------------------------------------------------
# Man chain (multi-step) capture
# ---------------------------------------------------------------------------

class TestManChainCapture:
    """Set up a zigzag so a man can capture 2+ pieces in sequence.

    Board layout (relevant squares):
      Black man at (4,0) = sq 20
      White man at (3,1) = sq 16   (adjacent diagonal, (4+0) even -> diag ok at 20)
      Landing at (2,2)  = sq 12   (empty)
      White man at (1,2) = sq 7    (orthogonal neighbour of 12)
      Landing at (0,2)  = sq 2    (empty)

    Chain: 20 -> 12 (captures 16), then 12 -> 2 (captures 7).
    """

    def test_chain_of_two(self):
        board = empty_board()
        board[20] = BLACK_MAN  # (4,0) – (4+0)=4 even, diags available
        board[16] = WHITE_MAN  # (3,1)
        board[7] = WHITE_MAN   # (1,2)

        chains = find_man_captures(board, 20, BLACK)

        # We expect at least one chain of length 2.
        max_len = max(len(c) for c in chains)
        assert max_len == 2

        # The two-hop chain should be 20->12 then 12->2.
        two_chains = [c for c in chains if len(c) == 2]
        assert any(
            c == [(20, 12), (12, 2)] for c in two_chains
        ), f"Expected [(20,12),(12,2)] in {two_chains}"


# ---------------------------------------------------------------------------
# King long leap
# ---------------------------------------------------------------------------

class TestKingLongLeap:
    """King at (0,0)=sq0, enemy at (2,2)=sq12.
    King should be able to land on (3,3)=sq18 or (4,4)=sq24."""

    def test_king_can_land_beyond(self):
        board = empty_board()
        board[0] = BLACK_KING   # (0,0)
        board[12] = WHITE_MAN   # (2,2)

        chains = find_king_captures(board, 0, BLACK)
        assert len(chains) >= 1

        landing_squares = {c[0][1] for c in chains}
        # Should include 18 = (3,3) and 24 = (4,4).
        assert 18 in landing_squares, f"Expected 18 in {landing_squares}"
        assert 24 in landing_squares, f"Expected 24 in {landing_squares}"

    def test_king_long_leap_chain_lengths(self):
        board = empty_board()
        board[0] = BLACK_KING
        board[12] = WHITE_MAN

        chains = find_king_captures(board, 0, BLACK)
        # Each chain captures exactly one piece (the one at 12).
        for chain in chains:
            assert len(chain) == 1


# ---------------------------------------------------------------------------
# King long leap blocked by friendly piece
# ---------------------------------------------------------------------------

class TestKingLongLeapBlocked:
    """King at (0,0)=sq0, enemy at (2,2)=sq12, friendly at (4,4)=sq24.
    King can land on (3,3)=sq18 but NOT on (4,4)=sq24."""

    def test_blocked_landing(self):
        board = empty_board()
        board[0] = BLACK_KING   # (0,0)
        board[12] = WHITE_MAN   # (2,2)
        board[24] = BLACK_MAN   # (4,4) – friendly blocker

        chains = find_king_captures(board, 0, BLACK)
        assert len(chains) >= 1

        landing_squares = {c[0][1] for c in chains}
        assert 18 in landing_squares, f"Expected 18 in {landing_squares}"
        assert 24 not in landing_squares, f"24 should be blocked, got {landing_squares}"


# ---------------------------------------------------------------------------
# Majority rule
# ---------------------------------------------------------------------------

class TestMajorityRule:
    """One piece can capture 1, another can capture 2.
    Only the 2-capture chain's first hop should appear in the mask."""

    def test_majority_filtering(self):
        """Piece 1 (man at sq 20) can chain-capture 2 enemies (zigzag).
        Piece 2 (man at sq 4) can capture only 1 enemy.

        Piece 1 layout:
          Man at (4,0)=20, enemy at (3,1)=16, landing (2,2)=12.
          Then from 12, enemy at (1,2)=7, landing (0,2)=2.
          Chain: 20->12->2, length 2.

        Piece 2 layout:
          Man at (0,4)=4, enemy at (1,4)=9, landing (2,4)=14.
          From (2,4)=14, parity (2+4)=6 even.  No enemies adjacent.
          Chain: 4->14, length 1.
        """
        board = empty_board()

        # Piece 1: zigzag double capture
        board[20] = BLACK_MAN  # (4,0)
        board[16] = WHITE_MAN  # (3,1) – first enemy
        board[7] = WHITE_MAN   # (1,2) – second enemy

        # Piece 2: single capture along column 4
        board[4] = BLACK_MAN   # (0,4)
        board[9] = WHITE_MAN   # (1,4) – only enemy for piece 2
        # Landing at (2,4)=14.  No enemies near 14.

        chains = find_all_capture_chains(board, BLACK)
        best = apply_majority_rule(chains)

        # The best chain(s) should have length 2.
        for c in best:
            assert len(c) == 2

        # The first-hops for the mask should include (20, 12).
        first_hops = get_capture_first_hops(board, BLACK)
        first_hop_set = set(first_hops)

        assert (20, 12) in first_hop_set, f"Expected (20,12) in {first_hop_set}"
        # The single-capture chain starting from 4 should be excluded.
        assert all(
            src != 4 for src, _ in first_hops
        ), f"Piece at 4 should not appear: {first_hops}"

    def test_majority_tie_keeps_all_best_first_hops(self):
        """If multiple sequences tie for max captures, all tied first hops
        remain legal.

        Board is chosen so BLACK has two distinct 2-capture sequences:
          - 0 -> 12 -> 4
          - 2 -> 4 -> 12
        and no longer chain exists.
        """
        board = [
            BLACK_MAN, BLACK_MAN, BLACK_MAN, WHITE_MAN, EMPTY,
            EMPTY, WHITE_MAN, EMPTY, WHITE_MAN, EMPTY,
            WHITE_MAN, WHITE_MAN, EMPTY, EMPTY, EMPTY,
            EMPTY, EMPTY, EMPTY, EMPTY, EMPTY,
            EMPTY, BLACK_MAN, EMPTY, WHITE_MAN, EMPTY,
        ]

        chains = find_all_capture_chains(board, BLACK)
        best = apply_majority_rule(chains)
        assert best, "Expected capture chains"
        assert all(len(c) == 2 for c in best), f"Expected only len-2 chains: {best}"

        first_hops = set(get_capture_first_hops(board, BLACK))
        assert (0, 12) in first_hops
        assert (2, 4) in first_hops

        mask = get_action_mask(board, BLACK)
        assert mask[0 * NUM_SQUARES + 12] == 1
        assert mask[2 * NUM_SQUARES + 4] == 1


# ---------------------------------------------------------------------------
# Immediate removal opens a path
# ---------------------------------------------------------------------------

class TestImmediateRemoval:
    """
    Scenario: A captured piece is removed instantly, opening a square
    for re-crossing.

    Layout (orthogonal captures along column 2):
      Black man at (4,2) = sq 22
      White man at (3,2) = sq 17   -> captured first, landing (2,2)=12
      White man at (1,2) = sq  7   -> captured second, landing (0,2)=2

    The man at 22 jumps to 12 (over 17 which is removed), then from 12
    jumps to 2 (over 7 which is removed).  Square 17 was cleared by
    immediate removal allowing the path to be valid.  (In this specific
    case the man never re-crosses 17, but the removal is essential to
    unblock further chains in other setups.)

    We additionally verify that the capture of sq 17 is no longer on the
    board after executing the first hop.
    """

    def test_immediate_removal_chain(self):
        board = empty_board()
        board[22] = BLACK_MAN  # (4,2)
        board[17] = WHITE_MAN  # (3,2)
        board[7] = WHITE_MAN   # (1,2)

        chains = find_man_captures(board, 22, BLACK)
        max_len = max(len(c) for c in chains)
        assert max_len == 2, f"Expected chain of 2, got {max_len}"

        # Verify that first hop removes enemy immediately.
        new_board, cap = execute_hop(board, 22, 12, BLACK)
        assert cap == 17
        assert new_board[17] == EMPTY
        assert new_board[12] == BLACK_MAN
        assert new_board[22] == EMPTY

    def test_landing_on_captured_square(self):
        """Verify that a man can land on a square that was vacated by a
        prior capture in the same chain (immediate-removal rule).

        We call find_man_captures with a pre-populated captured set to
        simulate a mid-chain position where a previously-captured square
        is available as a landing target.

        Setup: man at (2,2)=12, enemy at (1,2)=7.  Square (0,2)=2 holds
        a white piece on the real board, but we pass captured={2} to
        indicate it was already captured earlier.  The man should still
        be able to jump over 7 and land on 2 (treated as empty).
        """
        board = empty_board()
        board[12] = BLACK_MAN   # (2,2)
        board[7] = WHITE_MAN    # (1,2)
        board[2] = WHITE_MAN    # (0,2) -- physically present but "captured"

        # With captured={2}, sq 2 should be treated as empty for landing.
        chains = find_man_captures(board, 12, BLACK, captured={2})
        hops = {c[0] for c in chains}
        assert (12, 2) in hops, (
            f"Should be able to land on sq 2 (in captured set): {hops}"
        )

    def test_double_capture_along_column(self):
        """Two enemies along the same column: capturing the first makes the
        second capture reachable.

        Man at (2,2)=12, enemies at (3,2)=17 and (1,2)=7.
        Chain: 12->22 (captures 17), then 22 can't continue (no more enemies).
        OR:    12->2  (captures 7),  then 2 can't continue.
        No two-hop chain is possible here because each direction only has
        one enemy.  But we verify that both single captures are found.
        """
        board = empty_board()
        board[12] = BLACK_MAN   # (2,2)
        board[7] = WHITE_MAN    # (1,2)
        board[17] = WHITE_MAN   # (3,2)

        chains = find_man_captures(board, 12, BLACK)
        # There should be captures in both directions.
        hops = {c[0] for c in chains}
        assert (12, 2) in hops, f"Expected (12,2) in {hops}"
        assert (12, 22) in hops, f"Expected (12,22) in {hops}"

    def test_king_chain_can_recross_origin_line(self):
        """Immediate removal can open a line so a king crosses a previously
        occupied square later in the same sequence.

        Sequence checked here:
          12 -> 22 (captures 17), then 22 -> 2 (captures 7)
        The second hop path passes through square 12 (the chain origin).
        """
        board = empty_board()
        board[12] = BLACK_KING
        board[17] = WHITE_MAN
        board[7] = WHITE_MAN

        chains = find_king_captures(board, 12, BLACK)
        assert any(c[:2] == [(12, 22), (22, 2)] for c in chains), (
            f"Expected recross chain [(12,22),(22,2)] in {chains}"
        )

        # Piece-local continuation should preserve only the active-piece hops.
        board_after_first, _ = execute_hop(board, 12, 22, BLACK)
        piece_chains = find_piece_capture_chains(board_after_first, BLACK, 22)
        assert piece_chains, "Expected continuation chains for active piece"
        piece_hops = set(get_piece_capture_first_hops(board_after_first, BLACK, 22))
        assert (22, 2) in piece_hops


# ---------------------------------------------------------------------------
# Man backward capture
# ---------------------------------------------------------------------------

class TestManBackwardCapture:
    """Men can capture in any direction including backward.

    Black man at (0,2)=sq2.  White man at (1,2)=sq7.  Sq (2,2)=12 empty.
    This is a *forward* capture for Black (row increases).

    For backward: Black man at (2,2)=sq12.  White man at (1,2)=sq7.
    Landing at (0,2)=sq2.  Row decreases -> backward for Black.
    """

    def test_backward_capture(self):
        board = empty_board()
        board[12] = BLACK_MAN  # (2,2)
        board[7] = WHITE_MAN   # (1,2)

        chains = find_man_captures(board, 12, BLACK)
        assert len(chains) >= 1

        # The capture should go from 12 to 2 (row decreasing = backward).
        first_hops = {c[0] for c in chains}
        assert (12, 2) in first_hops

    def test_white_backward_capture(self):
        """White man captures backward (row increases, which is backward
        for White who advances row-decreasing)."""
        board = empty_board()
        board[12] = WHITE_MAN  # (2,2)
        board[17] = BLACK_MAN  # (3,2)

        chains = find_man_captures(board, 12, WHITE)
        assert len(chains) >= 1

        first_hops = {c[0] for c in chains}
        assert (12, 22) in first_hops


# ---------------------------------------------------------------------------
# Action mask dimensions
# ---------------------------------------------------------------------------

class TestActionMask:
    """Action mask must always be length 625."""

    def test_mask_length_empty_board(self):
        board = empty_board()
        mask = get_action_mask(board, BLACK)
        assert len(mask) == 625

    def test_mask_length_with_captures(self):
        board = empty_board()
        board[12] = BLACK_MAN
        board[7] = WHITE_MAN
        mask = get_action_mask(board, BLACK)
        assert len(mask) == 625

    def test_mask_entries_are_binary(self):
        board = empty_board()
        board[12] = BLACK_MAN
        board[7] = WHITE_MAN
        mask = get_action_mask(board, BLACK)
        assert all(v in (0, 1) for v in mask)

    def test_mask_captures_present(self):
        """When a capture is available, its first hop should be in the mask."""
        board = empty_board()
        board[12] = BLACK_MAN
        board[7] = WHITE_MAN
        mask = get_action_mask(board, BLACK)

        idx = 12 * NUM_SQUARES + 2  # src=12, dst=2
        assert mask[idx] == 1, f"Expected mask[{idx}]=1"

    def test_mask_no_pieces(self):
        """Empty board should yield all-zero mask."""
        board = empty_board()
        mask = get_action_mask(board, BLACK)
        assert sum(mask) == 0


# ---------------------------------------------------------------------------
# execute_hop
# ---------------------------------------------------------------------------

class TestExecuteHop:
    """Test the atomic hop execution."""

    def test_simple_move(self):
        """Non-capturing move: no enemy between src and dst."""
        board = empty_board()
        board[12] = BLACK_MAN

        # Move to adjacent empty (2,2)->(3,2)=17.
        new_board, cap = execute_hop(board, 12, 17, BLACK)
        assert cap is None
        assert new_board[12] == EMPTY
        assert new_board[17] == BLACK_MAN

    def test_capture_move(self):
        """Capturing hop removes the enemy."""
        board = empty_board()
        board[12] = BLACK_MAN
        board[7] = WHITE_MAN

        new_board, cap = execute_hop(board, 12, 2, BLACK)
        assert cap == 7
        assert new_board[7] == EMPTY
        assert new_board[2] == BLACK_MAN
        assert new_board[12] == EMPTY

    def test_king_long_capture(self):
        """King captures over an enemy and lands far away."""
        board = empty_board()
        board[0] = BLACK_KING
        board[12] = WHITE_MAN

        new_board, cap = execute_hop(board, 0, 24, BLACK)
        assert cap == 12
        assert new_board[12] == EMPTY
        assert new_board[24] == BLACK_KING
        assert new_board[0] == EMPTY

    def test_original_board_unmodified(self):
        """execute_hop must not mutate the original board."""
        board = empty_board()
        board[12] = BLACK_MAN
        board[7] = WHITE_MAN
        original = list(board)

        execute_hop(board, 12, 2, BLACK)
        assert board == original


# ---------------------------------------------------------------------------
# apply_majority_rule
# ---------------------------------------------------------------------------

class TestApplyMajorityRule:
    def test_empty_chains(self):
        assert apply_majority_rule([]) == []

    def test_filters_shorter(self):
        chains = [
            [(0, 10)],                     # length 1
            [(5, 15), (15, 20)],           # length 2
            [(3, 13), (13, 23)],           # length 2
        ]
        best = apply_majority_rule(chains)
        assert len(best) == 2
        for c in best:
            assert len(c) == 2

    def test_all_same_length(self):
        chains = [
            [(0, 10)],
            [(5, 15)],
        ]
        best = apply_majority_rule(chains)
        assert len(best) == 2


# ---------------------------------------------------------------------------
# King multi-capture
# ---------------------------------------------------------------------------

class TestKingMultiCapture:
    """King captures multiple enemies in a chain."""

    def test_king_double_capture(self):
        """King at (0,0)=0, enemies at (2,2)=12 and (2,4)=14.

        After capturing 12 and landing at, say, (3,3)=18 or (4,4)=24,
        the king should be able to continue capturing 14 if a valid
        direction exists.

        Better setup: King at (0,0)=0, enemy at (2,0)=10, enemy at (4,2)=22.
        After 0->(3,0)=15 or (4,0)=20 capturing 10, then from 20 going
        direction (0,1) along row 4: 20->21->22->23->24.  But 22 has enemy
        and (4+0)=4 even so diagonals work at 20.  Actually, for orthogonal
        captures from 20 east: ray is [21,22,23,24].  But king capture needs
        enemy piece then empty beyond.  22 is enemy, so need 21 to be empty
        (it is) and landing at 23 or 24.

        Wait, the king slides along the ray until hitting the enemy. From 20
        east, ray = [21, 22, 23, 24].  21 is empty, 22 is enemy. Landing at
        23 or 24 (both empty).  This works!
        """
        board = empty_board()
        board[0] = BLACK_KING
        board[10] = WHITE_MAN  # (2,0)
        board[22] = WHITE_MAN  # (4,2)

        chains = find_king_captures(board, 0, BLACK)
        max_len = max(len(c) for c in chains) if chains else 0
        assert max_len == 2, f"Expected chain of 2, got {max_len}"


# ---------------------------------------------------------------------------
# Edge: no captures available
# ---------------------------------------------------------------------------

class TestNoCapturesAvailable:
    def test_no_captures_returns_empty(self):
        board = empty_board()
        board[12] = BLACK_MAN
        chains = find_man_captures(board, 12, BLACK)
        assert chains == []

    def test_find_all_no_captures(self):
        board = empty_board()
        board[12] = BLACK_MAN
        chains = find_all_capture_chains(board, BLACK)
        assert chains == []

    def test_first_hops_empty(self):
        board = empty_board()
        board[12] = BLACK_MAN
        hops = get_capture_first_hops(board, BLACK)
        assert hops == []


# ---------------------------------------------------------------------------
# Diagonal vs orthogonal capture availability
# ---------------------------------------------------------------------------

class TestDiagonalCapture:
    """Diagonal captures are only possible from squares with even (r+c)."""

    def test_diagonal_capture_even_parity(self):
        """Black man at (0,0)=0 ((0+0)=0 even -> diags available).
        Enemy at (1,1)=6.  Landing at (2,2)=12."""
        board = empty_board()
        board[0] = BLACK_MAN
        board[6] = WHITE_MAN

        chains = find_man_captures(board, 0, BLACK)
        hops = {c[0] for c in chains}
        assert (0, 12) in hops

    def test_no_diagonal_odd_parity(self):
        """Black man at (0,1)=1 ((0+1)=1 odd -> NO diags).
        Placing enemy at (1,2)=7 is diagonal and should NOT be capturable
        from sq 1 because the diagonal direction doesn't exist."""
        board = empty_board()
        board[1] = BLACK_MAN
        board[7] = WHITE_MAN
        # If diags were available, man could jump 1->7->13 direction (1,1).
        # But sq 1 has odd parity so no diagonal adjacency.

        chains = find_man_captures(board, 1, BLACK)
        # Check that no chain jumps diagonally to 13.
        diag_hops = [c for c in chains if c[0] == (1, 13)]
        assert diag_hops == [], f"Diagonal capture should be impossible from odd-parity square: {diag_hops}"
