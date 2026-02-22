"""
Tests for khreibga.game -- GameState engine, turn management, promotion,
terminal conditions, observation tensor, and cloning.

Since the captures module (Agent 3) may not be available yet, we mock
its functions where needed, providing controlled capture behavior.
"""

from __future__ import annotations

import copy
from unittest.mock import patch, MagicMock

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
    initial_board,
    rc_to_sq,
    sq_to_rc,
)
from khreibga.move_gen import get_simple_moves


# ---------------------------------------------------------------------------
# Helper: build action mask from list of (src, dst) pairs
# ---------------------------------------------------------------------------

def _mask_from_moves(moves):
    """Build a 625-element action mask from a list of (src, dst) tuples."""
    mask = [0] * (NUM_SQUARES * NUM_SQUARES)
    for s, d in moves:
        mask[s * NUM_SQUARES + d] = 1
    return mask


def _action(src, dst):
    """Encode an (src, dst) pair as an action int."""
    return src * NUM_SQUARES + dst


# ---------------------------------------------------------------------------
# Mock captures module for tests
# ---------------------------------------------------------------------------

def _no_captures_get_action_mask(board, player):
    """Action mask that returns only simple moves (no captures)."""
    moves = get_simple_moves(board, player)
    return _mask_from_moves(moves)


def _no_captures_first_hops(board, player):
    """No captures available."""
    return []


def _no_captures_execute_hop(board, src, dst, player):
    """Should not be called when there are no captures."""
    raise RuntimeError("execute_hop called but no captures expected")


def _no_captures_find_all(board, player):
    """No capture chains."""
    return []


# We'll patch at the module level where game.py imports from
_CAPTURE_PATCHES = {
    "khreibga.game.captures_get_action_mask": _no_captures_get_action_mask,
    "khreibga.game.get_capture_first_hops": _no_captures_first_hops,
    "khreibga.game.execute_hop": _no_captures_execute_hop,
    "khreibga.game.find_all_capture_chains": _no_captures_find_all,
}


@pytest.fixture
def no_capture_patches():
    """Fixture that patches all capture-related imports to return no captures."""
    with patch("khreibga.game.captures_get_action_mask", side_effect=_no_captures_get_action_mask), \
         patch("khreibga.game.get_capture_first_hops", side_effect=_no_captures_first_hops), \
         patch("khreibga.game.execute_hop", side_effect=_no_captures_execute_hop), \
         patch("khreibga.game.find_all_capture_chains", side_effect=_no_captures_find_all):
        from khreibga.game import GameState
        yield GameState


@pytest.fixture
def game(no_capture_patches):
    """Return a fresh GameState with captures mocked out."""
    return no_capture_patches()


# ===================================================================
# Initial state
# ===================================================================

class TestInitialState:
    """Verify the game starts in the correct initial configuration."""

    def test_initial_board_matches(self, game):
        assert game.board == initial_board()

    def test_black_moves_first(self, game):
        assert game.current_player == BLACK

    def test_not_done(self, game):
        assert game.done is False

    def test_winner_is_none(self, game):
        assert game.winner is None

    def test_chain_piece_is_none(self, game):
        assert game.chain_piece is None

    def test_half_move_clock_zero(self, game):
        assert game.half_move_clock == 0

    def test_move_count_zero(self, game):
        assert game.move_count == 0

    def test_history_has_initial_position(self, game):
        assert len(game.history) == 1
        for count in game.history.values():
            assert count == 1


# ===================================================================
# Simple move execution
# ===================================================================

class TestSimpleMove:
    """Test non-capture moves."""

    def test_simple_move_changes_board(self, no_capture_patches):
        game = no_capture_patches()
        # Use a custom board where WHITE has moves after BLACK's turn.
        # BLACK_MAN at sq 2 (0,2) moves forward to sq 7 (1,2).
        # WHITE_MAN at sq 22 (4,2) can still move forward to sq 17 (3,2).
        game.board = [EMPTY] * NUM_SQUARES
        game.board[2] = BLACK_MAN   # (0,2)
        game.board[22] = WHITE_MAN  # (4,2) -- has forward moves
        from khreibga.game import compute_zobrist_hash
        game.current_hash = compute_zobrist_hash(game.board, game.current_player)
        game.history = {game.current_hash: 1}

        src = 2   # (0, 2) BLACK_MAN
        dst = 7   # (1, 2) EMPTY
        action = _action(src, dst)

        # Verify this is a legal move
        mask = game.get_action_mask()
        assert mask[action] == 1, f"Action {src}->{dst} should be legal"

        obs, new_mask, reward, done = game.step(action)

        assert game.board[src] == EMPTY
        assert game.board[dst] == BLACK_MAN
        assert done is False

    def test_simple_move_switches_player(self, no_capture_patches):
        game = no_capture_patches()
        src = 7
        dst = 12
        action = _action(src, dst)
        game.step(action)
        assert game.current_player == WHITE

    def test_simple_move_increments_move_count(self, no_capture_patches):
        game = no_capture_patches()
        src = 7
        dst = 12
        action = _action(src, dst)
        game.step(action)
        assert game.move_count == 1

    def test_man_move_resets_half_move_clock(self, no_capture_patches):
        game = no_capture_patches()
        game.half_move_clock = 10  # artificially set
        src = 7
        dst = 12
        action = _action(src, dst)
        game.step(action)
        assert game.half_move_clock == 0

    def test_king_move_increments_half_move_clock(self, no_capture_patches):
        game = no_capture_patches()
        # Set up a board with a black king
        game.board = [EMPTY] * NUM_SQUARES
        game.board[0] = BLACK_KING
        game.board[24] = WHITE_MAN  # opponent needs a piece
        game.half_move_clock = 5

        action = _action(0, 1)  # King at (0,0) moves to (0,1)
        mask = game.get_action_mask()
        assert mask[action] == 1

        game.step(action)
        assert game.half_move_clock == 6

    def test_illegal_action_raises(self, no_capture_patches):
        game = no_capture_patches()
        # Action 0 is src=0, dst=0 which is always illegal
        with pytest.raises(ValueError, match="Illegal action"):
            game.step(0)


# ===================================================================
# Capture execution
# ===================================================================

class TestCapture:
    """Test capture hop execution."""

    def test_capture_removes_enemy_piece(self):
        """Set up a simple capture scenario and verify piece removal."""
        # Board: Black man at sq 2, White man at sq 7, landing at sq 12
        # (0,2) -> capture (1,2) -> land (2,2)
        # Landing at sq 12 (row 2) avoids accidental promotion (row 4).
        board = [EMPTY] * NUM_SQUARES
        board[2] = BLACK_MAN    # (0,2)
        board[7] = WHITE_MAN    # (1,2) - enemy to capture
        # Need another white piece so game doesn't end on elimination
        board[24] = WHITE_MAN

        captured_sq = 7

        def mock_get_action_mask(b, player):
            if player == BLACK:
                return _mask_from_moves([(2, 12)])
            return _mask_from_moves([])

        def mock_first_hops(b, player):
            if player == BLACK and b[2] == BLACK_MAN:
                return [(2, 12)]
            return []

        def mock_execute_hop(b, src, dst, player):
            result = b[:]
            result[src] = EMPTY
            result[captured_sq] = EMPTY
            result[dst] = b[src]
            return result, captured_sq

        def mock_find_all(b, player):
            return []

        with patch("khreibga.game.captures_get_action_mask", side_effect=mock_get_action_mask), \
             patch("khreibga.game.get_capture_first_hops", side_effect=mock_first_hops), \
             patch("khreibga.game.execute_hop", side_effect=mock_execute_hop), \
             patch("khreibga.game.find_all_capture_chains", side_effect=mock_find_all):
            from khreibga.game import GameState
            gs = GameState.__new__(GameState)
            gs.board = board[:]
            gs.current_player = BLACK
            gs.chain_piece = None
            gs.half_move_clock = 10
            gs.move_count = 0
            gs.history = {}
            gs.done = False
            gs.winner = None
            from khreibga.game import compute_zobrist_hash
            gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
            gs.history[gs.current_hash] = 1

            obs, mask, reward, done = gs.step(_action(2, 12))

            assert gs.board[2] == EMPTY   # source cleared
            assert gs.board[7] == EMPTY   # captured piece removed
            assert gs.board[12] == BLACK_MAN  # piece landed (row 2, no promo)
            assert gs.half_move_clock == 0  # reset on capture

    def test_capture_resets_half_move_clock(self):
        """Verify half_move_clock resets to 0 after a capture."""
        board = [EMPTY] * NUM_SQUARES
        board[6] = BLACK_MAN    # (1,1)
        board[12] = WHITE_MAN   # (2,2) enemy
        board[24] = WHITE_MAN   # keep alive

        def mock_get_action_mask(b, player):
            if player == BLACK and b[6] == BLACK_MAN:
                return _mask_from_moves([(6, 18)])
            return _mask_from_moves([])

        def mock_first_hops(b, player):
            if player == BLACK and b[6] == BLACK_MAN:
                return [(6, 18)]
            return []

        def mock_execute_hop(b, src, dst, player):
            result = b[:]
            result[6] = EMPTY
            result[12] = EMPTY
            result[18] = BLACK_MAN
            return result, 12

        with patch("khreibga.game.captures_get_action_mask", side_effect=mock_get_action_mask), \
             patch("khreibga.game.get_capture_first_hops", side_effect=mock_first_hops), \
             patch("khreibga.game.execute_hop", side_effect=mock_execute_hop), \
             patch("khreibga.game.find_all_capture_chains", return_value=[]):
            from khreibga.game import GameState, compute_zobrist_hash
            gs = GameState.__new__(GameState)
            gs.board = board[:]
            gs.current_player = BLACK
            gs.chain_piece = None
            gs.half_move_clock = 30
            gs.move_count = 0
            gs.history = {}
            gs.done = False
            gs.winner = None
            gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
            gs.history[gs.current_hash] = 1

            gs.step(_action(6, 18))
            assert gs.half_move_clock == 0


# ===================================================================
# Multi-step chain
# ===================================================================

class TestMultiStepChain:
    """Test double-capture chain: after first hop, player does NOT switch.
    After second hop, player switches."""

    def test_chain_no_switch_then_switch(self):
        # Setup: Black man at sq 2 (0,2)
        #        White man at sq 7 (1,2) and sq 17 (3,2)
        #        Empty at sq 12 (2,2) and sq 22 (4,2)
        # Chain: 2 -> 12 (capture 7), then 12 -> 22 (capture 17)
        board = [EMPTY] * NUM_SQUARES
        board[2] = BLACK_MAN     # (0,2)
        board[7] = WHITE_MAN     # (1,2) - first enemy
        board[17] = WHITE_MAN    # (3,2) - second enemy
        board[24] = WHITE_MAN    # keep white alive after double capture

        call_count = [0]

        def mock_get_action_mask(b, player):
            if player == BLACK:
                if b[2] == BLACK_MAN:
                    return _mask_from_moves([(2, 12)])
                elif b[12] == BLACK_MAN:
                    return _mask_from_moves([(12, 22)])
            return _mask_from_moves([])

        def mock_first_hops(b, player):
            if player == BLACK:
                if b[2] == BLACK_MAN:
                    return [(2, 12)]
                if b[12] == BLACK_MAN and b[17] == WHITE_MAN:
                    return [(12, 22)]
            return []

        def mock_execute_hop(b, src, dst, player):
            result = b[:]
            if src == 2 and dst == 12:
                result[2] = EMPTY
                result[7] = EMPTY
                result[12] = BLACK_MAN
                return result, 7
            elif src == 12 and dst == 22:
                result[12] = EMPTY
                result[17] = EMPTY
                result[22] = BLACK_MAN
                return result, 17
            raise RuntimeError(f"Unexpected hop {src}->{dst}")

        with patch("khreibga.game.captures_get_action_mask", side_effect=mock_get_action_mask), \
             patch("khreibga.game.get_capture_first_hops", side_effect=mock_first_hops), \
             patch("khreibga.game.execute_hop", side_effect=mock_execute_hop), \
             patch("khreibga.game.find_all_capture_chains", return_value=[]):
            from khreibga.game import GameState, compute_zobrist_hash

            gs = GameState.__new__(GameState)
            gs.board = board[:]
            gs.current_player = BLACK
            gs.chain_piece = None
            gs.half_move_clock = 0
            gs.move_count = 0
            gs.history = {}
            gs.done = False
            gs.winner = None
            gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
            gs.history[gs.current_hash] = 1

            # First hop: 2 -> 12
            obs, mask, reward, done = gs.step(_action(2, 12))
            assert gs.current_player == BLACK, "Player should NOT switch mid-chain"
            assert gs.chain_piece == 12, "chain_piece should be set to landing square"

            # Second hop: 12 -> 22
            obs, mask, reward, done = gs.step(_action(12, 22))
            assert gs.current_player == WHITE, "Player should switch after chain ends"
            assert gs.chain_piece is None, "chain_piece should be cleared"


class TestMultiStepChainIntegration:
    """Integration checks with real capture logic (no mocks)."""

    def test_active_chain_piece_continues_even_if_other_piece_has_longer_global_chain(self):
        """Regression for a bug where mid-chain continuation was derived from
        global majority hops instead of active-piece-local hops.
        """
        from khreibga.captures import (
            execute_hop,
            get_capture_first_hops,
            get_piece_capture_first_hops,
        )
        from khreibga.game import GameState, compute_zobrist_hash

        # Encoded board:
        #   BLACK: king at 0, man at 2, man at 7, man at 13
        #   WHITE: men at 3,8,18,20 and kings at 9,15,22
        board = [
            BLACK_KING, EMPTY, BLACK_MAN, WHITE_MAN, EMPTY,
            EMPTY, EMPTY, BLACK_MAN, WHITE_MAN, WHITE_KING,
            EMPTY, EMPTY, EMPTY, BLACK_MAN, EMPTY,
            WHITE_KING, EMPTY, EMPTY, WHITE_MAN, EMPTY,
            WHITE_MAN, EMPTY, WHITE_KING, EMPTY, EMPTY,
        ]

        gs = GameState.__new__(GameState)
        gs.board = board[:]
        gs.current_player = BLACK
        gs.chain_piece = None
        gs.half_move_clock = 0
        gs.move_count = 0
        gs.history = {}
        gs.done = False
        gs.winner = None
        gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
        gs.history[gs.current_hash] = 1

        first_action = _action(0, 24)
        mask = gs.get_action_mask()
        assert mask[first_action] == 1, "Expected 0->24 to be legal"

        # Show why piece-local continuation is required:
        # after 0->24, global majority first-hops come from another piece.
        board_after_first, _ = execute_hop(board, 0, 24, BLACK)
        global_hops = get_capture_first_hops(board_after_first, BLACK)
        assert global_hops, "Expected global capture hops"
        assert all(src != 24 for src, _ in global_hops), (
            f"Global hops unexpectedly include active piece: {global_hops}"
        )

        piece_hops = get_piece_capture_first_hops(board_after_first, BLACK, 24)
        assert piece_hops, "Active piece must still have continuation hops"
        assert all(src == 24 for src, _ in piece_hops)

        # Engine step should keep BLACK to move and lock chain_piece=24.
        gs.step(first_action)
        assert gs.current_player == BLACK, "Player must not switch mid-chain"
        assert gs.chain_piece == 24, "Active chain piece should remain locked"

        continuation_mask = gs.get_action_mask()
        assert any(
            continuation_mask[idx] == 1 and idx // NUM_SQUARES == 24
            for idx in range(NUM_SQUARES * NUM_SQUARES)
        ), "Continuation mask must contain moves from the active chain piece"


# ===================================================================
# Promotion
# ===================================================================

class TestPromotion:
    """Test man promotion to king (corners-only Khreibaga rule)."""

    def test_black_man_promotes_on_left_corner(self, no_capture_patches):
        GS = no_capture_patches
        gs = GS()
        # Black man at (3,0)=sq15 moves forward to corner (4,0)=sq20
        gs.board = [EMPTY] * NUM_SQUARES
        gs.board[15] = BLACK_MAN  # (3, 0)
        gs.board[0] = WHITE_MAN   # keep opponent alive
        from khreibga.game import compute_zobrist_hash
        gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
        gs.history = {gs.current_hash: 1}

        action = _action(15, 20)  # move to corner (4,0)
        mask = gs.get_action_mask()
        assert mask[action] == 1

        gs.step(action)
        assert gs.board[20] == BLACK_KING, "Black man should promote on left corner sq 20"

    def test_black_man_promotes_on_right_corner(self, no_capture_patches):
        GS = no_capture_patches
        gs = GS()
        # Black man at (3,4)=sq19 moves forward to corner (4,4)=sq24
        gs.board = [EMPTY] * NUM_SQUARES
        gs.board[19] = BLACK_MAN  # (3, 4)
        gs.board[0] = WHITE_MAN   # keep opponent alive
        from khreibga.game import compute_zobrist_hash
        gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
        gs.history = {gs.current_hash: 1}

        action = _action(19, 24)  # move to corner (4,4)
        mask = gs.get_action_mask()
        assert mask[action] == 1

        gs.step(action)
        assert gs.board[24] == BLACK_KING, "Black man should promote on right corner sq 24"

    def test_white_man_promotes_on_left_corner(self, no_capture_patches):
        GS = no_capture_patches
        gs = GS()
        # White man at (1,0)=sq5 moves forward (row decreases) to corner (0,0)=sq0
        gs.board = [EMPTY] * NUM_SQUARES
        gs.board[5] = WHITE_MAN   # (1, 0)
        gs.board[24] = BLACK_MAN  # keep opponent alive
        gs.current_player = WHITE
        from khreibga.game import compute_zobrist_hash
        gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
        gs.history = {gs.current_hash: 1}

        action = _action(5, 0)  # move to corner (0,0)
        mask = gs.get_action_mask()
        assert mask[action] == 1

        gs.step(action)
        assert gs.board[0] == WHITE_KING, "White man should promote on left corner sq 0"

    def test_white_man_promotes_on_right_corner(self, no_capture_patches):
        GS = no_capture_patches
        gs = GS()
        # White man at (1,4)=sq9 moves forward to corner (0,4)=sq4
        gs.board = [EMPTY] * NUM_SQUARES
        gs.board[9] = WHITE_MAN   # (1, 4)
        gs.board[20] = BLACK_MAN  # keep opponent alive
        gs.current_player = WHITE
        from khreibga.game import compute_zobrist_hash
        gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
        gs.history = {gs.current_hash: 1}

        action = _action(9, 4)  # move to corner (0,4)
        mask = gs.get_action_mask()
        assert mask[action] == 1

        gs.step(action)
        assert gs.board[4] == WHITE_KING, "White man should promote on right corner sq 4"

    def test_black_man_does_NOT_promote_on_middle_of_row4(self, no_capture_patches):
        """Dead zone: squares 21, 22, 23 on row 4 must NOT promote."""
        GS = no_capture_patches
        gs = GS()
        # Black man at (3,2)=sq17 moves to (4,2)=sq22 -- middle of row 4
        gs.board = [EMPTY] * NUM_SQUARES
        gs.board[17] = BLACK_MAN  # (3, 2)
        gs.board[0] = WHITE_MAN   # keep opponent alive
        from khreibga.game import compute_zobrist_hash
        gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
        gs.history = {gs.current_hash: 1}

        action = _action(17, 22)  # move to middle sq 22
        mask = gs.get_action_mask()
        assert mask[action] == 1

        gs.step(action)
        assert gs.board[22] == BLACK_MAN, "Black man must NOT promote on middle square 22"

    def test_white_man_does_NOT_promote_on_middle_of_row0(self, no_capture_patches):
        """Dead zone: squares 1, 2, 3 on row 0 must NOT promote."""
        GS = no_capture_patches
        gs = GS()
        # White man at (1,2)=sq7 moves to (0,2)=sq2 -- middle of row 0
        gs.board = [EMPTY] * NUM_SQUARES
        gs.board[7] = WHITE_MAN  # (1, 2)
        gs.board[24] = BLACK_MAN  # keep opponent alive
        gs.current_player = WHITE
        from khreibga.game import compute_zobrist_hash
        gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
        gs.history = {gs.current_hash: 1}

        action = _action(7, 2)  # move to middle sq 2
        mask = gs.get_action_mask()
        assert mask[action] == 1

        gs.step(action)
        assert gs.board[2] == WHITE_MAN, "White man must NOT promote on middle square 2"


class TestPromotionDeniedMidChain:
    """Test that a man reaching the promotion rank mid-chain does NOT promote."""

    def test_no_promotion_mid_chain(self):
        # Black man at sq 12 (2,2), white at sq 17 (3,2) and sq 21 (4,1)
        # Chain: 12 -> 22 (capture 17, lands on row 4 = promo rank)
        #        22 -> 10 or similar (continues capturing, leaves row 4)
        # After first hop it's on row 4 but must keep jumping, so no promotion.
        board = [EMPTY] * NUM_SQUARES
        board[12] = BLACK_MAN  # (2,2)
        board[17] = WHITE_MAN  # (3,2)
        # After landing on 22 (4,2), there's another white at (3,1) = sq 16
        # and empty at (2,0) = sq 10 for the second jump
        board[16] = WHITE_MAN  # (3,1) -- would need diagonal from (4,2)
        # Actually let's use a simpler geometry:
        # 12(2,2) captures 17(3,2) lands 22(4,2)
        # 22(4,2) captures 23(4,3) lands 24(4,4)? No, that's not a jump over.
        # Let's use: after landing at 22(4,2), capture via 21(4,1) to land at 20(4,0)?
        # No, captures jump OVER an adjacent enemy.
        # Better: 22(4,2) has diagonal to 18(3,3), which jumps over to land at...
        # Actually, let's just use a direct board to mock the behavior.
        board[18] = WHITE_MAN  # (3,3)
        # Diagonal from (4,2) -> over (3,3) -> land (2,4) = sq 14?
        # (4,2) has (r+c)=6 even, so has diagonals. Direction (-1,1) lands at (3,3)=18,
        # then beyond is (2,4)=14.
        board[24] = WHITE_MAN  # keep white alive

        def mock_first_hops(b, player):
            if player == BLACK:
                if b[12] == BLACK_MAN:
                    return [(12, 22)]
                if b[22] == BLACK_MAN and b[18] == WHITE_MAN:
                    return [(22, 14)]
            return []

        def mock_get_action_mask(b, player):
            if player == BLACK:
                if b[12] == BLACK_MAN:
                    return _mask_from_moves([(12, 22)])
                if b[22] == BLACK_MAN:
                    return _mask_from_moves([(22, 14)])
            return _mask_from_moves([])

        def mock_execute_hop(b, src, dst, player):
            result = b[:]
            if src == 12 and dst == 22:
                result[12] = EMPTY
                result[17] = EMPTY
                result[22] = BLACK_MAN
                return result, 17
            elif src == 22 and dst == 14:
                result[22] = EMPTY
                result[18] = EMPTY
                result[14] = BLACK_MAN
                return result, 18
            raise RuntimeError(f"Unexpected hop {src}->{dst}")

        with patch("khreibga.game.captures_get_action_mask", side_effect=mock_get_action_mask), \
             patch("khreibga.game.get_capture_first_hops", side_effect=mock_first_hops), \
             patch("khreibga.game.execute_hop", side_effect=mock_execute_hop), \
             patch("khreibga.game.find_all_capture_chains", return_value=[]):
            from khreibga.game import GameState, compute_zobrist_hash

            gs = GameState.__new__(GameState)
            gs.board = board[:]
            gs.current_player = BLACK
            gs.chain_piece = None
            gs.half_move_clock = 0
            gs.move_count = 0
            gs.history = {}
            gs.done = False
            gs.winner = None
            gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
            gs.history[gs.current_hash] = 1

            # First hop: 12 -> 22 (lands on row 4 = promotion rank)
            gs.step(_action(12, 22))

            # Should NOT have promoted because chain continues
            assert gs.board[22] == BLACK_MAN, \
                "Man should NOT promote mid-chain even on promotion rank"
            assert gs.chain_piece == 22
            assert gs.current_player == BLACK

            # Second hop: 22 -> 14 (leaves row 4, chain ends)
            gs.step(_action(22, 14))

            # Now at sq 14 (2,4) which is NOT the promotion rank, so still a man
            assert gs.board[14] == BLACK_MAN
            assert gs.chain_piece is None
            assert gs.current_player == WHITE


# ===================================================================
# Elimination win
# ===================================================================

class TestElimination:
    """Removing all of one player's pieces should end the game."""

    def test_elimination_via_capture(self):
        """Capture the last enemy piece -> winner declared."""
        board = [EMPTY] * NUM_SQUARES
        board[6] = BLACK_MAN    # (1,1)
        board[12] = WHITE_MAN   # (2,2) -- the only white piece

        def mock_first_hops(b, player):
            if player == BLACK and b[6] == BLACK_MAN and b[12] == WHITE_MAN:
                return [(6, 18)]
            return []

        def mock_get_action_mask(b, player):
            if player == BLACK and b[6] == BLACK_MAN:
                return _mask_from_moves([(6, 18)])
            return _mask_from_moves([])

        def mock_execute_hop(b, src, dst, player):
            result = b[:]
            result[6] = EMPTY
            result[12] = EMPTY
            result[18] = BLACK_MAN
            return result, 12

        with patch("khreibga.game.captures_get_action_mask", side_effect=mock_get_action_mask), \
             patch("khreibga.game.get_capture_first_hops", side_effect=mock_first_hops), \
             patch("khreibga.game.execute_hop", side_effect=mock_execute_hop), \
             patch("khreibga.game.find_all_capture_chains", return_value=[]):
            from khreibga.game import GameState, compute_zobrist_hash

            gs = GameState.__new__(GameState)
            gs.board = board[:]
            gs.current_player = BLACK
            gs.chain_piece = None
            gs.half_move_clock = 0
            gs.move_count = 0
            gs.history = {}
            gs.done = False
            gs.winner = None
            gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
            gs.history[gs.current_hash] = 1

            obs, mask, reward, done = gs.step(_action(6, 18))

            assert gs.done is True
            assert gs.winner == BLACK
            assert reward == 1.0


# ===================================================================
# Stalemate
# ===================================================================

class TestStalemate:
    """Current player has pieces but no legal moves -> that player loses."""

    def test_stalemate_loss(self, no_capture_patches):
        GS = no_capture_patches
        gs = GS()

        # Set up board where BLACK has a piece but cannot move
        # Black man at (0,0) = sq 0, surrounded by own pieces or edge
        # But men can only move forward. Put black man at sq 20 (4,0) -- row 4
        # Black men at row 4 can only move forward = row 5 which doesn't exist
        # So a black man at row 4 with no captures has no moves
        gs.board = [EMPTY] * NUM_SQUARES
        gs.board[20] = BLACK_MAN  # (4,0) - can't move forward (off board)
        gs.board[0] = WHITE_MAN   # opponent has a piece

        gs.current_player = BLACK
        from khreibga.game import compute_zobrist_hash
        gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
        gs.history = {gs.current_hash: 1}

        # Trigger terminal check
        gs._check_terminal()

        assert gs.done is True
        assert gs.winner == WHITE, "Stalemated player should lose"


# ===================================================================
# 50-move draw
# ===================================================================

class TestFiftyMoveRule:
    """50 half-moves without capture or man move = DRAW."""

    def test_fifty_move_draw(self, no_capture_patches):
        GS = no_capture_patches
        gs = GS()

        # Directly set up the state to have half_move_clock at 49
        # and then make one more king move to trigger the rule.
        gs.board = [EMPTY] * NUM_SQUARES
        gs.board[0] = BLACK_KING
        gs.board[24] = WHITE_KING
        gs.current_player = BLACK
        gs.half_move_clock = 49  # one more king move will reach 50
        gs.move_count = 100
        gs.history = {}
        from khreibga.game import compute_zobrist_hash
        gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
        gs.history[gs.current_hash] = 1

        # Make one king move: BLACK_KING at 0 -> 1
        action = _action(0, 1)
        mask = gs.get_action_mask()
        assert mask[action] == 1

        obs, new_mask, reward, done = gs.step(action)

        assert gs.done is True
        assert gs.winner is None, "50-move rule should result in draw"
        assert gs.half_move_clock == 50


# ===================================================================
# 3-fold repetition
# ===================================================================

class TestThreeFoldRepetition:
    """Same board state occurring 3 times = DRAW."""

    def test_three_fold_draw(self, no_capture_patches):
        GS = no_capture_patches
        gs = GS()

        # Set up: two kings that shuffle back and forth
        gs.board = [EMPTY] * NUM_SQUARES
        gs.board[0] = BLACK_KING   # (0,0)
        gs.board[24] = WHITE_KING  # (4,4)
        gs.current_player = BLACK
        gs.half_move_clock = 0
        gs.move_count = 0
        gs.history = {}
        from khreibga.game import compute_zobrist_hash
        gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
        gs.history[gs.current_hash] = 1

        # Move pattern to create repetition:
        # Move 1: Black 0->5, Move 2: White 24->19
        # Move 3: Black 5->0, Move 4: White 19->24  (position repeats, count=2)
        # Move 5: Black 0->5, Move 6: White 24->19
        # Move 7: Black 5->0, Move 8: White 19->24  (position repeats, count=3 -> draw)

        moves_sequence = [
            _action(0, 5),    # Black king 0->5
            _action(24, 19),  # White king 24->19
            _action(5, 0),    # Black king 5->0
            _action(19, 24),  # White king 19->24  -- 2nd time
            _action(0, 5),    # Black king 0->5
            _action(24, 19),  # White king 24->19
            _action(5, 0),    # Black king 5->0
            _action(19, 24),  # White king 19->24  -- 3rd time
        ]

        for action in moves_sequence:
            if gs.done:
                break
            mask = gs.get_action_mask()
            assert mask[action] == 1, f"Action {action} should be legal"
            gs.step(action)

        assert gs.done is True
        assert gs.winner is None, "3-fold repetition should result in draw"


# ===================================================================
# Observation shape
# ===================================================================

class TestObservation:
    """Verify the observation tensor has the correct shape and structure."""

    def test_observation_shape(self, game):
        obs = game.get_observation()
        assert len(obs) == 7, "Should have 7 planes"
        for plane_idx, plane in enumerate(obs):
            assert len(plane) == 5, f"Plane {plane_idx} should have 5 rows"
            for row_idx, row in enumerate(plane):
                assert len(row) == 5, f"Plane {plane_idx} row {row_idx} should have 5 cols"

    def test_observation_initial_men_planes(self, game):
        """In initial position with BLACK to move:
        Plane 0 = BLACK men (current player's men)
        Plane 2 = WHITE men (opponent's men)
        """
        obs = game.get_observation()
        # Count non-zero cells in plane 0 (current player's men = BLACK_MAN)
        men_count = sum(obs[0][r][c] for r in range(5) for c in range(5))
        assert men_count == 12, "Should have 12 current player men initially"

        opp_men = sum(obs[2][r][c] for r in range(5) for c in range(5))
        assert opp_men == 12, "Should have 12 opponent men initially"

    def test_observation_no_kings_initially(self, game):
        obs = game.get_observation()
        king_count = sum(obs[1][r][c] for r in range(5) for c in range(5))
        assert king_count == 0, "No kings initially"
        opp_king_count = sum(obs[3][r][c] for r in range(5) for c in range(5))
        assert opp_king_count == 0, "No opponent kings initially"

    def test_observation_repetition_plane_initial(self, game):
        """Initial position has been seen once, so repetition = 0."""
        obs = game.get_observation()
        for r in range(5):
            for c in range(5):
                assert obs[4][r][c] == 0.0

    def test_observation_colour_plane_black(self, game):
        """When BLACK is current player, colour plane should be all 1s."""
        obs = game.get_observation()
        for r in range(5):
            for c in range(5):
                assert obs[5][r][c] == 1.0

    def test_observation_colour_plane_white(self, no_capture_patches):
        GS = no_capture_patches
        gs = GS()
        gs.current_player = WHITE
        from khreibga.game import compute_zobrist_hash
        gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
        gs.history = {gs.current_hash: 1}

        obs = gs.get_observation()
        for r in range(5):
            for c in range(5):
                assert obs[5][r][c] == 0.0

    def test_observation_move_count_plane(self, game):
        """At move 0, move count plane = 0/200 = 0."""
        obs = game.get_observation()
        for r in range(5):
            for c in range(5):
                assert obs[6][r][c] == 0.0

    def test_observation_move_count_nonzero(self, no_capture_patches):
        GS = no_capture_patches
        gs = GS()
        gs.move_count = 100
        obs = gs.get_observation()
        expected = 100 / 200
        for r in range(5):
            for c in range(5):
                assert abs(obs[6][r][c] - expected) < 1e-9

    def test_observation_flip_for_white(self, no_capture_patches):
        """When WHITE is current player, board should be flipped."""
        GS = no_capture_patches
        gs = GS()
        gs.board = [EMPTY] * NUM_SQUARES
        gs.board[0] = BLACK_MAN   # (0,0)
        gs.board[24] = WHITE_MAN  # (4,4)
        gs.current_player = WHITE
        from khreibga.game import compute_zobrist_hash
        gs.current_hash = compute_zobrist_hash(gs.board, gs.current_player)
        gs.history = {gs.current_hash: 1}

        obs = gs.get_observation()
        # For WHITE, flip means (r,c) -> (4-r, 4-c)
        # WHITE_MAN at (4,4) maps to obs position (0,0) and it's current player's man
        assert obs[0][0][0] == 1.0, "White man at (4,4) should appear at (0,0) in flipped view"
        # BLACK_MAN at (0,0) maps to obs position (4,4) and it's opponent's man
        assert obs[2][4][4] == 1.0, "Black man at (0,0) should appear at (4,4) in flipped view"


# ===================================================================
# Clone independence
# ===================================================================

class TestClone:
    """Cloning should create an independent copy."""

    def test_clone_is_equal(self, game):
        clone = game.clone()
        assert clone.board == game.board
        assert clone.current_player == game.current_player
        assert clone.chain_piece == game.chain_piece
        assert clone.half_move_clock == game.half_move_clock
        assert clone.move_count == game.move_count
        assert clone.history == game.history
        assert clone.current_hash == game.current_hash
        assert clone.done == game.done
        assert clone.winner == game.winner

    def test_clone_board_independence(self, game):
        """Modifying original board should not affect clone."""
        clone = game.clone()
        game.board[0] = EMPTY
        assert clone.board[0] == BLACK_MAN, "Clone board should be independent"

    def test_clone_history_independence(self, game):
        """Modifying original history should not affect clone."""
        clone = game.clone()
        game.history[99999] = 5
        assert 99999 not in clone.history, "Clone history should be independent"

    def test_clone_state_independence(self, game):
        """Modifying original game state fields should not affect clone."""
        clone = game.clone()
        game.current_player = WHITE
        game.done = True
        game.winner = BLACK
        game.move_count = 999
        game.half_move_clock = 999
        game.chain_piece = 10

        assert clone.current_player == BLACK
        assert clone.done is False
        assert clone.winner is None
        assert clone.move_count == 0
        assert clone.half_move_clock == 0
        assert clone.chain_piece is None


# ===================================================================
# Zobrist hashing
# ===================================================================

class TestZobrist:
    """Test Zobrist hashing consistency."""

    def test_initial_hash_is_deterministic(self, no_capture_patches):
        gs1 = no_capture_patches()
        gs2 = no_capture_patches()
        assert gs1.current_hash == gs2.current_hash

    def test_hash_changes_after_move(self, no_capture_patches):
        gs = no_capture_patches()
        initial_hash = gs.current_hash
        # Make a move: sq 7 (1,2) -> sq 12 (2,2) forward
        src, dst = 7, 12
        gs.step(_action(src, dst))
        assert gs.current_hash != initial_hash

    def test_hash_matches_recomputed(self, no_capture_patches):
        gs = no_capture_patches()
        src, dst = 7, 12
        gs.step(_action(src, dst))
        from khreibga.game import compute_zobrist_hash
        expected = compute_zobrist_hash(gs.board, gs.current_player)
        assert gs.current_hash == expected


# ===================================================================
# Opponent helper
# ===================================================================

class TestOpponent:
    """Test the opponent static helper."""

    def test_opponent_of_black(self):
        from khreibga.game import opponent
        assert opponent(BLACK) == WHITE

    def test_opponent_of_white(self):
        from khreibga.game import opponent
        assert opponent(WHITE) == BLACK


# ===================================================================
# Reset
# ===================================================================

class TestReset:
    """Test game reset functionality."""

    def test_reset_returns_obs_and_mask(self, game):
        obs, mask = game.reset()
        assert len(obs) == 7
        assert len(mask) == 625

    def test_reset_clears_state(self, no_capture_patches):
        gs = no_capture_patches()
        gs.move_count = 50
        gs.half_move_clock = 30
        gs.done = True
        gs.winner = BLACK

        gs.reset()
        assert gs.move_count == 0
        assert gs.half_move_clock == 0
        assert gs.done is False
        assert gs.winner is None
        assert gs.current_player == BLACK


# ===================================================================
# Action mask
# ===================================================================

class TestActionMask:
    """Test action mask generation."""

    def test_action_mask_length(self, game):
        mask = game.get_action_mask()
        assert len(mask) == 625

    def test_action_mask_has_legal_moves(self, game):
        mask = game.get_action_mask()
        assert sum(mask) > 0, "Initial position should have legal moves"

    def test_action_mask_done_game(self, game):
        game.done = True
        mask = game.get_action_mask()
        assert sum(mask) == 0, "Done game should have no legal moves"


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Various edge case tests."""

    def test_step_on_done_game(self, game):
        game.done = True
        obs, mask, reward, done = game.step(0)
        assert done is True
        assert reward == 0.0

    def test_repr(self, game):
        r = repr(game)
        assert "GameState" in r
        assert "BLACK" in r
        assert "ongoing" in r


# ===================================================================
# Additional edge cases
# ===================================================================

class TestAdditionalGameEdges:
    """Additional checks for observation planes and chain-specific masking."""

    def test_observation_repetition_plane_seen_before(self, no_capture_patches):
        gs = no_capture_patches()
        gs.history[gs.current_hash] = 2

        obs = gs.get_observation()
        for r in range(5):
            for c in range(5):
                assert obs[4][r][c] == 1.0

    def test_observation_move_count_plane_is_capped(self, no_capture_patches):
        gs = no_capture_patches()
        gs.move_count = 999

        obs = gs.get_observation()
        for r in range(5):
            for c in range(5):
                assert obs[6][r][c] == 1.0

    def test_reset_recreates_single_history_entry(self, no_capture_patches):
        gs = no_capture_patches()
        gs.history[123] = 4
        gs.move_count = 17
        gs.done = True
        gs.chain_piece = 12

        gs.reset()

        assert len(gs.history) == 1
        assert gs.history[gs.current_hash] == 1
        assert gs.move_count == 0
        assert gs.done is False
        assert gs.chain_piece is None

    def test_get_action_mask_mid_chain_uses_piece_local_hops_only(self):
        from khreibga.game import GameState

        gs = GameState.__new__(GameState)
        gs.board = [EMPTY] * NUM_SQUARES
        gs.current_player = BLACK
        gs.chain_piece = 12
        gs.done = False

        with patch("khreibga.game.get_piece_capture_first_hops", return_value=[(12, 2), (12, 22)]), \
             patch("khreibga.game.captures_get_action_mask", side_effect=RuntimeError("must not be called mid-chain")):
            mask = gs.get_action_mask()

        assert mask[_action(12, 2)] == 1
        assert mask[_action(12, 22)] == 1
        assert sum(mask) == 2

    def test_step_on_done_game_does_not_mutate_state(self, game):
        game.done = True
        before_board = game.board[:]
        before_player = game.current_player
        before_move_count = game.move_count
        before_hash = game.current_hash

        _obs, _mask, reward, done = game.step(_action(7, 12))

        assert done is True
        assert reward == 0.0
        assert game.board == before_board
        assert game.current_player == before_player
        assert game.move_count == before_move_count
        assert game.current_hash == before_hash

    def test_repr_includes_chain_piece_marker(self, game):
        game.chain_piece = 12
        assert "chain@12" in repr(game)
