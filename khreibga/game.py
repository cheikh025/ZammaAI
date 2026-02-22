"""
Khreibaga (Mauritanian Alquerque) game engine.

Implements the full game state machine, including:
- Turn management with multi-step capture chains
- Promotion rules (deferred during chains)
- Terminal condition detection (elimination, stalemate, 50-move, 3-fold repetition)
- Zobrist hashing for efficient repetition detection
- Canonical observation tensor (5x5x7) for RL agent input
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

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
    sq_to_rc,
    rc_to_sq,
)
from khreibga.move_gen import (
    get_simple_moves,
    get_player_pieces,
    is_player_piece,
)
from khreibga.captures import (
    get_capture_first_hops,
    get_piece_capture_first_hops,
    get_action_mask as captures_get_action_mask,
    execute_hop,
    find_all_capture_chains,
)

# ---------------------------------------------------------------------------
# Zobrist hashing table (deterministic, seeded with 42)
# ---------------------------------------------------------------------------

_rng = random.Random(42)

# 25 squares * 4 piece types (BLACK_MAN=1, BLACK_KING=2, WHITE_MAN=3, WHITE_KING=4)
# Index: ZOBRIST_PIECE[sq][piece_type - 1]
ZOBRIST_PIECE: List[List[int]] = [
    [_rng.getrandbits(64) for _ in range(4)]
    for _ in range(NUM_SQUARES)
]

ZOBRIST_SIDE: int = _rng.getrandbits(64)

# Action space size
ACTION_SPACE = NUM_SQUARES * NUM_SQUARES  # 625

# Promotion rows
_PROMO_ROW: Dict[int, int] = {
    BLACK: BOARD_SIZE - 1,  # Row 4
    WHITE: 0,               # Row 0
}

# Promotion squares (corners only -- Khreibaga rule).
# A man must land on one of these two corner squares to become a King.
# Landing on the three middle squares of the back row does NOT promote.
_PROMO_SQUARES: Dict[int, set] = {
    BLACK: {rc_to_sq(BOARD_SIZE - 1, 0), rc_to_sq(BOARD_SIZE - 1, BOARD_SIZE - 1)},  # sq 20, 24
    WHITE: {rc_to_sq(0, 0), rc_to_sq(0, BOARD_SIZE - 1)},                            # sq 0, 4
}

# Piece type mappings
_PLAYER_MAN: Dict[int, int] = {BLACK: BLACK_MAN, WHITE: WHITE_MAN}
_PLAYER_KING: Dict[int, int] = {BLACK: BLACK_KING, WHITE: WHITE_KING}

# Max half-moves for 50-move rule
_FIFTY_MOVE_LIMIT = 50

# Max move count for observation normalization
_MAX_STEPS = 200


# ---------------------------------------------------------------------------
# Zobrist helper functions
# ---------------------------------------------------------------------------

def compute_zobrist_hash(board: List[int], current_player: int) -> int:
    """Compute full Zobrist hash from scratch for a board position."""
    h = 0
    for sq in range(NUM_SQUARES):
        piece = board[sq]
        if piece != EMPTY:
            h ^= ZOBRIST_PIECE[sq][piece - 1]
    if current_player == BLACK:
        h ^= ZOBRIST_SIDE
    return h


def _zobrist_toggle_piece(h: int, sq: int, piece: int) -> int:
    """XOR in/out a piece at a given square."""
    return h ^ ZOBRIST_PIECE[sq][piece - 1]


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------

class GameState:
    """Full game state for Khreibaga with RL environment interface.

    The NN outputs one atomic hop at a time. The environment tracks whether
    a capture chain is in progress and restricts the action mask accordingly.
    """

    def __init__(self) -> None:
        self.board: List[int] = initial_board()
        self.current_player: int = BLACK  # Black moves first
        self.chain_piece: Optional[int] = None  # sq of piece mid-chain, or None
        self.half_move_clock: int = 0  # for 50-move rule
        self.move_count: int = 0  # total half-moves played
        self.history: Dict[int, int] = {}  # hash -> count
        self.done: bool = False
        self.winner: Optional[int] = None  # BLACK, WHITE, or None (draw)

        # Compute initial hash and record it
        self.current_hash: int = compute_zobrist_hash(self.board, self.current_player)
        self.history[self.current_hash] = self.history.get(self.current_hash, 0) + 1

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[list, List[int]]:
        """Reset to initial position. Return (observation, action_mask)."""
        self.__init__()
        return self.get_observation(), self.get_action_mask()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int) -> Tuple[list, List[int], float, bool]:
        """Execute one atomic hop.

        Parameters
        ----------
        action : int
            Action index 0..624.  ``src = action // 25``, ``dst = action % 25``.

        Returns
        -------
        observation : nested list (5x5x7)
        action_mask : list of 625 ints
        reward : float  (+1 win, -1 loss, 0 draw/ongoing) from perspective of
                 the player who just acted (i.e. self.current_player BEFORE switch).
        done : bool
        """
        if self.done:
            return (
                self.get_observation(),
                self.get_action_mask(),
                0.0,
                True,
            )

        src = action // NUM_SQUARES
        dst = action % NUM_SQUARES

        # Validate action against current mask
        mask = self.get_action_mask()
        if mask[action] != 1:
            raise ValueError(
                f"Illegal action {action} (src={src}, dst={dst}). "
                f"Not in current action mask."
            )

        acting_player = self.current_player

        # Determine if this is a capture or simple move
        is_capture = self._is_capture_action(src, dst)

        if is_capture:
            self._execute_capture(src, dst)
        else:
            self._execute_simple_move(src, dst)

        # Update move count (always, for every half-move)
        self.move_count += 1

        # Update Zobrist hash and history
        self.current_hash = compute_zobrist_hash(self.board, self.current_player)
        self.history[self.current_hash] = self.history.get(self.current_hash, 0) + 1

        # Check terminal conditions
        self._check_terminal()

        # Compute reward from acting player's perspective
        reward = 0.0
        if self.done:
            if self.winner == acting_player:
                reward = 1.0
            elif self.winner is None:
                reward = 0.0  # draw
            else:
                reward = -1.0

        return (
            self.get_observation(),
            self.get_action_mask(),
            reward,
            self.done,
        )

    # ------------------------------------------------------------------
    # Capture execution
    # ------------------------------------------------------------------

    def _is_capture_action(self, src: int, dst: int) -> bool:
        """Determine if the action from src to dst is a capture hop.

        If we are mid-chain, it's always a capture. Otherwise, check if
        capture hops are available (which would mean captures are mandatory).
        """
        if self.chain_piece is not None:
            return True

        # Check if any captures exist for current player
        capture_hops = get_capture_first_hops(self.board, self.current_player)
        if capture_hops:
            return True

        return False

    def _execute_capture(self, src: int, dst: int) -> None:
        """Execute a capture hop and manage chain state."""
        new_board, captured_sq = execute_hop(
            self.board, src, dst, self.current_player
        )
        self.board = new_board

        # Reset half-move clock on any capture
        self.half_move_clock = 0

        # Check if further captures exist from dst for same piece
        further_captures = self._get_further_captures(dst)

        if further_captures:
            # Chain continues: do NOT switch player, do NOT promote
            self.chain_piece = dst
        else:
            # Chain ends
            self.chain_piece = None
            self._check_promotion(dst)
            self._switch_player()

    def _get_further_captures(self, sq: int) -> List[Tuple[int, int]]:
        """Check if the piece at sq can continue capturing.

        Returns list of (src, dst) capture hops from that square.
        Uses piece-local capture generation because mid-chain rules lock the
        active piece; global majority across other pieces must not apply here.
        """
        return get_piece_capture_first_hops(self.board, self.current_player, sq)

    # ------------------------------------------------------------------
    # Simple move execution
    # ------------------------------------------------------------------

    def _execute_simple_move(self, src: int, dst: int) -> None:
        """Execute a simple (non-capture) move."""
        piece = self.board[src]

        # Track if it's a man move for 50-move rule
        is_man_move = piece in (BLACK_MAN, WHITE_MAN)

        # Move the piece
        self.board[src] = EMPTY
        self.board[dst] = piece

        # Check promotion
        self._check_promotion(dst)

        # Update half-move clock
        if is_man_move:
            self.half_move_clock = 0
        else:
            self.half_move_clock += 1

        # Switch player
        self._switch_player()

    # ------------------------------------------------------------------
    # Player switching
    # ------------------------------------------------------------------

    def _switch_player(self) -> None:
        """Toggle current player between BLACK and WHITE."""
        self.current_player = WHITE if self.current_player == BLACK else BLACK

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def _check_promotion(self, sq: int) -> None:
        """Promote a man to king if it ended its turn on a promotion corner.

        Khreibaga rule: only the two corner squares of the opponent's back
        row are promotion squares.  The three middle squares are a "dead
        zone" where the man stays unpromoted.
        """
        piece = self.board[sq]

        if piece == BLACK_MAN and sq in _PROMO_SQUARES[BLACK]:
            self.board[sq] = BLACK_KING
        elif piece == WHITE_MAN and sq in _PROMO_SQUARES[WHITE]:
            self.board[sq] = WHITE_KING

    # ------------------------------------------------------------------
    # Terminal condition check
    # ------------------------------------------------------------------

    def _check_terminal(self) -> None:
        """Check all terminal conditions and set done/winner."""
        if self.done:
            return

        # 1. Elimination: current player has 0 pieces -> current player loses
        current_pieces = get_player_pieces(self.board, self.current_player)
        if len(current_pieces) == 0:
            self.done = True
            self.winner = opponent(self.current_player)
            return

        # Also check if opponent has 0 pieces (the acting player captured last one)
        opp = opponent(self.current_player)
        opp_pieces = get_player_pieces(self.board, opp)
        if len(opp_pieces) == 0:
            self.done = True
            self.winner = self.current_player
            return

        # 2. Stalemate: current player has pieces but no legal moves
        # Only check if not mid-chain (mid-chain always has moves)
        if self.chain_piece is None:
            mask = self.get_action_mask()
            if sum(mask) == 0:
                self.done = True
                self.winner = opponent(self.current_player)
                return

        # 3. 50-move rule
        if self.half_move_clock >= _FIFTY_MOVE_LIMIT:
            self.done = True
            self.winner = None  # draw
            return

        # 4. 3-fold repetition
        if self.history.get(self.current_hash, 0) >= 3:
            self.done = True
            self.winner = None  # draw
            return

    # ------------------------------------------------------------------
    # Action mask
    # ------------------------------------------------------------------

    def get_action_mask(self) -> List[int]:
        """Return a list of 625 ints (0 or 1) indicating legal actions.

        If chain_piece is set, only capture continuations from that piece
        are legal. Otherwise, delegate to captures.get_action_mask.
        """
        if self.done:
            return [0] * ACTION_SPACE

        if self.chain_piece is not None:
            # Mid-chain: only allow capture hops from chain_piece
            mask = [0] * ACTION_SPACE
            further = self._get_further_captures(self.chain_piece)
            for s, d in further:
                mask[s * NUM_SQUARES + d] = 1
            return mask

        # Normal turn: delegate to captures module
        return captures_get_action_mask(self.board, self.current_player)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self) -> list:
        """Return 5x5x7 observation tensor as nested lists.

        Canonical form: if current player is WHITE, flip the board so
        they are always 'moving up' (from row 0 toward row 4).

        Planes:
          0: current player's men
          1: current player's kings
          2: opponent's men
          3: opponent's kings
          4: repetition (1 if this board state appeared before)
          5: colour (all 1s if current player is BLACK, all 0s if WHITE)
          6: move count (t/200, capped at 1.0)
        """
        if self.current_player == BLACK:
            my_man, my_king = BLACK_MAN, BLACK_KING
            opp_man, opp_king = WHITE_MAN, WHITE_KING
            flip = False
        else:
            my_man, my_king = WHITE_MAN, WHITE_KING
            opp_man, opp_king = BLACK_MAN, BLACK_KING
            flip = True

        # Build the 7 planes
        obs = [[[0.0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)] for _ in range(7)]

        # Repetition value: 1 if current hash has appeared before (count > 1)
        rep_val = 1.0 if self.history.get(self.current_hash, 0) > 1 else 0.0

        # Colour plane value
        colour_val = 1.0 if self.current_player == BLACK else 0.0

        # Move count plane value
        mc_val = min(self.move_count / _MAX_STEPS, 1.0)

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if flip:
                    # Flip: map (r, c) -> (4-r, 4-c)
                    board_r = BOARD_SIZE - 1 - r
                    board_c = BOARD_SIZE - 1 - c
                else:
                    board_r = r
                    board_c = c

                sq = rc_to_sq(board_r, board_c)
                piece = self.board[sq]

                # Plane 0: current player's men
                if piece == my_man:
                    obs[0][r][c] = 1.0
                # Plane 1: current player's kings
                if piece == my_king:
                    obs[1][r][c] = 1.0
                # Plane 2: opponent's men
                if piece == opp_man:
                    obs[2][r][c] = 1.0
                # Plane 3: opponent's kings
                if piece == opp_king:
                    obs[3][r][c] = 1.0

                # Plane 4: repetition
                obs[4][r][c] = rep_val
                # Plane 5: colour
                obs[5][r][c] = colour_val
                # Plane 6: move count
                obs[6][r][c] = mc_val

        return obs

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def clone(self) -> GameState:
        """Return a deep copy of this game state (for MCTS)."""
        new = GameState.__new__(GameState)
        new.board = self.board[:]
        new.current_player = self.current_player
        new.chain_piece = self.chain_piece
        new.half_move_clock = self.half_move_clock
        new.move_count = self.move_count
        new.history = dict(self.history)
        new.current_hash = self.current_hash
        new.done = self.done
        new.winner = self.winner
        return new

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "done" if self.done else "ongoing"
        player = "BLACK" if self.current_player == BLACK else "WHITE"
        chain = f", chain@{self.chain_piece}" if self.chain_piece is not None else ""
        return (
            f"GameState({status}, {player} to move, "
            f"move={self.move_count}, clock={self.half_move_clock}{chain})"
        )


# ---------------------------------------------------------------------------
# Static helper
# ---------------------------------------------------------------------------

def opponent(player: int) -> int:
    """Return the other player."""
    return WHITE if player == BLACK else BLACK
