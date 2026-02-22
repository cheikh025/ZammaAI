from __future__ import annotations

from typing import Optional

from khreibga.board import BLACK, WHITE
from khreibga.game import GameState
from khreibga.move_gen import get_player_pieces

from .session import Session


def session_to_payload(session_id: str, session: Session) -> dict:
    """Convert a Session into the standard API response payload."""
    gs = session.game
    mask = gs.get_action_mask()
    return {
        "session_id":        session_id,
        "board":             gs.board,
        "current_player":    gs.current_player,   # 1=BLACK, 2=WHITE
        "chain_piece":       gs.chain_piece,
        "done":              gs.done,
        "winner":            gs.winner,            # 1=BLACK, 2=WHITE, None=draw
        "terminal_reason":   session.terminal_reason,
        "move_count":        gs.move_count,
        "half_move_clock":   gs.half_move_clock,
        "action_mask":       mask,
        "last_action":       session.last_action,
        "legal_actions_count": sum(mask),
    }


def compute_terminal_reason(gs: GameState) -> Optional[str]:
    """Determine why the game ended from the final GameState."""
    if not gs.done:
        return None

    if gs.winner is None:
        # Draw: either threefold repetition or 50-move rule
        if gs.history.get(gs.current_hash, 0) >= 3:
            return "threefold_repetition"
        return "fifty_move_rule"

    # Someone won: distinguish elimination from stalemate
    loser = WHITE if gs.winner == BLACK else BLACK
    if len(get_player_pieces(gs.board, loser)) == 0:
        return "elimination"
    return "stalemate"
