from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field


class CreateGameRequest(BaseModel):
    mode: Literal["hvh", "hvr", "hvai", "aivh"] = "hvh"
    ai_simulations: int = Field(200, ge=1, le=5000)


class MoveRequest(BaseModel):
    action: Optional[int] = None   # 0..624
    src: Optional[int] = None      # 0..24
    dst: Optional[int] = None      # 0..24

    def resolve_action(self) -> int:
        if self.action is not None:
            return self.action
        if self.src is not None and self.dst is not None:
            return self.src * 25 + self.dst
        raise ValueError("Provide 'action' or both 'src' and 'dst'.")


class GameStateResponse(BaseModel):
    session_id: str
    board: list[int]
    current_player: int
    chain_piece: Optional[int]
    done: bool
    winner: Optional[int]
    terminal_reason: Optional[str]
    move_count: int
    half_move_clock: int
    action_mask: list[int]
    last_action: Optional[int]
    legal_actions_count: int
