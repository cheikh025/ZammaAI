from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from khreibga.game import GameState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_TTL      = 2 * 60 * 60   # 2 hours of inactivity
CLEANUP_INTERVAL = 10 * 60       # run cleanup every 10 minutes


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class Session:
    game: GameState
    mode: str               # 'hvh' | 'hvr' | 'hvai'
    ai_simulations: int
    terminal_reason: Optional[str]
    last_action: Optional[int]
    created_at: float       = field(default_factory=time.time)
    last_accessed: float    = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, mode: str, ai_simulations: int = 200) -> tuple[str, Session]:
        session_id = str(uuid.uuid4())
        session = Session(
            game=GameState(),
            mode=mode,
            ai_simulations=ai_simulations,
            terminal_reason=None,
            last_action=None,
        )
        self._sessions[session_id] = session
        return session_id, session

    def get(self, session_id: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if session is not None:
            session.last_accessed = time.time()
        return session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # TTL cleanup
    # ------------------------------------------------------------------

    def cleanup_stale(self) -> int:
        cutoff = time.time() - SESSION_TTL
        stale = [sid for sid, s in self._sessions.items() if s.last_accessed < cutoff]
        for sid in stale:
            del self._sessions[sid]
        return len(stale)

    async def cleanup_loop(self) -> None:
        """Background coroutine: purge stale sessions every CLEANUP_INTERVAL seconds."""
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL)
            removed = self.cleanup_stale()
            if removed:
                print(f"[SessionManager] Removed {removed} stale session(s).")


# ---------------------------------------------------------------------------
# Module-level singleton (imported by routes)
# ---------------------------------------------------------------------------

session_manager = SessionManager()
