from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..schemas.game import (
    CreateGameRequest, MoveRequest, GameStateResponse,
    EvaluationResponse, HintResponse,
)
from ..services.session import session_manager
from ..services.serializer import session_to_payload, compute_terminal_reason
from ..services.ai import get_ai_action, evaluate_position, hint_actions

router = APIRouter(prefix="/games")


# ---------------------------------------------------------------------------
# Error helpers (returns the exact contract: {error_code, message, details})
# ---------------------------------------------------------------------------

def _err(status: int, error_code: str, message: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error_code": error_code,
            "message": message,
            "details": details or {},
        },
    )


def _get_session_or_404(session_id: str):
    session = session_manager.get(session_id)
    if session is None:
        return None, _err(
            404, "SESSION_NOT_FOUND",
            f"Session '{session_id}' not found.",
        )
    return session, None


# ---------------------------------------------------------------------------
# POST /api/games  — create a new session
# ---------------------------------------------------------------------------

@router.post("", response_model=GameStateResponse)
def create_game(req: CreateGameRequest):
    session_id, session = session_manager.create(req.mode, req.ai_simulations)
    return session_to_payload(session_id, session)


# ---------------------------------------------------------------------------
# GET /api/games/{session_id}  — fetch current state
# ---------------------------------------------------------------------------

@router.get("/{session_id}", response_model=GameStateResponse)
def get_game(session_id: str):
    session, err = _get_session_or_404(session_id)
    if err:
        return err
    return session_to_payload(session_id, session)


# ---------------------------------------------------------------------------
# POST /api/games/{session_id}/move  — human makes one atomic hop
# ---------------------------------------------------------------------------

@router.post("/{session_id}/move", response_model=GameStateResponse)
def make_move(session_id: str, req: MoveRequest):
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    if session.game.done:
        return _err(409, "GAME_ALREADY_OVER", "The game has already ended.")

    # Resolve action index
    try:
        action = req.resolve_action()
    except ValueError as exc:
        return _err(422, "INVALID_ACTION_FORMAT", str(exc))

    if not (0 <= action < 625):
        return _err(
            422, "INVALID_ACTION_FORMAT",
            f"Action {action} is out of range 0..624.",
            {"action": action},
        )

    mask = session.game.get_action_mask()
    if mask[action] != 1:
        return _err(
            422, "ILLEGAL_ACTION",
            "Action is not legal in the current position.",
            {"action": action},
        )

    session.last_action = action
    session.game.step(action)

    if session.game.done:
        session.terminal_reason = compute_terminal_reason(session.game)

    return session_to_payload(session_id, session)


# ---------------------------------------------------------------------------
# POST /api/games/{session_id}/ai-move  — AI makes one atomic hop
# ---------------------------------------------------------------------------

@router.post("/{session_id}/ai-move", response_model=GameStateResponse)
def ai_move(session_id: str):
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    if session.mode == "hvh":
        return _err(409, "AI_NOT_CONFIGURED", "Session mode 'hvh' has no AI.")

    if session.game.done:
        return _err(409, "GAME_ALREADY_OVER", "The game has already ended.")

    action = get_ai_action(session.game, session.mode, session.ai_simulations)

    session.last_action = action
    session.game.step(action)

    if session.game.done:
        session.terminal_reason = compute_terminal_reason(session.game)

    return session_to_payload(session_id, session)


# ---------------------------------------------------------------------------
# POST /api/games/{session_id}/evaluate  — NN position evaluation
# ---------------------------------------------------------------------------

@router.post("/{session_id}/evaluate", response_model=EvaluationResponse)
def evaluate(session_id: str):
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    game = session.game
    if game.done:
        # Return definitive value without NN call
        if game.winner is None:
            value = 0.0
        elif game.winner == game.current_player:
            value = 1.0
        else:
            value = -1.0
    else:
        value = evaluate_position(game)

    return EvaluationResponse(
        session_id=session_id,
        value=value,
        current_player=game.current_player,
    )


# ---------------------------------------------------------------------------
# POST /api/games/{session_id}/hint  — top moves from lightweight MCTS
# ---------------------------------------------------------------------------

@router.post("/{session_id}/hint", response_model=HintResponse)
def hint(session_id: str):
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    if session.game.done:
        return _err(409, "GAME_ALREADY_OVER", "The game has already ended.")

    hints = hint_actions(session.game)
    return HintResponse(session_id=session_id, hints=hints)


# ---------------------------------------------------------------------------
# POST /api/games/{session_id}/reset  — restart the session
# ---------------------------------------------------------------------------

@router.post("/{session_id}/reset", response_model=GameStateResponse)
def reset_game(session_id: str):
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session.game.reset()
    session.terminal_reason = None
    session.last_action = None

    return session_to_payload(session_id, session)


# ---------------------------------------------------------------------------
# DELETE /api/games/{session_id}  — clean up a session
# ---------------------------------------------------------------------------

@router.delete("/{session_id}", status_code=204)
def delete_game(session_id: str):
    session_manager.delete(session_id)
