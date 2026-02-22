import type { GameState } from '../types'

const PLAYER: Record<number, string> = { 1: 'Black', 2: 'White' }
const REASON_LABEL: Record<string, string> = {
  elimination: 'Elimination',
  stalemate: 'Stalemate',
  fifty_move_rule: '50-Move Rule',
  threefold_repetition: 'Threefold Repetition',
}

interface Props {
  gameState: GameState
  onNewGame: () => void
  onRematch: () => void
}

export function GameModal({ gameState, onNewGame, onRematch }: Props) {
  const { winner, terminal_reason } = gameState

  return (
    <div className="modal-overlay">
      <div className="modal">
        <div className="modal-icon">
          {winner !== null ? (
            <span
              className="modal-piece"
              style={{
                background: winner === 1 ? '#1a0f0a' : '#f5ede0',
                border: winner === 2 ? '3px solid #9e8060' : 'none',
              }}
            />
          ) : (
            <span className="modal-draw-icon">½</span>
          )}
        </div>

        <h2 className="modal-title">
          {winner !== null ? `${PLAYER[winner]} wins!` : 'Draw'}
        </h2>

        {terminal_reason && (
          <p className="modal-reason">
            {REASON_LABEL[terminal_reason] ?? terminal_reason}
          </p>
        )}

        <div className="modal-actions">
          <button className="btn-primary" onClick={onRematch}>
            Rematch
          </button>
          <button className="btn-secondary" onClick={onNewGame}>
            New Game
          </button>
        </div>
      </div>
    </div>
  )
}
