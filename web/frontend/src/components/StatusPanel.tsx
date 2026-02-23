import type { GameState, GameMode } from '../types'

const PLAYER: Record<number, string> = { 1: 'Black', 2: 'White' }
const MODE_LABEL: Record<GameMode, string> = {
  hvh: 'Human vs Human',
  hvr: 'Human vs Random AI',
  hvai: 'Human vs MCTS AI',
  aivh: 'MCTS AI vs Human',
}
const REASON_LABEL: Record<string, string> = {
  elimination: 'Elimination',
  stalemate: 'Stalemate',
  fifty_move_rule: '50-Move Rule',
  threefold_repetition: 'Threefold Repetition',
}

interface Props {
  gameState: GameState
  mode: GameMode
  aiThinking: boolean
  showEvalBar: boolean
  showHints: boolean
  showFeedback: boolean
  onToggleEvalBar: () => void
  onToggleHints: () => void
  onToggleFeedback: () => void
  onRequestHints: () => void
  hintsLoading: boolean
}

export function StatusPanel({
  gameState, mode, aiThinking,
  showEvalBar, showHints, showFeedback,
  onToggleEvalBar, onToggleHints, onToggleFeedback,
  onRequestHints, hintsLoading,
}: Props) {
  const {
    current_player,
    done,
    winner,
    terminal_reason,
    move_count,
    half_move_clock,
    chain_piece,
    legal_actions_count,
  } = gameState

  return (
    <div className="status-panel">
      <h2 className="panel-title">Status</h2>

      <div className="stat-row">
        <span className="stat-label">Mode</span>
        <span className="stat-value">{MODE_LABEL[mode]}</span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Move</span>
        <span className="stat-value">{move_count}</span>
      </div>
      <div className="stat-row">
        <span className="stat-label">50-move clock</span>
        <span className="stat-value" style={{ color: half_move_clock >= 40 ? '#e74c3c' : undefined }}>
          {half_move_clock} / 50
        </span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Legal actions</span>
        <span className="stat-value">{legal_actions_count}</span>
      </div>

      <div className="divider" />

      {!done ? (
        <div className="turn-section">
          <div
            className="turn-indicator"
            style={{ '--player-color': current_player === 1 ? '#1a0f0a' : '#f5ede0' } as React.CSSProperties}
          >
            <span className="piece-dot" style={{ background: current_player === 1 ? '#1a0f0a' : '#f5ede0', border: current_player === 2 ? '2px solid #9e8060' : 'none' }} />
            {PLAYER[current_player]}'s turn
          </div>

          {chain_piece !== null && (
            <div className="badge badge-chain">
              ⛓ Forced capture chain — continue piece at {sqAlg(chain_piece)}
            </div>
          )}

          {aiThinking && (
            <div className="badge badge-ai">
              <span className="spinner" /> AI thinking…
            </div>
          )}
        </div>
      ) : (
        <div className="result-section">
          <div className="result-headline">
            {winner !== null ? `${PLAYER[winner]} wins!` : 'Draw'}
          </div>
          {terminal_reason && (
            <div className="result-reason">
              {REASON_LABEL[terminal_reason] ?? terminal_reason}
            </div>
          )}
        </div>
      )}

      <div className="divider" />

      <h3 className="panel-subtitle">AI Insights</h3>

      <div className="toggle-row">
        <span className="toggle-label">Eval Bar</span>
        <label className="toggle-switch">
          <input type="checkbox" checked={showEvalBar} onChange={onToggleEvalBar} />
          <span className="toggle-slider" />
        </label>
      </div>

      <div className="toggle-row">
        <span className="toggle-label">Move Feedback</span>
        <label className="toggle-switch">
          <input type="checkbox" checked={showFeedback} onChange={onToggleFeedback} />
          <span className="toggle-slider" />
        </label>
      </div>

      <div className="toggle-row">
        <span className="toggle-label">Hints</span>
        <label className="toggle-switch">
          <input type="checkbox" checked={showHints} onChange={onToggleHints} />
          <span className="toggle-slider" />
        </label>
      </div>

      {showHints && (
        <button
          className="btn-secondary btn-sm"
          style={{ marginTop: 6, width: '100%' }}
          onClick={onRequestHints}
          disabled={aiThinking || done || hintsLoading}
        >
          {hintsLoading ? 'Thinking…' : 'Show Hint'}
        </button>
      )}
    </div>
  )
}

function sqAlg(sq: number): string {
  const r = Math.floor(sq / 5)
  const c = sq % 5
  return 'abcde'[c] + (r + 1)
}
