import { useState } from 'react'
import { useGame } from './state/useGame'
import { Board } from './components/Board'
import { StatusPanel } from './components/StatusPanel'
import { MoveHistory } from './components/MoveHistory'
import { GameModal } from './components/GameModal'
import type { GameMode } from './types'

// ---------------------------------------------------------------------------
// Setup screen
// ---------------------------------------------------------------------------

interface SetupProps {
  onStart: (mode: GameMode, sims: number) => void
  loading: boolean
  error: string | null
}

function SetupScreen({ onStart, loading, error }: SetupProps) {
  const [mode, setMode] = useState<GameMode>('hvh')
  const [sims, setSims] = useState(10)

  return (
    <div className="setup-screen">
      <div className="setup-card">
        <div className="setup-logo">
          <div className="logo-pieces">
            <span className="logo-piece black" />
            <span className="logo-piece white" />
          </div>
          <h1>Khreibga Zero</h1>
          <p className="setup-subtitle">Mauritanian Alquerque · AlphaZero AI</p>
        </div>

        <div className="setup-form">
          <div className="form-group">
            <label className="form-label">Game Mode</label>
            <select
              className="form-select"
              value={mode}
              onChange={(e) => setMode(e.target.value as GameMode)}
            >
              <option value="hvh">Human vs Human</option>
              <option value="hvr">Human vs Random AI</option>
              <option value="hvai">Human vs MCTS AI</option>
              <option value="aivh">MCTS AI vs Human (AI plays first)</option>
            </select>
          </div>

          {(mode === 'hvai' || mode === 'aivh') && (
            <div className="form-group">
              <label className="form-label">
                MCTS Simulations
                <span className="form-hint"> (higher = stronger, slower)</span>
              </label>
              <input
                type="number"
                className="form-input"
                value={sims}
                min={50}
                max={2000}
                step={50}
                onChange={(e) => setSims(Number(e.target.value))}
              />
            </div>
          )}

          <button
            className="btn-primary btn-large"
            onClick={() => onStart(mode, sims)}
            disabled={loading}
          >
            {loading ? 'Starting…' : 'Start Game'}
          </button>

          {error && <div className="error-msg">{error}</div>}
        </div>

        <div className="setup-rules">
          <h3>How to play</h3>
          <ul>
            <li>Click a piece to select it — legal destinations highlight in yellow.</li>
            <li>Captures are mandatory; the longest chain must be played.</li>
            <li>Pieces promote to Kings at the two corner squares of the far rank.</li>
            <li>Kings move any distance along clear lines (orthogonal or diagonal).</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

export default function App() {
  const {
    gameState,
    mode,
    selectedSq,
    sources,
    dests,
    loading,
    aiThinking,
    error,
    moveLog,
    startGame,
    leaveGame,
    resetGame,
    clickSquare,
  } = useGame()

  // Persist mode/sims so modal "rematch" button uses same settings
  const [lastMode, setLastMode] = useState<GameMode>('hvh')
  const [lastSims, setLastSims] = useState(10)

  const handleStart = (m: GameMode, sims: number) => {
    setLastMode(m)
    setLastSims(sims)
    startGame(m, sims)
  }

  if (!gameState) {
    return (
      <SetupScreen
        onStart={handleStart}
        loading={loading}
        error={error}
      />
    )
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-brand">
          <span className="header-dot black-dot" />
          <span className="header-dot white-dot" />
          <span className="header-title">Khreibga Zero</span>
        </div>
        <div className="header-actions">
          <button
            className="btn-secondary btn-sm"
            onClick={resetGame}
            disabled={loading || aiThinking}
          >
            Restart
          </button>
          <button
            className="btn-ghost btn-sm"
            onClick={leaveGame}
          >
            ← Menu
          </button>
        </div>
      </header>

      <main className="game-layout">
        <section className="board-area">
          <div className="board-wrapper">
            <Board
              gameState={gameState}
              selectedSq={selectedSq}
              sources={sources}
              dests={dests}
              onSquareClick={clickSquare}
            />
            {(loading || aiThinking) && (
              <div className="board-overlay">
                <span className="spinner-large" />
              </div>
            )}
          </div>
        </section>

        <aside className="side-panel">
          <StatusPanel
            gameState={gameState}
            mode={mode}
            aiThinking={aiThinking}
          />
          <MoveHistory moveLog={moveLog} />
        </aside>
      </main>

      {error && (
        <div className="error-bar">
          {error}
          <button className="error-dismiss" onClick={() => {}}>×</button>
        </div>
      )}

      {gameState.done && (
        <GameModal
          gameState={gameState}
          onRematch={() => startGame(lastMode, lastSims)}
          onNewGame={leaveGame}
        />
      )}
    </div>
  )
}
