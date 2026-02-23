import { useState, useEffect, useRef } from 'react'
import { useGame } from './state/useGame'
import { useInsights } from './state/useInsights'
import { Board } from './components/Board'
import { StatusPanel } from './components/StatusPanel'
import { MoveHistory } from './components/MoveHistory'
import { GameModal } from './components/GameModal'
import { EvalBar } from './components/EvalBar'
import { HintOverlay } from './components/HintOverlay'
import { MoveFeedback } from './components/MoveFeedback'
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

  const insights = useInsights()

  // Track state for move detection
  const prevMoveCount = useRef<number>(0)
  const prevAiThinking = useRef<boolean>(false)
  const humanClickedRef = useRef<boolean>(false)

  // Wrap clickSquare to flag human moves (set synchronously before any async/state batching)
  const handleSquareClick = (sq: number) => {
    humanClickedRef.current = true
    clickSquare(sq)
  }

  // Detect human moves via move_count changes
  useEffect(() => {
    if (!gameState) return
    const mc = gameState.move_count
    if (mc !== prevMoveCount.current && humanClickedRef.current) {
      // move_count changed after a human click — skip mid-chain hops
      if (gameState.chain_piece === null) {
        insights.onPositionChange(gameState.session_id, gameState.current_player, gameState.done, true)
        humanClickedRef.current = false
      }
    }
    prevMoveCount.current = mc
  }, [gameState?.move_count]) // eslint-disable-line react-hooks/exhaustive-deps

  // Detect AI-finished: aiThinking went true→false
  useEffect(() => {
    if (!gameState) return
    if (prevAiThinking.current && !aiThinking) {
      insights.onPositionChange(gameState.session_id, gameState.current_player, gameState.done, false)
      humanClickedRef.current = false
    }
    prevAiThinking.current = aiThinking
  }, [aiThinking]) // eslint-disable-line react-hooks/exhaustive-deps

  // Catch-up eval fetch when toggling eval bar or feedback ON mid-game
  useEffect(() => {
    if (!gameState || gameState.done) return
    if ((insights.showEvalBar || insights.showFeedback) && !insights.evaluation) {
      insights.onPositionChange(gameState.session_id, gameState.current_player, gameState.done, false)
    }
  }, [insights.showEvalBar, insights.showFeedback]) // eslint-disable-line react-hooks/exhaustive-deps

  // Persist mode/sims so modal "rematch" button uses same settings
  const [lastMode, setLastMode] = useState<GameMode>('hvh')
  const [lastSims, setLastSims] = useState(10)

  const handleStart = (m: GameMode, sims: number) => {
    setLastMode(m)
    setLastSims(sims)
    insights.clearInsights()
    prevMoveCount.current = 0
    startGame(m, sims)
  }

  const handleLeave = () => {
    insights.clearInsights()
    prevMoveCount.current = 0
    leaveGame()
  }

  const handleReset = () => {
    insights.clearInsights()
    prevMoveCount.current = 0
    resetGame()
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
            onClick={handleReset}
            disabled={loading || aiThinking}
          >
            Restart
          </button>
          <button
            className="btn-ghost btn-sm"
            onClick={handleLeave}
          >
            ← Menu
          </button>
        </div>
      </header>

      <main className="game-layout">
        <section className="board-area">
          <div className="board-container">
            {insights.showEvalBar && (
              <EvalBar evaluation={insights.evaluation} />
            )}
            <div className="board-wrapper" style={{ position: 'relative' }}>
              <Board
                gameState={gameState}
                selectedSq={selectedSq}
                sources={sources}
                dests={dests}
                onSquareClick={handleSquareClick}
                hintOverlay={insights.showHints && insights.hints.length > 0
                  ? <HintOverlay hints={insights.hints} />
                  : undefined
                }
              />
              {(loading || aiThinking) && (
                <div className="board-overlay">
                  <span className="spinner-large" />
                </div>
              )}
            </div>
          </div>
        </section>

        <aside className="side-panel">
          <StatusPanel
            gameState={gameState}
            mode={mode}
            aiThinking={aiThinking}
            showEvalBar={insights.showEvalBar}
            showHints={insights.showHints}
            showFeedback={insights.showFeedback}
            onToggleEvalBar={insights.toggleEvalBar}
            onToggleHints={insights.toggleHints}
            onToggleFeedback={insights.toggleFeedback}
            onRequestHints={() => gameState && insights.requestHints(gameState.session_id)}
            hintsLoading={insights.hintsLoading}
          />
          <MoveHistory moveLog={moveLog} />
        </aside>
      </main>

      {insights.showFeedback && (
        <MoveFeedback
          evaluation={insights.evaluation}
          prevEvaluation={insights.prevEvaluation}
          trigger={insights.feedbackTrigger}
        />
      )}

      {error && (
        <div className="error-bar">
          {error}
          <button className="error-dismiss" onClick={() => {}}>×</button>
        </div>
      )}

      {gameState.done && (
        <GameModal
          gameState={gameState}
          onRematch={() => handleStart(lastMode, lastSims)}
          onNewGame={handleLeave}
        />
      )}
    </div>
  )
}
