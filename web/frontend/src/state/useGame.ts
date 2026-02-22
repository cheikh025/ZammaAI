import { useState, useCallback } from 'react'
import type { GameState, GameMode, MoveEntry } from '../types'
import * as api from '../api/client'

// ---------------------------------------------------------------------------
// Pure helpers (no hook deps needed)
// ---------------------------------------------------------------------------

function legalSources(mask: number[]): Set<number> {
  const s = new Set<number>()
  for (let a = 0; a < 625; a++) {
    if (mask[a]) s.add(Math.floor(a / 25))
  }
  return s
}

function legalDests(mask: number[], src: number): Set<number> {
  const s = new Set<number>()
  for (let d = 0; d < 25; d++) {
    if (mask[src * 25 + d]) s.add(d)
  }
  return s
}

function aiPlayer(mode: GameMode): 1 | 2 | null {
  if (mode === 'hvr' || mode === 'hvai') return 2
  if (mode === 'aivh') return 1
  return null
}

function needsAi(gs: GameState, mode: GameMode): boolean {
  const ap = aiPlayer(mode)
  return !gs.done && ap !== null && gs.current_player === ap
}

function entryFromAction(action: number, player: 1 | 2, moveCount: number): MoveEntry {
  return { src: Math.floor(action / 25), dst: action % 25, player, moveCount }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export interface GameHook {
  gameState: GameState | null
  mode: GameMode
  selectedSq: number | null
  sources: Set<number>
  dests: Set<number>
  loading: boolean
  aiThinking: boolean
  error: string | null
  moveLog: MoveEntry[]
  startGame: (mode: GameMode, sims?: number) => Promise<void>
  leaveGame: () => void
  resetGame: () => Promise<void>
  clickSquare: (sq: number) => Promise<void>
}

export function useGame(): GameHook {
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [mode, setMode] = useState<GameMode>('hvh')
  const [selectedSq, setSelectedSq] = useState<number | null>(null)
  const [sources, setSources] = useState<Set<number>>(new Set())
  const [dests, setDests] = useState<Set<number>>(new Set())
  const [loading, setLoading] = useState(false)
  const [aiThinking, setAiThinking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [moveLog, setMoveLog] = useState<MoveEntry[]>([])

  // -------------------------------------------------------------------------
  // After a state update: sync derived UI state and optionally run AI loop
  // -------------------------------------------------------------------------
  const applyState = useCallback(
    async (gs: GameState, m: GameMode, entry?: MoveEntry) => {
      // If chain is active, auto-select the locked piece
      const autoSel = gs.chain_piece !== null ? gs.chain_piece : null
      const autoSrc = legalSources(gs.action_mask)
      const autoDst = autoSel !== null ? legalDests(gs.action_mask, autoSel) : new Set<number>()

      setGameState(gs)
      setSelectedSq(autoSel)
      setSources(autoSrc)
      setDests(autoDst)
      setLoading(false)
      setError(null)
      if (entry) setMoveLog((prev) => [...prev, entry])

      // Run AI loop until it's the human's turn or game ends
      if (needsAi(gs, m)) {
        setAiThinking(true)
        let current = gs
        try {
          do {
            const movedBy = current.current_player
            const next = await api.aiMove(current.session_id)
            // Log each hop the AI makes
            if (next.last_action !== null) {
              const e = entryFromAction(next.last_action, movedBy, next.move_count)
              setMoveLog((prev) => [...prev, e])
            }
            current = next
          } while (needsAi(current, m))

          const finalSrc = legalSources(current.action_mask)
          const finalSel = current.chain_piece
          const finalDst =
            finalSel !== null ? legalDests(current.action_mask, finalSel) : new Set<number>()

          setGameState(current)
          setSelectedSq(finalSel)
          setSources(finalSrc)
          setDests(finalDst)
        } catch (e) {
          setError(String(e))
        } finally {
          setAiThinking(false)
        }
      }
    },
    [],
  )

  // -------------------------------------------------------------------------
  // Public actions
  // -------------------------------------------------------------------------

  const startGame = useCallback(
    async (m: GameMode, sims = 200) => {
      setLoading(true)
      setError(null)
      setMoveLog([])
      setMode(m)
      try {
        const gs = await api.createGame(m, sims)
        await applyState(gs, m)
      } catch (e) {
        setLoading(false)
        setError(String(e))
      }
    },
    [applyState],
  )

  const leaveGame = useCallback(() => {
    setGameState(null)
    setSelectedSq(null)
    setSources(new Set())
    setDests(new Set())
    setMoveLog([])
    setError(null)
  }, [])

  const resetGame = useCallback(async () => {
    if (!gameState) return
    setLoading(true)
    setError(null)
    setMoveLog([])
    try {
      const gs = await api.resetGame(gameState.session_id)
      await applyState(gs, mode)
    } catch (e) {
      setLoading(false)
      setError(String(e))
    }
  }, [gameState, mode, applyState])

  const clickSquare = useCallback(
    async (sq: number) => {
      if (!gameState || gameState.done || loading || aiThinking) return
      if (needsAi(gameState, mode)) return

      // --- Mid-chain: only legal destinations from chain_piece are clickable ---
      if (gameState.chain_piece !== null) {
        if (!dests.has(sq)) return
        setLoading(true)
        try {
          const gs = await api.makeMove(gameState.session_id, gameState.chain_piece, sq)
          const entry = entryFromAction(gameState.chain_piece * 25 + sq, gameState.current_player, gs.move_count)
          await applyState(gs, mode, entry)
        } catch (e) {
          setLoading(false)
          setError(String(e))
        }
        return
      }

      // --- Normal selection flow ---
      if (selectedSq === null) {
        // First click: select a legal source
        if (!sources.has(sq)) return
        setSelectedSq(sq)
        setDests(legalDests(gameState.action_mask, sq))
      } else if (sq === selectedSq) {
        // Click same square: deselect
        setSelectedSq(null)
        setDests(new Set())
      } else if (sources.has(sq) && !dests.has(sq)) {
        // Click another legal source: switch selection
        setSelectedSq(sq)
        setDests(legalDests(gameState.action_mask, sq))
      } else if (dests.has(sq)) {
        // Click a legal destination: execute the move
        setLoading(true)
        try {
          const gs = await api.makeMove(gameState.session_id, selectedSq, sq)
          const entry = entryFromAction(selectedSq * 25 + sq, gameState.current_player, gs.move_count)
          await applyState(gs, mode, entry)
        } catch (e) {
          setLoading(false)
          setError(String(e))
        }
      }
      // else: illegal destination — keep selection, no feedback needed
    },
    [gameState, mode, selectedSq, sources, dests, loading, aiThinking, applyState],
  )

  return {
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
  }
}
