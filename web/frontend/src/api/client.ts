import type { GameState, GameMode } from '../types'

const BASE = '/api'

export async function createGame(mode: GameMode, aiSimulations = 10): Promise<GameState> {
  const res = await fetch(`${BASE}/games`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, ai_simulations: aiSimulations }),
  })
  if (!res.ok) throw new Error(await extractError(res))
  return res.json()
}

export async function getGame(sessionId: string): Promise<GameState> {
  const res = await fetch(`${BASE}/games/${sessionId}`)
  if (!res.ok) throw new Error(await extractError(res))
  return res.json()
}

export async function makeMove(
  sessionId: string,
  src: number,
  dst: number,
): Promise<GameState> {
  const res = await fetch(`${BASE}/games/${sessionId}/move`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ src, dst }),
  })
  if (!res.ok) throw new Error(await extractError(res))
  return res.json()
}

export async function aiMove(sessionId: string): Promise<GameState> {
  const res = await fetch(`${BASE}/games/${sessionId}/ai-move`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error(await extractError(res))
  return res.json()
}

export async function resetGame(sessionId: string): Promise<GameState> {
  const res = await fetch(`${BASE}/games/${sessionId}/reset`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error(await extractError(res))
  return res.json()
}

export async function deleteGame(sessionId: string): Promise<void> {
  await fetch(`${BASE}/games/${sessionId}`, { method: 'DELETE' })
}

async function extractError(res: Response): Promise<string> {
  try {
    const body = await res.json()
    return body.message ?? body.error_code ?? `HTTP ${res.status}`
  } catch {
    return `HTTP ${res.status}`
  }
}
