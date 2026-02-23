import type { ReactNode } from 'react'
import type { GameState } from '../types'
import { CELL, PAD, SVG, sqXY } from './boardGeometry'

// ---------------------------------------------------------------------------
// Board geometry (shared constants imported from boardGeometry.ts)
// ---------------------------------------------------------------------------

const PIECE_R = 28    // piece radius
const HIT_R   = PIECE_R + 6  // clickable area radius

// ---------------------------------------------------------------------------
// Precompute board lines (done once at module load)
// ---------------------------------------------------------------------------

interface Line { x1: number; y1: number; x2: number; y2: number; diag: boolean }

function buildLines(): Line[] {
  const lines: Line[] = []
  // Horizontal orthogonal lines (one per row)
  for (let r = 0; r < 5; r++) {
    const [x1, y1] = sqXY(r * 5)
    const [x2, y2] = sqXY(r * 5 + 4)
    lines.push({ x1, y1, x2, y2, diag: false })
  }
  // Vertical orthogonal lines (one per column)
  for (let c = 0; c < 5; c++) {
    const [x1, y1] = sqXY(c)
    const [x2, y2] = sqXY(20 + c)
    lines.push({ x1, y1, x2, y2, diag: false })
  }
  // Diagonal lines: only from even-parity squares (r+c even)
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 5; c++) {
      if ((r + c) % 2 !== 0) continue
      if (c < 4) {
        const [x1, y1] = sqXY(r * 5 + c)
        const [x2, y2] = sqXY((r + 1) * 5 + c + 1)
        lines.push({ x1, y1, x2, y2, diag: true })
      }
      if (c > 0) {
        const [x1, y1] = sqXY(r * 5 + c)
        const [x2, y2] = sqXY((r + 1) * 5 + c - 1)
        lines.push({ x1, y1, x2, y2, diag: true })
      }
    }
  }
  return lines
}

const LINES = buildLines()

// ---------------------------------------------------------------------------
// King crown helper
// ---------------------------------------------------------------------------

/**
 * Build a 3-spike crown polygon points string centred at (cx, cy).
 *
 *        *           ← center spike (tallest)
 *      /   \
 *    *       *       ← side spikes
 *   / \_   _/ \
 *  /    \_/    \
 * |             |    ← base band
 * |_____________|
 */
function crownPoints(cx: number, cy: number, r: number): string {
  const hw  = r * 0.43          // half-width of crown
  const bot = cy + r * 0.27     // flat base bottom
  const base= cy + r * 0.03     // base top / spike roots
  const vly = cy - r * 0.05     // valley between spikes
  const sh  = cy - r * 0.27     // side spike tip
  const ch  = cy - r * 0.42     // center spike tip (tallest)

  const pts: [number, number][] = [
    [cx - hw, bot],              // bottom-left
    [cx - hw, sh],               // left spike tip
    [cx - hw * 0.45, vly],      // left inner valley
    [cx,      ch ],              // center spike tip
    [cx + hw * 0.45, vly],      // right inner valley
    [cx + hw, sh],               // right spike tip
    [cx + hw, bot],              // bottom-right
  ]
  return pts.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ')
}

// ---------------------------------------------------------------------------
// Piece sub-component
// ---------------------------------------------------------------------------

interface PieceProps {
  piece: number   // 1=black-man 2=black-king 3=white-man 4=white-king
  cx: number
  cy: number
  onClick: () => void
}

function Piece({ piece, cx, cy, onClick }: PieceProps) {
  const isBlack = piece === 1 || piece === 2
  const isKing  = piece === 2 || piece === 4

  const fill      = isBlack ? '#1a0f0a' : '#f5ede0'
  const stroke    = isBlack ? '#5a3a28' : '#9e8060'
  const sheen     = isBlack ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.55)'
  const crownFill = isBlack ? '#d4af37' : '#7a5c2e'
  const crownEdge = isBlack ? 'rgba(255,255,255,0.25)' : 'rgba(0,0,0,0.20)'

  return (
    <g onClick={onClick} style={{ cursor: 'pointer' }}>
      {/* Drop shadow */}
      <circle cx={cx + 2} cy={cy + 3} r={PIECE_R} fill="rgba(0,0,0,0.35)" />
      {/* Body */}
      <circle cx={cx} cy={cy} r={PIECE_R} fill={fill} stroke={stroke} strokeWidth={2} />
      {/* Sheen highlight */}
      <circle cx={cx - 9} cy={cy - 9} r={9} fill={sheen} />
      {/* King crown */}
      {isKing && (
        <polygon
          points={crownPoints(cx, cy, PIECE_R)}
          fill={crownFill}
          stroke={crownEdge}
          strokeWidth={0.8}
        />
      )}
    </g>
  )
}

// ---------------------------------------------------------------------------
// Board component
// ---------------------------------------------------------------------------

interface BoardProps {
  gameState: GameState
  selectedSq: number | null
  sources: Set<number>
  dests: Set<number>
  onSquareClick: (sq: number) => void
  hintOverlay?: ReactNode
}

export function Board({ gameState, selectedSq, sources, dests, onSquareClick, hintOverlay }: BoardProps) {
  const { board, chain_piece, done } = gameState

  return (
    <svg
      viewBox={`0 0 ${SVG} ${SVG}`}
      style={{ width: '100%', maxWidth: `${SVG}px`, height: 'auto', display: 'block' }}
      aria-label="Khreibga board"
    >
      {/* Board background */}
      <rect
        x={PAD - 22} y={PAD - 22}
        width={4 * CELL + 44} height={4 * CELL + 44}
        rx={10} ry={10}
        fill="#c8a96e"
        stroke="#8b6914"
        strokeWidth={3}
        filter="url(#board-shadow)"
      />

      <defs>
        <filter id="board-shadow" x="-5%" y="-5%" width="110%" height="115%">
          <feDropShadow dx="0" dy="4" stdDeviation="6" floodOpacity="0.4" />
        </filter>
      </defs>

      {/* Lines */}
      {LINES.map((ln, i) => (
        <line
          key={i}
          x1={ln.x1} y1={ln.y1}
          x2={ln.x2} y2={ln.y2}
          stroke="#5c3d1e"
          strokeWidth={ln.diag ? 1.5 : 2.5}
        />
      ))}

      {/* Intersection dots */}
      {Array.from({ length: 25 }, (_, sq) => {
        const [x, y] = sqXY(sq)
        return <circle key={`dot-${sq}`} cx={x} cy={y} r={4} fill="#5c3d1e" />
      })}

      {/* Highlight rings (behind pieces) */}
      {Array.from({ length: 25 }, (_, sq) => {
        const [x, y] = sqXY(sq)
        const isSel   = sq === selectedSq
        const isChain = sq === chain_piece
        const isSrc   = sources.has(sq) && board[sq] !== 0
        const isDest  = dests.has(sq)

        if (!isSel && !isChain && !isSrc && !isDest) return null

        let fill = 'none'
        let stroke = 'none'
        let sw = 3

        // Priority: selected > chain > destination > source
        if (isSrc)   { fill = 'rgba(76,175,80,0.25)';  stroke = '#4CAF50'; sw = 3 }
        if (isDest)  { fill = 'rgba(255,193,7,0.30)';   stroke = '#FFC107'; sw = 3 }
        if (isChain) { fill = 'rgba(255,87,34,0.25)';   stroke = '#FF5722'; sw = 4 }
        if (isSel)   { fill = 'rgba(33,150,243,0.30)';  stroke = '#2196F3'; sw = 4 }

        return (
          <circle
            key={`hl-${sq}`}
            cx={x} cy={y}
            r={HIT_R}
            fill={fill}
            stroke={stroke}
            strokeWidth={sw}
          />
        )
      })}

      {/* Pieces */}
      {Array.from({ length: 25 }, (_, sq) => {
        const piece = board[sq]
        if (piece === 0) return null
        const [x, y] = sqXY(sq)
        return (
          <Piece
            key={`piece-${sq}`}
            piece={piece}
            cx={x} cy={y}
            onClick={() => !done && onSquareClick(sq)}
          />
        )
      })}

      {/* Invisible click targets on empty legal-destination squares */}
      {Array.from({ length: 25 }, (_, sq) => {
        if (board[sq] !== 0 || !dests.has(sq)) return null
        const [x, y] = sqXY(sq)
        return (
          <circle
            key={`hit-${sq}`}
            cx={x} cy={y}
            r={HIT_R}
            fill="transparent"
            style={{ cursor: 'pointer' }}
            onClick={() => onSquareClick(sq)}
          />
        )
      })}

      {/* Hint arrows overlay */}
      {hintOverlay}
    </svg>
  )
}
