import type { HintAction } from '../api/client'
import { sqXY } from './boardGeometry'

interface Props {
  hints: HintAction[]
}

const RANK_STYLES = [
  { color: '#4CAF50', opacity: 1.0, width: 6 },    // rank 1: green
  { color: '#FFC107', opacity: 0.65, width: 4 },    // rank 2: yellow
  { color: '#2196F3', opacity: 0.45, width: 3 },    // rank 3: blue
]

const SHORTEN = 20 // shorten arrows by 20px on each end

export function HintOverlay({ hints }: Props) {
  if (hints.length === 0) return null

  return (
    <g>
      <defs>
        {RANK_STYLES.map((s, i) => (
          <marker
            key={`arrow-${i}`}
            id={`hint-arrow-${i}`}
            markerWidth="8"
            markerHeight="6"
            refX="7"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0, 8 3, 0 6" fill={s.color} />
          </marker>
        ))}
      </defs>
      {hints.map((hint, i) => {
        const style = RANK_STYLES[i] ?? RANK_STYLES[2]
        const [x1, y1] = sqXY(hint.src)
        const [x2, y2] = sqXY(hint.dst)

        // Shorten line on each end
        const dx = x2 - x1
        const dy = y2 - y1
        const len = Math.sqrt(dx * dx + dy * dy)
        if (len === 0) return null
        const ux = dx / len
        const uy = dy / len
        const sx = x1 + ux * SHORTEN
        const sy = y1 + uy * SHORTEN
        const ex = x2 - ux * SHORTEN
        const ey = y2 - uy * SHORTEN

        return (
          <line
            key={`hint-${i}`}
            x1={sx} y1={sy}
            x2={ex} y2={ey}
            stroke={style.color}
            strokeWidth={style.width}
            strokeOpacity={style.opacity}
            strokeLinecap="round"
            markerEnd={`url(#hint-arrow-${i})`}
          />
        )
      })}
    </g>
  )
}
