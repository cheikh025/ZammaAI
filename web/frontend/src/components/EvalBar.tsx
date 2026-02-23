import type { EvaluationData } from '../api/client'

interface Props {
  evaluation: EvaluationData | null
}

export function EvalBar({ evaluation }: Props) {
  // Default to 50/50 when no evaluation yet
  let blackPct = 50

  if (evaluation) {
    // Convert to Black's perspective: positive = good for Black
    const blackValue = evaluation.current_player === 1
      ? evaluation.value
      : -evaluation.value
    // Map [-1, +1] to [0, 100], clamp to [2, 98]
    blackPct = Math.min(98, Math.max(2, (blackValue + 1) * 50))
  }

  const whitePct = 100 - blackPct

  // Bar layout: White on top (matching board: White's home = top rows),
  // Black on bottom (Black's home = bottom rows)
  return (
    <div className="eval-bar">
      <div
        className="eval-bar-white"
        style={{ height: `${whitePct}%` }}
      >
        {whitePct >= 10 && (
          <span className="eval-label">{Math.round(whitePct)}%</span>
        )}
      </div>
      <div
        className="eval-bar-black"
        style={{ height: `${blackPct}%` }}
      >
        {blackPct >= 10 && (
          <span className="eval-label">{Math.round(blackPct)}%</span>
        )}
      </div>
    </div>
  )
}
