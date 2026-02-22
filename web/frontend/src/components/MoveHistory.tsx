import { useEffect, useRef } from 'react'
import type { MoveEntry } from '../types'

function sqAlg(sq: number): string {
  const r = Math.floor(sq / 5)
  const c = sq % 5
  return 'abcde'[c] + (r + 1)
}

interface Props {
  moveLog: MoveEntry[]
}

export function MoveHistory({ moveLog }: Props) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [moveLog.length])

  return (
    <div className="move-history">
      <h2 className="panel-title">Move History</h2>

      {moveLog.length === 0 ? (
        <div className="history-empty">No moves yet</div>
      ) : (
        <div className="history-list">
          {moveLog.map((entry, i) => (
            <div key={i} className="history-row">
              <span className="history-num">{i + 1}.</span>
              <span
                className="history-dot"
                style={{
                  background: entry.player === 1 ? '#1a0f0a' : '#f5ede0',
                  border: entry.player === 2 ? '1.5px solid #9e8060' : 'none',
                }}
              />
              <span className="history-move">
                {sqAlg(entry.src)}→{sqAlg(entry.dst)}
              </span>
            </div>
          ))}
          <div ref={endRef} />
        </div>
      )}
    </div>
  )
}
