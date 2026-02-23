import { useEffect, useState } from 'react'
import type { EvaluationData } from '../api/client'

interface Props {
  evaluation: EvaluationData | null
  prevEvaluation: EvaluationData | null
  /** Incremented each time the human makes a (non-chain) move */
  trigger: number
}

interface FeedbackInfo {
  label: string
  symbol: string
  color: string
  comment: string
}

const CATEGORIES: { min: number; label: string; symbol: string; color: string; comments: string[] }[] = [
  { min: 0.15,  label: 'Brilliant', symbol: '!!', color: '#2ecc71', comments: [
    'Outstanding move!', 'Engine-level play!', 'Perfectly calculated!', 'Masterful!',
    'You saw that?! Wow.', 'Top-tier thinking.', 'The computer approves.',
    'That was chef\'s kiss.', 'Absolutely clinical.', 'Are you a GM?',
  ]},
  { min: 0.03,  label: 'Good', symbol: '!', color: '#27ae60', comments: [
    'Solid choice.', 'Well played.', 'Strong move.', 'Good instincts.',
    'Clean and effective.', 'Hard to argue with that.', 'Keeps the pressure on.',
    'Textbook stuff.', 'That\'ll do nicely.', 'Right idea, right time.',
  ]},
  { min: -0.03, label: 'Neutral', symbol: '', color: '#95a5a6', comments: [
    'A reasonable move.', 'Nothing wrong with that.', 'Keeps things balanced.', 'Fair enough.',
    'Steady as she goes.', 'Playing it safe.', 'No harm done.',
    'The position holds.', 'A quiet move.', 'Maintaining the balance.',
  ]},
  { min: -0.10, label: 'Inaccuracy', symbol: '?!', color: '#f39c12', comments: [
    'Slightly imprecise.', 'There was a better option.', 'A small slip.', 'Not the best.',
    'Close, but not quite.', 'The engine squints.', 'Room for improvement.',
    'Hmm, almost.', 'A tiny wobble.', 'You can do better.',
  ]},
  { min: -0.25, label: 'Mistake', symbol: '?', color: '#e67e22', comments: [
    'That hurts a bit.', 'A costly error.', 'Your opponent is happy.', 'Missed something.',
    'Oops, that\'s not ideal.', 'The advantage shifts.', 'That one stings.',
    'A painful oversight.', 'Trouble brewing.', 'Not your finest moment.',
  ]},
  { min: -Infinity, label: 'Blunder', symbol: '??', color: '#e74c3c', comments: [
    'Oof, that stings.', 'A serious blunder!', 'Back to the drawing board.', 'That changes everything.',
    'Oh no, oh no, oh no.', 'The wheels came off.', 'Your opponent says thanks.',
    'Disaster strikes!', 'That was rough.', 'We don\'t talk about this one.',
  ]},
]

function classify(delta: number): FeedbackInfo {
  for (const cat of CATEGORIES) {
    if (delta >= cat.min) {
      const comment = cat.comments[Math.floor(Math.random() * cat.comments.length)]
      return { label: cat.label, symbol: cat.symbol, color: cat.color, comment }
    }
  }
  // fallback (should never hit)
  const last = CATEGORIES[CATEGORIES.length - 1]
  return { label: last.label, symbol: last.symbol, color: last.color, comment: last.comments[0] }
}

export function MoveFeedback({ evaluation, prevEvaluation, trigger }: Props) {
  const [feedback, setFeedback] = useState<FeedbackInfo | null>(null)
  const [visible, setVisible] = useState(false)
  const [fading, setFading] = useState(false)

  useEffect(() => {
    if (trigger === 0) return
    if (!evaluation || !prevEvaluation) return

    // Normalize both evaluations to the mover's perspective.
    // The mover is the player who just moved — that's prevEvaluation's current_player.
    // prevEvaluation.value is from prev current_player's perspective (the mover).
    // evaluation.value is from new current_player's perspective (the opponent).
    const moverPrevValue = prevEvaluation.value
    // After the move, current_player flipped, so negate to get mover's perspective:
    const moverNewValue = -evaluation.value

    const delta = moverNewValue - moverPrevValue
    const info = classify(delta)

    setFeedback(info)
    setVisible(true)
    setFading(false)

    const fadeTimer = setTimeout(() => setFading(true), 2500)
    const hideTimer = setTimeout(() => {
      setVisible(false)
      setFading(false)
    }, 3000)

    return () => {
      clearTimeout(fadeTimer)
      clearTimeout(hideTimer)
    }
  }, [trigger]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!visible || !feedback) return null

  return (
    <div
      className={`move-feedback-toast ${fading ? 'toast-out' : 'toast-in'}`}
      style={{ borderLeftColor: feedback.color }}
    >
      <span className="feedback-symbol" style={{ color: feedback.color }}>
        {feedback.symbol || '-'}
      </span>
      <div className="feedback-body">
        <strong>{feedback.label}</strong>
        <span className="feedback-comment">{feedback.comment}</span>
      </div>
    </div>
  )
}
