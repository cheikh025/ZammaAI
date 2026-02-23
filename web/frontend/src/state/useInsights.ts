import { useState, useCallback, useRef } from 'react'
import type { EvaluationData, HintAction } from '../api/client'
import * as api from '../api/client'

export interface InsightsHook {
  showEvalBar: boolean
  showHints: boolean
  showFeedback: boolean
  evaluation: EvaluationData | null
  prevEvaluation: EvaluationData | null
  hints: HintAction[]
  hintsLoading: boolean
  feedbackTrigger: number
  toggleEvalBar: () => void
  toggleHints: () => void
  toggleFeedback: () => void
  requestHints: (sessionId: string) => Promise<void>
  onPositionChange: (sessionId: string, currentPlayer: number, done: boolean, isHumanMove: boolean) => void
  clearInsights: () => void
}

export function useInsights(): InsightsHook {
  const [showEvalBar, setShowEvalBar] = useState(false)
  const [showHints, setShowHints] = useState(false)
  const [showFeedback, setShowFeedback] = useState(false)

  const [evaluation, setEvaluation] = useState<EvaluationData | null>(null)
  const [prevEvaluation, setPrevEvaluation] = useState<EvaluationData | null>(null)
  const [hints, setHints] = useState<HintAction[]>([])
  const [hintsLoading, setHintsLoading] = useState(false)
  const [feedbackTrigger, setFeedbackTrigger] = useState(0)

  // Refs for async callbacks to avoid stale closures
  const showEvalBarRef = useRef(showEvalBar)
  const showFeedbackRef = useRef(showFeedback)
  const evaluationRef = useRef(evaluation)
  showEvalBarRef.current = showEvalBar
  showFeedbackRef.current = showFeedback
  evaluationRef.current = evaluation

  const toggleEvalBar = useCallback(() => setShowEvalBar((v) => !v), [])
  const toggleHints = useCallback(() => {
    setShowHints((v) => {
      if (v) setHints([]) // clear hints when toggling off
      return !v
    })
  }, [])
  const toggleFeedback = useCallback(() => setShowFeedback((v) => !v), [])

  const requestHints = useCallback(async (sessionId: string) => {
    setHintsLoading(true)
    try {
      const data = await api.getHints(sessionId)
      setHints(data.hints)
    } catch {
      setHints([])
    } finally {
      setHintsLoading(false)
    }
  }, [])

  const onPositionChange = useCallback(
    (sessionId: string, _currentPlayer: number, _done: boolean, isHumanMove: boolean) => {
      // Clear previous hints on any position change
      setHints([])

      if (!showEvalBarRef.current && !showFeedbackRef.current) return

      // For human moves with feedback on: save current eval as prev, then fetch new
      // and bump the feedback trigger AFTER the new eval arrives
      if (isHumanMove && showFeedbackRef.current) {
        const snapshot = evaluationRef.current
        setPrevEvaluation(snapshot)

        api.evaluatePosition(sessionId)
          .then((data) => {
            setEvaluation(data)
            // Now both prevEvaluation (snapshot) and evaluation (data) are set —
            // bump trigger so MoveFeedback can compute the delta
            setFeedbackTrigger((t) => t + 1)
          })
          .catch(() => {})
      } else {
        // AI move or no feedback — just update eval
        api.evaluatePosition(sessionId)
          .then((data) => setEvaluation(data))
          .catch(() => {})
      }
    },
    [],
  )

  const clearInsights = useCallback(() => {
    setEvaluation(null)
    setPrevEvaluation(null)
    setHints([])
    setHintsLoading(false)
    setFeedbackTrigger(0)
  }, [])

  return {
    showEvalBar,
    showHints,
    showFeedback,
    evaluation,
    prevEvaluation,
    hints,
    hintsLoading,
    feedbackTrigger,
    toggleEvalBar,
    toggleHints,
    toggleFeedback,
    requestHints,
    onPositionChange,
    clearInsights,
  }
}
