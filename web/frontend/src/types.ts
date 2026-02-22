/** Raw payload returned by all game API endpoints. */
export interface GameState {
  session_id: string
  /** Flat array of 25 ints. 0=empty 1=black-man 2=black-king 3=white-man 4=white-king */
  board: number[]
  /** 1=BLACK 2=WHITE */
  current_player: 1 | 2
  /** Square index 0-24 of the piece mid-chain, or null. */
  chain_piece: number | null
  done: boolean
  /** 1=BLACK won, 2=WHITE won, null=draw or ongoing */
  winner: 1 | 2 | null
  terminal_reason: 'elimination' | 'stalemate' | 'fifty_move_rule' | 'threefold_repetition' | null
  move_count: number
  half_move_clock: number
  /** 625-element binary array. action = src*25+dst */
  action_mask: number[]
  /** Last action played as index, or null on game start. */
  last_action: number | null
  legal_actions_count: number
}

export type GameMode = 'hvh' | 'hvr' | 'hvai'

export interface MoveEntry {
  src: number
  dst: number
  player: 1 | 2
  moveCount: number
}
