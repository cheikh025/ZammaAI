# Khreibaga Zero -- Project Plan

## Game Summary

**Khreibaga** is Mauritanian Alquerque played on a **5x5 grid** (25 intersections). Each player starts with **12 pieces**, leaving only the center point empty. The board has orthogonal connections everywhere, but **diagonal connections only where `(row + col)` is even**.

### Key Rules

| Rule | Detail |
|---|---|
| **Men movement** | Forward/forward-diagonal only; backward allowed *only* during captures |
| **Kings ("Flying Kings")** | Move/capture any distance along a clear line (orthogonal or diagonal) |
| **Promotion** | Man stops on opponent's back rank (not mid-chain) |
| **Compulsory capture** | Must capture if possible; **majority rule** forces the longest chain |
| **Immediate removal** | Captured pieces removed instantly (enables revisiting squares) |
| **Terminal** | Elimination, stalemate (loss), 50-move rule (draw), 3-fold repetition (draw) |

### RL Architecture at a Glance

- **Input:** `5x5x7` tensor (4 piece planes + repetition + colour + move count)
- **Network:** Conv block + 6 ResBlocks (64 filters) + Policy head (625 logits) + Value head (tanh)
- **Action space:** 25x25 = 625 source-target pairs, one atomic hop per NN query
- **Training:** AlphaZero self-play, 200 MCTS sims/move, temperature schedule

---

## Milestones

### Milestone 1 -- Game Engine (Foundation)

Everything else depends on a bug-free rule engine.

- [ ] **1. Board representation & adjacency graph** -- Build the 25-node graph with orthogonal + conditional diagonal edges. Hard-code or compute the `(r+c) % 2 == 0` diagonal rule.
- [ ] **2. Game state class** -- Board array (empty/black-man/black-king/white-man/white-king), current player, mid-chain tracking piece, move counter, state history hash set.
- [ ] **3. Simple move generator** -- Men forward moves, King flying moves along clear lines.
- [ ] **4. Capture move generator** -- Recursive DFS for capture chains (men: adjacent leap; kings: long leap with variable landing). Track captured pieces per chain to enforce immediate removal & re-crossing.
- [ ] **5. Majority capture rule** -- Enumerate all chains, keep only max-length, expose first-hop mask.
- [ ] **6. Turn / chain state machine** -- After each atomic hop, check for continuation; if yes, lock the active piece and stay on the same player; if no, switch player, check promotion.
- [ ] **7. Terminal detection** -- Elimination, stalemate, 50-move rule, 3-fold repetition (Zobrist hashing).
- [ ] **8. Unit tests** -- Edge cases: flying king branching, mid-chain promotion denial, majority tie-breaking, cyclic draw, capture re-crossing.

### Milestone 2 -- RL Interface Layer

Bridge between the engine and the neural network.

- [ ] **9. Canonical state encoder** -- Convert board to `5x5x7` tensor (flip board for White so current player always moves "up").
- [ ] **10. Action mask generator** -- `GetActionMask()` returning a 625-dim binary vector per the spec's Algorithm 1.
- [ ] **11. `step()` / `reset()` API** -- Gym-like interface: `reset() -> obs, mask`; `step(action) -> obs, mask, reward, done`.

### Milestone 3 -- Neural Network

- [ ] **12. Model architecture** -- Conv block + 6 ResBlocks + Policy head (625 logits) + Value head (tanh scalar), all in PyTorch.
- [ ] **13. Canonical input pipeline** -- Batched tensor conversion, GPU transfer.

### Milestone 4 -- MCTS

- [ ] **14. Core MCTS** -- UCB selection, expansion with NN evaluation, backpropagation.
- [ ] **15. Multi-step capture handling** -- During simulation, chain hops are internal environment steps (not full MCTS expansions); the NN is queried only at true decision points.
- [ ] **16. Action masking integration** -- Illegal actions get `-inf` prior, zero visit probability.

### Milestone 5 -- Training Pipeline

- [ ] **17. Self-play worker** -- Generate games using current best model (200 sims/move, temperature schedule: τ=1 for first 10 moves, τ→0 after).
- [ ] **18. Replay buffer & optimizer** -- Store `(state, π, z)` tuples; combined MSE + cross-entropy + L2 loss; Adam optimizer.
- [ ] **19. Evaluation gate** -- Every 50 steps, pit new vs. old over 50 games; promote if win rate > 55%.
- [ ] **20. Cycle / draw defenses** -- Repetition plane in input, Zobrist hash check, 50-move counter normalization.

### Milestone 6 -- Polish & Tooling

- [ ] **21. CLI / visualization** -- Text-based board renderer, human-vs-AI play mode.
- [ ] **22. Logging & checkpoints** -- TensorBoard metrics, model serialization.
- [ ] **23. Performance profiling** -- Bottleneck the move generator (likely the hot path), consider C extension if needed.

---

## Prioritized To-Do List

| Priority | Task | Why |
|---|---|---|
| **P0** | Adjacency graph + board state | Everything depends on correct topology |
| **P0** | Capture chain generator + majority rule | The hardest logic; blocks all RL work |
| **P0** | Turn state machine (atomic hop protocol) | Required for correct MCTS interaction |
| **P0** | Terminal detection | Games must end correctly |
| **P0** | Exhaustive unit tests for engine | A single rule bug corrupts all training |
| **P1** | State encoder (5x5x7 tensor) + action mask | NN can't train without these |
| **P1** | Gym-like `step`/`reset` API | Standard interface for self-play |
| **P1** | Neural network (ResNet + dual heads) | Needed before MCTS |
| **P1** | MCTS with multi-step chain support | Core of AlphaZero |
| **P2** | Self-play loop + replay buffer + training | Actual learning |
| **P2** | Evaluation gate (new vs. old model) | Quality control for training |
| **P3** | CLI visualization / human play mode | Nice-to-have for debugging & demos |
| **P3** | Performance optimization | Only if self-play is too slow |

---

## Tech Stack

- **Language:** Python 3.11+
- **ML Framework:** PyTorch
- **Numerical:** NumPy
- **Optional:** Cython/C++ for move generator if speed becomes a bottleneck
