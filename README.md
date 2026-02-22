# Khreibga Zero

AlphaZero-style reinforcement learning agent for **Khreibaga** — a traditional Mauritanian board game played on a 5×5 Alquerque grid.

---

## Requirements

- Python 3.10+
- Node.js 18+ and npm (for the web interface)

---

## Installation

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **GPU (optional):** If you have a CUDA GPU, install the matching PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) for faster training and MCTS inference.

### 2. Install frontend dependencies

```bash
cd web/frontend
npm install
cd ../..
```

---

## Running the Web Interface

The game is played through a browser — a FastAPI backend serves the game logic and a React frontend renders the board.

### Development mode (two terminals)

**Terminal 1 — backend:**
```bash
python -m uvicorn web.backend.app.main:app --port 8000 --reload
```

**Terminal 2 — frontend:**
```bash
cd web/frontend
npm run dev
```

Then open **http://localhost:5173** in your browser.

### Production mode (single terminal)

Build the frontend once, then the backend serves everything.

> **Note:** `npm run build` must be run from inside `web/frontend/`, not the repo root.

```bash
# Step 1 — build the frontend assets
cd web/frontend
npm run build
cd ../..

# Step 2 — serve everything from the backend
python -m uvicorn web.backend.app.main:app --port 8000
```

Then open **http://localhost:8000** in your browser.

---

## Game Modes

| Mode | Description |
|------|-------------|
| Human vs Human | Two players on the same machine |
| Human vs Random AI | Play against a random-move opponent |
| Human vs MCTS AI | Play against the neural network (you move first) |
| MCTS AI vs Human | Play against the neural network (AI moves first) |

The **MCTS Simulations** slider controls AI strength — higher values are stronger but slower (default: 200).

---

## Training the AI

Train a new model from scratch:

```bash
python train.py
```

Resume from a checkpoint:

```bash
python train.py --checkpoint-in checkpoints/run_001_last.pt
```

Common options:

| Flag | Default | Description |
|------|---------|-------------|
| `--num-iterations` | 200 | Total training iterations |
| `--games-per-iteration` | 25 | Self-play games per iteration |
| `--num-simulations` | 200 | MCTS simulations per move |
| `--device` | auto | `auto`, `cpu`, `cuda`, or `mps` |
| `--checkpoint-out` | `checkpoints/run_001_last.pt` | Where to save the model |
| `--enable-tensorboard` | off | Log metrics to TensorBoard |

Checkpoints are saved to the `checkpoints/` folder and picked up automatically by the web backend when you start the server.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

320 tests covering the game engine, RL interface, neural network, and MCTS.

---

## Project Structure

```
ZammaAI/
├── khreibga/           # Core Python package
│   ├── board.py        # Board representation
│   ├── game.py         # Game state and rules
│   ├── move_gen.py     # Move generation
│   ├── captures.py     # Capture logic
│   ├── encoder.py      # (7,5,5) tensor encoder
│   ├── env.py          # Gym-like RL environment
│   ├── model.py        # KhreibagaNet (ResNet policy + value)
│   ├── mcts.py         # Monte Carlo Tree Search
│   ├── self_play.py    # Self-play data generation
│   └── trainer.py      # AlphaZero training loop
├── web/
│   ├── backend/        # FastAPI server
│   └── frontend/       # React + TypeScript + Vite UI
├── tests/              # Pytest test suite
├── train.py            # Training entrypoint
├── requirements.txt    # Python dependencies
└── README.md
```
