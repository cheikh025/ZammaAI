"""
AlphaZero training pipeline for Khreibaga Zero.

Provides:
  - ``TrainingConfig`` : all hyper-parameters in one place.
  - ``ReplayBuffer``   : stores (obs, policy, value) examples for sampling.
  - ``Trainer``        : orchestrates the self-play / train / evaluate cycle.

Loss (from spec Section 3.3)::

    l = (z - v)^2  -  pi^T log(p)  +  c ||theta||^2
          value          policy           L2 (weight_decay)
"""

from __future__ import annotations

import copy
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from khreibga.model import KhreibagaNet, get_device, ACTION_SPACE
from khreibga.self_play import self_play_game, evaluate_models


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All hyper-parameters for the training pipeline."""

    # ---- Self-play ----
    num_simulations: int = 200
    c_puct: float = 1.0
    temp_threshold: int = 10
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    games_per_iteration: int = 25
    self_play_parallel: bool = False
    num_self_play_workers: int = 1
    self_play_worker_device: str = "cpu"
    seed: int | None = None

    # ---- Training ----
    lr: float = 1e-3
    l2_reg: float = 1e-4
    batch_size: int = 256
    buffer_size: int = 100_000
    training_steps_per_iteration: int = 100

    # ---- Evaluation gate ----
    eval_interval: int = 50      # evaluate every N iterations
    eval_games: int = 50
    eval_simulations: int = 200
    win_threshold: float = 0.55

    # ---- Outer loop ----
    num_iterations: int = 1000


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

def _split_game_seeds(
    game_seeds: list[int | None],
    num_chunks: int,
) -> list[list[int | None]]:
    """Split per-game seeds into near-even ordered chunks."""
    if num_chunks <= 0:
        return []
    n = len(game_seeds)
    if n == 0:
        return []

    base = n // num_chunks
    rem = n % num_chunks
    chunks: list[list[int | None]] = []
    start = 0
    for i in range(num_chunks):
        size = base + (1 if i < rem else 0)
        if size <= 0:
            continue
        end = start + size
        chunks.append(game_seeds[start:end])
        start = end
    return chunks


def _self_play_worker(
    model_state_dict: dict[str, torch.Tensor],
    game_seeds: list[int | None],
    num_simulations: int,
    c_puct: float,
    temp_threshold: int,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    worker_device: str,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Run one self-play worker job (picklable for spawn-based multiprocessing)."""
    device = torch.device(worker_device)

    model = KhreibagaNet().to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    all_examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    for seed in game_seeds:
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        examples = self_play_game(
            model,
            num_simulations=num_simulations,
            c_puct=c_puct,
            temp_threshold=temp_threshold,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            device=device,
            rng=rng,
        )
        all_examples.extend(examples)

    return all_examples


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity ring buffer of ``(obs, policy, value)`` examples.

    Parameters
    ----------
    max_size : int
        Maximum number of examples stored.  Oldest examples are evicted
        when the buffer is full.
    """

    def __init__(self, max_size: int = 100_000) -> None:
        self.buffer: deque[tuple[np.ndarray, np.ndarray, float]] = deque(
            maxlen=max_size,
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def add_game(
        self,
        examples: list[tuple[np.ndarray, np.ndarray, float]],
    ) -> None:
        """Append all examples from a single game."""
        for ex in examples:
            self.buffer.append(ex)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random mini-batch.

        Parameters
        ----------
        batch_size : int
            Number of examples to draw.  If the buffer is smaller,
            sampling is done *with* replacement.
        rng : np.random.Generator or None

        Returns
        -------
        obs      : ``(B, 7, 5, 5)`` float32
        policies : ``(B, 625)``     float32
        values   : ``(B,)``         float32
        """
        if rng is None:
            rng = np.random.default_rng()

        n = len(self.buffer)
        replace = n < batch_size
        indices = rng.choice(n, size=batch_size, replace=replace)

        obs_list, pol_list, val_list = [], [], []
        for i in indices:
            o, p, v = self.buffer[i]
            obs_list.append(o)
            pol_list.append(p)
            val_list.append(v)

        return (
            np.stack(obs_list),
            np.stack(pol_list),
            np.array(val_list, dtype=np.float32),
        )

    @property
    def max_size(self) -> int:
        """Configured maximum capacity."""
        return int(self.buffer.maxlen or 0)

    def metadata(self) -> dict[str, int]:
        """Return lightweight serialization metadata."""
        return {
            "max_size": self.max_size,
            "size": len(self.buffer),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates the AlphaZero self-play / train / evaluate cycle.

    Parameters
    ----------
    config : TrainingConfig
        Hyper-parameters.
    device : torch.device or None
        Target device for model and tensors.
    model : KhreibagaNet or None
        Starting model.  If ``None``, a fresh one is created.
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        device: torch.device | None = None,
        model: KhreibagaNet | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.device = device or get_device()

        # Current model (being trained).
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = KhreibagaNet().to(self.device)

        # Best model (used for self-play data generation).
        self.best_model = self._clone_model()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.l2_reg,
        )

        self.replay_buffer = ReplayBuffer(max_size=self.config.buffer_size)

        # Counters
        self.iteration: int = 0
        self.training_step: int = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, num_iterations: int | None = None) -> None:
        """Execute the full training loop.

        Parameters
        ----------
        num_iterations : int or None
            Override ``config.num_iterations`` if given.
        """
        n = num_iterations or self.config.num_iterations
        cfg = self.config

        for _ in range(n):
            self.iteration += 1

            # ---- 1. Self-play ----
            self._run_self_play_iteration()

            # ---- 2. Training ----
            if len(self.replay_buffer) >= cfg.batch_size:
                for _ in range(cfg.training_steps_per_iteration):
                    self.train_step()
                    self.training_step += 1

            # ---- 3. Evaluation gate ----
            if self.iteration % cfg.eval_interval == 0:
                self.model.eval()
                self.best_model.eval()
                win_rate = evaluate_models(
                    self.model,
                    self.best_model,
                    num_games=cfg.eval_games,
                    num_simulations=cfg.eval_simulations,
                    c_puct=cfg.c_puct,
                    device=self.device,
                )
                if win_rate > cfg.win_threshold:
                    self.best_model = self._clone_model()

    def _run_self_play_iteration(self) -> None:
        """Generate self-play data (sequential or process-parallel)."""
        cfg = self.config
        self.best_model.eval()

        game_seeds = self._build_game_seeds(cfg.games_per_iteration)

        use_parallel = (
            cfg.self_play_parallel
            and cfg.num_self_play_workers > 1
            and cfg.games_per_iteration > 1
        )

        if use_parallel:
            examples = self._generate_self_play_parallel(game_seeds)
            self.replay_buffer.add_game(examples)
            return

        for seed in game_seeds:
            rng = np.random.default_rng(seed) if seed is not None else None
            examples = self_play_game(
                self.best_model,
                num_simulations=cfg.num_simulations,
                c_puct=cfg.c_puct,
                temp_threshold=cfg.temp_threshold,
                dirichlet_alpha=cfg.dirichlet_alpha,
                dirichlet_epsilon=cfg.dirichlet_epsilon,
                device=self.device,
                rng=rng,
            )
            self.replay_buffer.add_game(examples)

    def _build_game_seeds(self, num_games: int) -> list[int | None]:
        """Create deterministic per-game seeds for the current iteration."""
        if num_games <= 0:
            return []
        if self.config.seed is None:
            return [None] * num_games

        seed_seq = np.random.SeedSequence([self.config.seed, self.iteration])
        children = seed_seq.spawn(num_games)
        return [
            int(child.generate_state(1, dtype=np.uint64)[0])
            for child in children
        ]

    def _best_model_state_dict_cpu(self) -> dict[str, torch.Tensor]:
        """Return a CPU copy of best-model weights for process hand-off."""
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.best_model.state_dict().items()
        }

    def _generate_self_play_parallel(
        self,
        game_seeds: list[int | None],
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Run self-play in a process pool and return aggregated examples."""
        cfg = self.config
        if not game_seeds:
            return []

        max_workers = max(1, min(cfg.num_self_play_workers, len(game_seeds)))
        seed_chunks = _split_game_seeds(game_seeds, max_workers)
        if not seed_chunks:
            return []

        model_state_dict = self._best_model_state_dict_cpu()
        ctx = mp.get_context("spawn")

        all_examples: list[tuple[np.ndarray, np.ndarray, float]] = []
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(
                    _self_play_worker,
                    model_state_dict,
                    chunk,
                    cfg.num_simulations,
                    cfg.c_puct,
                    cfg.temp_threshold,
                    cfg.dirichlet_alpha,
                    cfg.dirichlet_epsilon,
                    cfg.self_play_worker_device,
                )
                for chunk in seed_chunks
            ]
            # Consume futures in submission order for deterministic aggregation.
            for fut in futures:
                all_examples.extend(fut.result())

        return all_examples

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(self) -> tuple[float, float, float]:
        """Run one gradient update on a mini-batch.

        Returns
        -------
        total_loss, value_loss, policy_loss : float
        """
        self.model.train()
        cfg = self.config

        obs, policies, values = self.replay_buffer.sample(cfg.batch_size)

        obs_t = torch.from_numpy(obs).to(self.device)
        pol_t = torch.from_numpy(policies).to(self.device)
        val_t = torch.from_numpy(values).unsqueeze(1).to(self.device)  # (B,1)

        policy_logits, pred_values = self.model(obs_t)

        # Value loss: MSE
        loss_v = F.mse_loss(pred_values, val_t)

        # Policy loss: cross-entropy  -π^T log(p)
        log_probs = F.log_softmax(policy_logits, dim=1)
        loss_p = -(pol_t * log_probs).sum(dim=1).mean()

        # L2 regularisation is handled by optimizer weight_decay.
        loss = loss_v + loss_p

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_v.item(), loss_p.item()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clone_model(self) -> KhreibagaNet:
        """Deep-copy the current model for use as the best model."""
        clone = KhreibagaNet().to(self.device)
        clone.load_state_dict(copy.deepcopy(self.model.state_dict()))
        clone.eval()
        return clone

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        extra_metadata: dict | None = None,
    ) -> None:
        """Save a training checkpoint.

        Saved content:
          - model / best_model weights
          - optimizer state
          - counters (iteration, training_step)
          - TrainingConfig values
          - replay-buffer metadata (size, max_size)
          - optional user metadata
        """
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "version": 1,
            "config": vars(self.config).copy(),
            "model_state_dict": self.model.state_dict(),
            "best_model_state_dict": self.best_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "training_step": self.training_step,
            "replay_buffer": self.replay_buffer.metadata(),
            "device": str(self.device),
        }
        if extra_metadata is not None:
            checkpoint["extra_metadata"] = extra_metadata

        torch.save(checkpoint, ckpt_path)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizer: bool = True,
    ) -> dict:
        """Load a checkpoint into this trainer instance.

        Notes
        -----
        - Config values from the checkpoint are applied to this instance.
        - Replay-buffer metadata is restored as capacity; examples are not.

        Returns
        -------
        dict
            The raw checkpoint dict.
        """
        ckpt_path = Path(path)
        checkpoint = torch.load(
            ckpt_path,
            map_location=self.device,
            weights_only=False,
        )

        # Restore config first so optimizer recreation uses checkpoint params.
        cfg_data = checkpoint.get("config", {})
        self.config = TrainingConfig(**cfg_data)

        # Ensure models exist and load weights.
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_model.load_state_dict(checkpoint["best_model_state_dict"])
        self.model.to(self.device)
        self.best_model.to(self.device)
        self.best_model.eval()

        # Recreate optimizer from restored config, then optionally restore
        # full optimizer state (momenta, etc.).
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.l2_reg,
        )
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.iteration = int(checkpoint.get("iteration", 0))
        self.training_step = int(checkpoint.get("training_step", 0))

        rb_meta = checkpoint.get("replay_buffer", {})
        rb_max_size = int(rb_meta.get("max_size", self.config.buffer_size))
        self.replay_buffer = ReplayBuffer(max_size=rb_max_size)

        return checkpoint

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | None = None,
        *,
        load_optimizer: bool = True,
    ) -> Trainer:
        """Create a new trainer instance from a checkpoint file."""
        # Build from checkpoint config first, then load exact states.
        checkpoint = torch.load(
            Path(path),
            map_location=device or get_device(),
            weights_only=False,
        )
        cfg_data = checkpoint.get("config", {})
        trainer = cls(config=TrainingConfig(**cfg_data), device=device)
        trainer.load_checkpoint(path, load_optimizer=load_optimizer)
        return trainer
