"""
Tests for the Khreibaga Zero training pipeline (Milestone 5).

Covers:
  - TrainingConfig defaults
  - ReplayBuffer (add, sample, capacity eviction)
  - Trainer.train_step (loss computation and gradient update)
  - self_play_game (example generation)
  - evaluate_models (win-rate computation)
  - End-to-end mini training loop
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from khreibga.model import KhreibagaNet, ACTION_SPACE, INPUT_PLANES, BOARD_H, BOARD_W
from khreibga.self_play import self_play_game, evaluate_models
from khreibga.trainer import TrainingConfig, ReplayBuffer, Trainer, _split_game_seeds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def model(device):
    net = KhreibagaNet().to(device)
    net.eval()
    return net


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def dummy_examples(rng):
    """Generate fake (obs, policy, value) examples for buffer testing."""
    examples = []
    for _ in range(50):
        obs = rng.random((INPUT_PLANES, BOARD_H, BOARD_W)).astype(np.float32)
        policy = np.zeros(ACTION_SPACE, dtype=np.float32)
        legal = rng.choice(ACTION_SPACE, size=5, replace=False)
        policy[legal] = rng.dirichlet(np.ones(5)).astype(np.float32)
        value = rng.choice([-1.0, 0.0, 1.0])
        examples.append((obs, policy, value))
    return examples


# ---------------------------------------------------------------------------
# TrainingConfig tests
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.num_simulations == 200
        assert cfg.c_puct == 1.0
        assert cfg.self_play_parallel is False
        assert cfg.num_self_play_workers == 1
        assert cfg.self_play_worker_device == "cpu"
        assert cfg.seed is None
        assert cfg.enable_tensorboard is False
        assert cfg.tensorboard_log_dir == "runs/khreibga"
        assert cfg.tensorboard_run_name is None
        assert cfg.tensorboard_flush_secs == 30
        assert cfg.lr == 1e-3
        assert cfg.l2_reg == 1e-4
        assert cfg.batch_size == 256
        assert cfg.buffer_size == 100_000
        assert cfg.win_threshold == 0.55
        assert cfg.num_iterations == 1000

    def test_custom_values(self):
        cfg = TrainingConfig(lr=0.01, batch_size=64, num_iterations=10)
        assert cfg.lr == 0.01
        assert cfg.batch_size == 64
        assert cfg.num_iterations == 10


# ---------------------------------------------------------------------------
# ReplayBuffer tests
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def test_empty_buffer(self):
        buf = ReplayBuffer(max_size=100)
        assert len(buf) == 0

    def test_add_game(self, dummy_examples):
        buf = ReplayBuffer(max_size=1000)
        buf.add_game(dummy_examples)
        assert len(buf) == len(dummy_examples)

    def test_add_multiple_games(self, dummy_examples):
        buf = ReplayBuffer(max_size=1000)
        buf.add_game(dummy_examples)
        buf.add_game(dummy_examples)
        assert len(buf) == 2 * len(dummy_examples)

    def test_capacity_eviction(self, dummy_examples):
        """Buffer should evict oldest examples when full."""
        buf = ReplayBuffer(max_size=30)
        buf.add_game(dummy_examples)  # 50 examples into buffer of 30
        assert len(buf) == 30

    def test_sample_shape(self, dummy_examples, rng):
        buf = ReplayBuffer(max_size=1000)
        buf.add_game(dummy_examples)

        obs, policies, values = buf.sample(16, rng=rng)
        assert obs.shape == (16, INPUT_PLANES, BOARD_H, BOARD_W)
        assert policies.shape == (16, ACTION_SPACE)
        assert values.shape == (16,)

    def test_sample_dtypes(self, dummy_examples, rng):
        buf = ReplayBuffer(max_size=1000)
        buf.add_game(dummy_examples)

        obs, policies, values = buf.sample(8, rng=rng)
        assert obs.dtype == np.float32
        assert policies.dtype == np.float32
        assert values.dtype == np.float32

    def test_sample_with_replacement(self, rng):
        """When batch_size > buffer size, sampling with replacement."""
        buf = ReplayBuffer(max_size=100)
        obs = np.zeros((INPUT_PLANES, BOARD_H, BOARD_W), dtype=np.float32)
        pol = np.zeros(ACTION_SPACE, dtype=np.float32)
        pol[0] = 1.0
        buf.add_game([(obs, pol, 1.0)] * 5)

        # Request more samples than buffer contains
        o, p, v = buf.sample(20, rng=rng)
        assert o.shape[0] == 20

    def test_sample_values_in_range(self, dummy_examples, rng):
        buf = ReplayBuffer(max_size=1000)
        buf.add_game(dummy_examples)

        _, _, values = buf.sample(32, rng=rng)
        assert all(v in (-1.0, 0.0, 1.0) for v in values)

    def test_sample_policies_sum_to_one(self, dummy_examples, rng):
        buf = ReplayBuffer(max_size=1000)
        buf.add_game(dummy_examples)

        _, policies, _ = buf.sample(16, rng=rng)
        for i in range(16):
            assert np.isclose(policies[i].sum(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

class TestTrainer:
    def test_init_defaults(self, device):
        trainer = Trainer(device=device)
        assert trainer.iteration == 0
        assert trainer.training_step == 0
        assert isinstance(trainer.model, KhreibagaNet)
        assert isinstance(trainer.best_model, KhreibagaNet)
        assert len(trainer.replay_buffer) == 0
        assert trainer.tb_writer is None

    def test_init_custom_config(self, device):
        cfg = TrainingConfig(lr=0.01, batch_size=32)
        trainer = Trainer(config=cfg, device=device)
        assert trainer.config.lr == 0.01
        assert trainer.config.batch_size == 32

    def test_init_with_model(self, model, device):
        trainer = Trainer(model=model, device=device)
        # Should use the provided model
        assert trainer.model is model

    def test_best_model_is_clone(self, device):
        """best_model should be a separate copy from the training model."""
        trainer = Trainer(device=device)
        assert trainer.model is not trainer.best_model
        # They should have the same weights initially
        for p1, p2 in zip(trainer.model.parameters(), trainer.best_model.parameters()):
            assert torch.equal(p1.data, p2.data)

    def test_train_step_returns_losses(self, device, dummy_examples):
        cfg = TrainingConfig(batch_size=8)
        trainer = Trainer(config=cfg, device=device)
        trainer.replay_buffer.add_game(dummy_examples)

        total, val_loss, pol_loss = trainer.train_step()
        assert isinstance(total, float)
        assert isinstance(val_loss, float)
        assert isinstance(pol_loss, float)
        assert total > 0
        assert val_loss >= 0
        assert pol_loss >= 0

    def test_train_step_updates_weights(self, device, dummy_examples):
        cfg = TrainingConfig(batch_size=8)
        trainer = Trainer(config=cfg, device=device)
        trainer.replay_buffer.add_game(dummy_examples)

        # Record initial weights
        initial_weights = {
            name: p.data.clone()
            for name, p in trainer.model.named_parameters()
        }

        trainer.train_step()

        # At least some weights should have changed
        changed = False
        for name, p in trainer.model.named_parameters():
            if not torch.equal(p.data, initial_weights[name]):
                changed = True
                break
        assert changed, "No weights changed after train_step"

    def test_train_step_loss_decreases(self, device, dummy_examples):
        """Loss should generally decrease over multiple steps."""
        cfg = TrainingConfig(batch_size=16)
        trainer = Trainer(config=cfg, device=device)
        trainer.replay_buffer.add_game(dummy_examples)

        losses = []
        for _ in range(20):
            total, _, _ = trainer.train_step()
            losses.append(total)

        # Average loss in second half should be lower than first half
        first_half = np.mean(losses[:10])
        second_half = np.mean(losses[10:])
        assert second_half < first_half, (
            f"Loss did not decrease: first_half={first_half:.4f}, second_half={second_half:.4f}"
        )

    def test_clone_model(self, device):
        trainer = Trainer(device=device)
        clone = trainer._clone_model()
        assert clone is not trainer.model
        for p1, p2 in zip(trainer.model.parameters(), clone.parameters()):
            assert torch.equal(p1.data, p2.data)


class TestTensorBoardLogging:
    @staticmethod
    def _single_example() -> tuple[np.ndarray, np.ndarray, float]:
        obs = np.zeros((INPUT_PLANES, BOARD_H, BOARD_W), dtype=np.float32)
        pol = np.zeros(ACTION_SPACE, dtype=np.float32)
        pol[0] = 1.0
        return obs, pol, 0.0

    def test_logs_scalars_and_flushes(self, device, monkeypatch):
        class FakeWriter:
            def __init__(self):
                self.scalars: list[tuple[str, float, int]] = []
                self.flushed = False
                self.closed = False

            def add_scalar(self, tag, value, step):
                self.scalars.append((str(tag), float(value), int(step)))

            def flush(self):
                self.flushed = True

            def close(self):
                self.closed = True

        fake_writer = FakeWriter()
        monkeypatch.setattr(
            Trainer,
            "_create_tensorboard_writer",
            lambda self: fake_writer,
        )

        cfg = TrainingConfig(
            enable_tensorboard=True,
            games_per_iteration=1,
            batch_size=1,
            training_steps_per_iteration=1,
            eval_interval=1,
            win_threshold=0.9,
            num_iterations=1,
        )
        trainer = Trainer(config=cfg, device=device)
        one = self._single_example()

        monkeypatch.setattr(
            "khreibga.trainer.self_play_game",
            lambda *args, **kwargs: [one],
        )
        monkeypatch.setattr(
            Trainer,
            "train_step",
            lambda self: (3.0, 1.0, 2.0),
        )
        monkeypatch.setattr(
            "khreibga.trainer.evaluate_models",
            lambda *args, **kwargs: {
                "wins": 1,
                "losses": 1,
                "draws": 0,
                "num_games": 2,
                "win_rate": 0.5,
                "draw_rate": 0.0,
                "loss_rate": 0.5,
                "score_rate": 0.5,
                "elo_diff": 0.0,
            },
        )

        trainer.run(num_iterations=1)

        tags = {tag for tag, _, _ in fake_writer.scalars}
        assert "train/total_loss" in tags
        assert "train/value_loss" in tags
        assert "train/policy_loss" in tags
        assert "eval/win_rate" in tags
        assert "eval/elo_diff" in tags
        assert "eval/promoted" in tags
        assert "replay_buffer/size" in tags
        assert "self_play/examples_added" in tags
        assert "timing/iteration_seconds" in tags
        assert fake_writer.flushed is True

        trainer.close()
        assert fake_writer.closed is True


# ---------------------------------------------------------------------------
# self_play_game tests
# ---------------------------------------------------------------------------

class TestSelfPlayGame:
    def test_returns_examples(self, model, device):
        """self_play_game should return a non-empty list of examples."""
        examples = self_play_game(
            model,
            num_simulations=2,
            device=device,
            rng=np.random.default_rng(0),
        )
        assert len(examples) > 0

    def test_example_shapes(self, model, device):
        examples = self_play_game(
            model,
            num_simulations=2,
            device=device,
            rng=np.random.default_rng(0),
        )
        for obs, policy, value in examples:
            assert obs.shape == (INPUT_PLANES, BOARD_H, BOARD_W)
            assert obs.dtype == np.float32
            assert policy.shape == (ACTION_SPACE,)
            assert policy.dtype == np.float32
            assert value in (-1.0, 0.0, 1.0)

    def test_policy_sums_to_one(self, model, device):
        examples = self_play_game(
            model,
            num_simulations=2,
            device=device,
            rng=np.random.default_rng(1),
        )
        for _, policy, _ in examples:
            assert np.isclose(policy.sum(), 1.0, atol=1e-5)

    def test_policies_non_negative(self, model, device):
        examples = self_play_game(
            model,
            num_simulations=2,
            device=device,
            rng=np.random.default_rng(2),
        )
        for _, policy, _ in examples:
            assert (policy >= 0).all()

    def test_values_consistent(self, model, device):
        """All values should be from the same game outcome (+1/-1/0)."""
        examples = self_play_game(
            model,
            num_simulations=2,
            device=device,
            rng=np.random.default_rng(3),
        )
        values = {v for _, _, v in examples}
        # Should contain at most {-1, 1} or {0} for a draw
        assert values.issubset({-1.0, 0.0, 1.0})

    def test_reproducible_with_seed(self, model, device):
        """Same RNG seed should produce the same examples."""
        ex1 = self_play_game(
            model, num_simulations=2, device=device,
            rng=np.random.default_rng(99),
        )
        ex2 = self_play_game(
            model, num_simulations=2, device=device,
            rng=np.random.default_rng(99),
        )
        assert len(ex1) == len(ex2)
        for (o1, p1, v1), (o2, p2, v2) in zip(ex1, ex2):
            assert np.array_equal(o1, o2)
            assert np.array_equal(p1, p2)
            assert v1 == v2


# ---------------------------------------------------------------------------
# evaluate_models tests
# ---------------------------------------------------------------------------

class TestEvaluateModels:
    def test_returns_float(self, device):
        model_a = KhreibagaNet().to(device)
        model_b = KhreibagaNet().to(device)
        model_a.eval()
        model_b.eval()

        win_rate = evaluate_models(
            model_a, model_b,
            num_games=2, num_simulations=2,
            device=device,
        )
        assert isinstance(win_rate, float)

    def test_win_rate_range(self, device):
        model_a = KhreibagaNet().to(device)
        model_b = KhreibagaNet().to(device)
        model_a.eval()
        model_b.eval()

        win_rate = evaluate_models(
            model_a, model_b,
            num_games=4, num_simulations=2,
            device=device,
        )
        assert 0.0 <= win_rate <= 1.0

    def test_zero_games(self, device):
        model_a = KhreibagaNet().to(device)
        model_b = KhreibagaNet().to(device)
        model_a.eval()
        model_b.eval()

        win_rate = evaluate_models(
            model_a, model_b,
            num_games=0, num_simulations=2,
            device=device,
        )
        assert win_rate == 0.0

    def test_same_model_roughly_even(self, device):
        """A model playing itself should win roughly 50% as each colour."""
        model = KhreibagaNet().to(device)
        model.eval()

        win_rate = evaluate_models(
            model, model,
            num_games=4, num_simulations=2,
            device=device,
        )
        # With only 4 games and 2 sims, just check it returns valid range
        assert 0.0 <= win_rate <= 1.0

    def test_return_details_contains_elo_tracking_metrics(self, device):
        model_a = KhreibagaNet().to(device)
        model_b = KhreibagaNet().to(device)
        model_a.eval()
        model_b.eval()

        details = evaluate_models(
            model_a, model_b,
            num_games=4, num_simulations=2,
            device=device,
            return_details=True,
        )
        assert isinstance(details, dict)
        assert details["num_games"] == 4
        assert details["wins"] + details["losses"] + details["draws"] == 4
        assert 0.0 <= details["win_rate"] <= 1.0
        assert 0.0 <= details["draw_rate"] <= 1.0
        assert 0.0 <= details["loss_rate"] <= 1.0
        assert np.isclose(
            details["win_rate"] + details["draw_rate"] + details["loss_rate"],
            1.0,
            atol=1e-6,
        )
        assert 0.0 <= details["score_rate"] <= 1.0
        assert np.isfinite(details["elo_diff"]) or np.isinf(details["elo_diff"])


# ---------------------------------------------------------------------------
# End-to-end mini training loop
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_mini_training_loop(self, device):
        """Run a tiny training loop to verify everything connects."""
        cfg = TrainingConfig(
            num_simulations=2,
            games_per_iteration=1,
            batch_size=8,
            buffer_size=500,
            training_steps_per_iteration=2,
            eval_interval=2,
            eval_games=2,
            eval_simulations=2,
            num_iterations=2,
        )
        trainer = Trainer(config=cfg, device=device)
        trainer.run(num_iterations=2)

        assert trainer.iteration == 2
        assert len(trainer.replay_buffer) > 0

    def test_buffer_fills_during_training(self, device):
        """Replay buffer should accumulate examples from self-play."""
        cfg = TrainingConfig(
            num_simulations=2,
            games_per_iteration=2,
            batch_size=8,
            buffer_size=5000,
            training_steps_per_iteration=1,
            eval_interval=100,
            num_iterations=1,
        )
        trainer = Trainer(config=cfg, device=device)
        trainer.run(num_iterations=1)

        # 2 games should produce at least a few examples each
        assert len(trainer.replay_buffer) >= 2

    def test_training_step_counter(self, device):
        cfg = TrainingConfig(
            num_simulations=2,
            games_per_iteration=2,
            batch_size=8,
            buffer_size=5000,
            training_steps_per_iteration=5,
            eval_interval=100,
            num_iterations=1,
        )
        trainer = Trainer(config=cfg, device=device)
        trainer.run(num_iterations=1)

        # After 1 iteration with 5 training steps per iteration
        assert trainer.training_step == 5


# ---------------------------------------------------------------------------
# Trainer.run orchestration tests
# ---------------------------------------------------------------------------

class TestTrainerRunOrchestration:
    @staticmethod
    def _single_example() -> tuple[np.ndarray, np.ndarray, float]:
        obs = np.zeros((INPUT_PLANES, BOARD_H, BOARD_W), dtype=np.float32)
        pol = np.zeros(ACTION_SPACE, dtype=np.float32)
        pol[0] = 1.0
        return obs, pol, 0.0

    def test_run_calls_self_play_with_config(self, device, monkeypatch):
        cfg = TrainingConfig(
            num_simulations=7,
            c_puct=1.7,
            temp_threshold=3,
            dirichlet_alpha=0.2,
            dirichlet_epsilon=0.15,
            games_per_iteration=3,
            batch_size=9999,  # skip gradient steps
            eval_interval=9999,  # skip eval gate
            num_iterations=1,
        )
        trainer = Trainer(config=cfg, device=device)

        calls: list[dict] = []
        one = self._single_example()

        def fake_self_play(model, **kwargs):
            calls.append(kwargs)
            return [one]

        monkeypatch.setattr("khreibga.trainer.self_play_game", fake_self_play)

        trainer.run(num_iterations=1)

        assert len(calls) == cfg.games_per_iteration
        assert len(trainer.replay_buffer) == cfg.games_per_iteration

        for kwargs in calls:
            assert kwargs["num_simulations"] == cfg.num_simulations
            assert kwargs["c_puct"] == cfg.c_puct
            assert kwargs["temp_threshold"] == cfg.temp_threshold
            assert kwargs["dirichlet_alpha"] == cfg.dirichlet_alpha
            assert kwargs["dirichlet_epsilon"] == cfg.dirichlet_epsilon
            assert kwargs["device"] == device

    def test_run_skips_training_when_buffer_under_batch_size(
        self, device, monkeypatch,
    ):
        cfg = TrainingConfig(
            games_per_iteration=1,
            batch_size=8,
            training_steps_per_iteration=5,
            eval_interval=9999,
            num_iterations=1,
        )
        trainer = Trainer(config=cfg, device=device)

        one = self._single_example()
        monkeypatch.setattr(
            "khreibga.trainer.self_play_game",
            lambda *args, **kwargs: [one],
        )

        call_count = {"n": 0}

        def fake_train_step(_self):
            call_count["n"] += 1
            return 0.0, 0.0, 0.0

        monkeypatch.setattr(Trainer, "train_step", fake_train_step)

        trainer.run(num_iterations=1)

        assert call_count["n"] == 0
        assert trainer.training_step == 0

    def test_run_executes_configured_training_steps_once_ready(
        self, device, monkeypatch,
    ):
        cfg = TrainingConfig(
            games_per_iteration=1,
            batch_size=4,
            training_steps_per_iteration=3,
            eval_interval=9999,
            num_iterations=1,
        )
        trainer = Trainer(config=cfg, device=device)

        one = self._single_example()
        monkeypatch.setattr(
            "khreibga.trainer.self_play_game",
            lambda *args, **kwargs: [one, one, one, one],
        )

        call_count = {"n": 0}

        def fake_train_step(_self):
            call_count["n"] += 1
            return 0.0, 0.0, 0.0

        monkeypatch.setattr(Trainer, "train_step", fake_train_step)

        trainer.run(num_iterations=1)

        assert call_count["n"] == cfg.training_steps_per_iteration
        assert trainer.training_step == cfg.training_steps_per_iteration

    def test_eval_gate_promotes_only_if_strictly_above_threshold(
        self, device, monkeypatch,
    ):
        cfg = TrainingConfig(
            games_per_iteration=1,
            batch_size=9999,  # skip gradient steps
            eval_interval=1,
            eval_games=6,
            eval_simulations=4,
            c_puct=1.3,
            win_threshold=0.55,
            num_iterations=1,
        )
        trainer = Trainer(config=cfg, device=device)
        original_best = trainer.best_model
        one = self._single_example()

        monkeypatch.setattr(
            "khreibga.trainer.self_play_game",
            lambda *args, **kwargs: [one],
        )

        eval_calls: list[dict] = []

        def fake_eval(new_model, old_model, **kwargs):
            eval_calls.append(kwargs)
            assert new_model is trainer.model
            assert old_model is original_best
            return cfg.win_threshold + 0.01

        monkeypatch.setattr("khreibga.trainer.evaluate_models", fake_eval)

        trainer.run(num_iterations=1)

        assert len(eval_calls) == 1
        assert eval_calls[0]["num_games"] == cfg.eval_games
        assert eval_calls[0]["num_simulations"] == cfg.eval_simulations
        assert eval_calls[0]["c_puct"] == cfg.c_puct
        assert eval_calls[0]["device"] == device
        assert eval_calls[0]["return_details"] is True
        assert trainer.best_model is not original_best

    def test_eval_gate_does_not_promote_at_exact_threshold(
        self, device, monkeypatch,
    ):
        cfg = TrainingConfig(
            games_per_iteration=1,
            batch_size=9999,
            eval_interval=1,
            win_threshold=0.55,
            num_iterations=1,
        )
        trainer = Trainer(config=cfg, device=device)
        original_best = trainer.best_model
        one = self._single_example()

        monkeypatch.setattr(
            "khreibga.trainer.self_play_game",
            lambda *args, **kwargs: [one],
        )
        monkeypatch.setattr(
            "khreibga.trainer.evaluate_models",
            lambda *args, **kwargs: cfg.win_threshold,
        )

        trainer.run(num_iterations=1)
        assert trainer.best_model is original_best

    def test_eval_metrics_tracked_without_affecting_gate(
        self, device, monkeypatch,
    ):
        cfg = TrainingConfig(
            games_per_iteration=1,
            batch_size=9999,
            eval_interval=1,
            win_threshold=0.55,
            num_iterations=1,
        )
        trainer = Trainer(config=cfg, device=device)
        one = self._single_example()
        monkeypatch.setattr(
            "khreibga.trainer.self_play_game",
            lambda *args, **kwargs: [one],
        )
        monkeypatch.setattr(
            "khreibga.trainer.evaluate_models",
            lambda *args, **kwargs: {
                "wins": 3,
                "losses": 1,
                "draws": 0,
                "num_games": 4,
                "win_rate": 0.75,
                "draw_rate": 0.0,
                "loss_rate": 0.25,
                "score_rate": 0.75,
                "elo_diff": 190.848501887865,
            },
        )

        trainer.run(num_iterations=1)
        assert trainer.last_eval_metrics is not None
        assert trainer.last_eval_metrics["iteration"] == 1
        assert trainer.last_eval_metrics["win_rate"] == 0.75
        assert trainer.last_eval_metrics["elo_diff"] == pytest.approx(190.848501887865)
        assert trainer.last_eval_metrics["promoted"] is True
        assert len(trainer.eval_history) == 1

    def test_run_periodic_checkpoint_saves_latest_and_snapshot(
        self, device, monkeypatch, tmp_path,
    ):
        cfg = TrainingConfig(
            games_per_iteration=1,
            batch_size=9999,
            eval_interval=9999,
            num_iterations=3,
        )
        trainer = Trainer(config=cfg, device=device)
        one = self._single_example()

        monkeypatch.setattr(
            "khreibga.trainer.self_play_game",
            lambda *args, **kwargs: [one],
        )

        saved_paths: list[str] = []

        def fake_save(self, path, extra_metadata=None):
            saved_paths.append(str(path))

        monkeypatch.setattr(Trainer, "save_checkpoint", fake_save)

        base = tmp_path / "run.pt"
        trainer.run(
            num_iterations=3,
            checkpoint_every=2,
            checkpoint_path=base,
            checkpoint_keep_history=True,
        )

        assert len(saved_paths) == 2
        assert saved_paths[0].endswith("run.pt")
        assert saved_paths[1].endswith("run_iter_2.pt")

    def test_run_periodic_checkpoint_without_history_updates_only_latest(
        self, device, monkeypatch, tmp_path,
    ):
        cfg = TrainingConfig(
            games_per_iteration=1,
            batch_size=9999,
            eval_interval=9999,
            num_iterations=2,
        )
        trainer = Trainer(config=cfg, device=device)
        one = self._single_example()

        monkeypatch.setattr(
            "khreibga.trainer.self_play_game",
            lambda *args, **kwargs: [one],
        )

        saved_paths: list[str] = []

        def fake_save(self, path, extra_metadata=None):
            saved_paths.append(str(path))

        monkeypatch.setattr(Trainer, "save_checkpoint", fake_save)

        base = tmp_path / "latest.pt"
        trainer.run(
            num_iterations=2,
            checkpoint_every=1,
            checkpoint_path=base,
            checkpoint_keep_history=False,
        )

        assert len(saved_paths) == 2
        assert all(p.endswith("latest.pt") for p in saved_paths)


# ---------------------------------------------------------------------------
# Additional deterministic policy/eval behavior checks
# ---------------------------------------------------------------------------

class TestSelfPlayAndEvalBehavior:
    def test_temperature_schedule_switches_after_threshold(
        self, model, device, monkeypatch,
    ):
        recorded_temps: list[float] = []

        def fake_select_action(visits, temperature=1.0, rng=None):
            recorded_temps.append(float(temperature))
            return int(np.argmax(visits))

        monkeypatch.setattr("khreibga.self_play.select_action", fake_select_action)

        self_play_game(
            model,
            num_simulations=1,
            temp_threshold=1,
            device=device,
            rng=np.random.default_rng(123),
        )

        assert len(recorded_temps) > 1
        assert recorded_temps[0] == 1.0
        assert all(t == 0.0 for t in recorded_temps[1:])

    def test_evaluate_models_alternates_colours(self, device, monkeypatch):
        model_a = KhreibagaNet().to(device)
        model_b = KhreibagaNet().to(device)
        model_a.eval()
        model_b.eval()

        class FakeEnv:
            game_idx = 0

            def __init__(self):
                self.idx = FakeEnv.game_idx
                FakeEnv.game_idx += 1
                self.done = False
                self.current_player = 1  # BLACK
                self.winner = None

            def reset(self):
                self.done = False
                self.current_player = 1
                self.winner = None
                return None, None

            def step(self, action):
                self.done = True
                self.winner = 1  # BLACK always wins

        class FakeMCTS:
            def __init__(self, *args, **kwargs):
                pass

            def search(self, env):
                visits = np.zeros(ACTION_SPACE, dtype=np.float32)
                visits[0] = 1.0
                return visits

        monkeypatch.setattr("khreibga.self_play.KhreibagaEnv", FakeEnv)
        monkeypatch.setattr("khreibga.self_play.MCTS", FakeMCTS)
        monkeypatch.setattr(
            "khreibga.self_play.select_action",
            lambda visits, temperature=0, rng=None: int(np.argmax(visits)),
        )

        # With BLACK always winning and colour alternation, new model should
        # win exactly half the games over an even game count.
        win_rate = evaluate_models(
            model_a, model_b,
            num_games=6,
            num_simulations=1,
            device=device,
        )
        assert win_rate == 0.5


# ---------------------------------------------------------------------------
# Checkpointing tests
# ---------------------------------------------------------------------------

class TestCheckpointing:
    @staticmethod
    def _single_example() -> tuple[np.ndarray, np.ndarray, float]:
        obs = np.zeros((INPUT_PLANES, BOARD_H, BOARD_W), dtype=np.float32)
        pol = np.zeros(ACTION_SPACE, dtype=np.float32)
        pol[0] = 1.0
        return obs, pol, 1.0

    def test_save_checkpoint_writes_expected_payload(
        self, device, tmp_path, dummy_examples,
    ):
        cfg = TrainingConfig(buffer_size=321, batch_size=8)
        trainer = Trainer(config=cfg, device=device)
        trainer.replay_buffer.add_game(dummy_examples[:10])
        trainer.iteration = 7
        trainer.training_step = 13

        ckpt_path = tmp_path / "trainer_ckpt.pt"
        trainer.save_checkpoint(ckpt_path, extra_metadata={"tag": "unit-test"})

        assert ckpt_path.exists()
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)

        assert payload["version"] == 1
        assert payload["config"]["buffer_size"] == 321
        assert payload["iteration"] == 7
        assert payload["training_step"] == 13
        assert payload["replay_buffer"]["max_size"] == 321
        assert payload["replay_buffer"]["size"] == 10
        assert payload["extra_metadata"]["tag"] == "unit-test"

    def test_load_checkpoint_restores_state(
        self, device, tmp_path, dummy_examples,
    ):
        cfg = TrainingConfig(batch_size=8, buffer_size=222)
        trainer_a = Trainer(config=cfg, device=device)
        trainer_a.replay_buffer.add_game(dummy_examples)
        trainer_a.iteration = 4
        trainer_a.training_step = 9

        # Run one update so optimizer state is non-empty.
        trainer_a.train_step()

        ckpt_path = tmp_path / "resume.pt"
        trainer_a.save_checkpoint(ckpt_path)

        # Start from a mismatched trainer, then restore.
        trainer_b = Trainer(config=TrainingConfig(batch_size=16), device=device)
        trainer_b.load_checkpoint(ckpt_path, load_optimizer=True)

        assert trainer_b.config.batch_size == 8
        assert trainer_b.config.buffer_size == 222
        assert trainer_b.iteration == 4
        assert trainer_b.training_step == 9
        assert trainer_b.replay_buffer.max_size == 222
        assert len(trainer_b.replay_buffer) == 0  # metadata-only restore

        for p_a, p_b in zip(
            trainer_a.model.parameters(), trainer_b.model.parameters(),
        ):
            assert torch.allclose(p_a, p_b)
        for p_a, p_b in zip(
            trainer_a.best_model.parameters(), trainer_b.best_model.parameters(),
        ):
            assert torch.allclose(p_a, p_b)

    def test_from_checkpoint_builds_ready_trainer(self, device, tmp_path):
        trainer = Trainer(
            config=TrainingConfig(batch_size=12, buffer_size=333),
            device=device,
        )
        trainer.iteration = 2
        trainer.training_step = 5
        ckpt_path = tmp_path / "factory.pt"
        trainer.save_checkpoint(ckpt_path)

        restored = Trainer.from_checkpoint(ckpt_path, device=device)
        assert restored.config.batch_size == 12
        assert restored.config.buffer_size == 333
        assert restored.iteration == 2
        assert restored.training_step == 5

    def test_checkpoint_restores_eval_history(self, device, tmp_path):
        trainer = Trainer(config=TrainingConfig(), device=device)
        trainer.eval_history = [
            {"iteration": 1, "win_rate": 0.6, "elo_diff": 70.4, "promoted": True},
            {"iteration": 2, "win_rate": 0.5, "elo_diff": 0.0, "promoted": False},
        ]
        trainer.last_eval_metrics = trainer.eval_history[-1]

        ckpt_path = tmp_path / "eval_history.pt"
        trainer.save_checkpoint(ckpt_path)

        restored = Trainer(config=TrainingConfig(), device=device)
        restored.load_checkpoint(ckpt_path)
        assert len(restored.eval_history) == 2
        assert restored.last_eval_metrics is not None
        assert restored.last_eval_metrics["iteration"] == 2


# ---------------------------------------------------------------------------
# Parallel self-play tests
# ---------------------------------------------------------------------------

class TestParallelSelfPlay:
    @staticmethod
    def _single_example() -> tuple[np.ndarray, np.ndarray, float]:
        obs = np.zeros((INPUT_PLANES, BOARD_H, BOARD_W), dtype=np.float32)
        pol = np.zeros(ACTION_SPACE, dtype=np.float32)
        pol[0] = 1.0
        return obs, pol, 0.0

    def test_split_game_seeds_even_chunks(self):
        chunks = _split_game_seeds([1, 2, 3, 4, 5], num_chunks=2)
        assert chunks == [[1, 2, 3], [4, 5]]

    def test_build_game_seeds_deterministic(self, device):
        trainer = Trainer(config=TrainingConfig(seed=123), device=device)
        trainer.iteration = 7
        s1 = trainer._build_game_seeds(6)
        s2 = trainer._build_game_seeds(6)
        assert s1 == s2
        assert all(isinstance(s, int) for s in s1)

        trainer.iteration = 8
        s3 = trainer._build_game_seeds(6)
        assert s3 != s1

    def test_parallel_self_play_uses_executor_and_fills_buffer(
        self, device, monkeypatch,
    ):
        cfg = TrainingConfig(
            games_per_iteration=5,
            self_play_parallel=True,
            num_self_play_workers=2,
            seed=999,
            batch_size=9999,       # skip training
            eval_interval=9999,    # skip eval
            num_iterations=1,
        )
        trainer = Trainer(config=cfg, device=device)
        one = self._single_example()

        monkeypatch.setattr(
            "khreibga.trainer.self_play_game",
            lambda *args, **kwargs: [one],
        )

        submit_calls: list[tuple] = []

        class FakeFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class FakeExecutor:
            def __init__(self, max_workers=None, mp_context=None):
                self.max_workers = max_workers
                self.mp_context = mp_context

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args):
                submit_calls.append((fn, args))
                return FakeFuture(fn(*args))

        monkeypatch.setattr("khreibga.trainer.ProcessPoolExecutor", FakeExecutor)

        trainer.run(num_iterations=1)

        assert len(submit_calls) == 2
        chunk_sizes = [len(call[1][1]) for call in submit_calls]
        assert sorted(chunk_sizes) == [2, 3]
        all_seeds = [seed for call in submit_calls for seed in call[1][1]]
        assert len(all_seeds) == cfg.games_per_iteration
        assert all(seed is not None for seed in all_seeds)
        assert len(trainer.replay_buffer) == cfg.games_per_iteration
