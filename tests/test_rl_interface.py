"""
Tests for the RL interface layer (Milestone 2).

Validates:
  - khreibga.encoder  -- canonical NumPy state encoder
  - khreibga.env      -- Gymnasium-compatible environment wrapper

These modules are being developed by Teammates 1 and 2 respectively.
The tests here serve as an executable contract for both interfaces.
"""

from __future__ import annotations

import numpy as np
import pytest

from khreibga.board import (
    BLACK,
    BLACK_KING,
    BLACK_MAN,
    BOARD_SIZE,
    EMPTY,
    NUM_SQUARES,
    WHITE,
    WHITE_KING,
    WHITE_MAN,
    initial_board,
    rc_to_sq,
    sq_to_rc,
)
from khreibga.game import ACTION_SPACE, GameState, compute_zobrist_hash

from khreibga.encoder import encode_observation, encode_action_mask
from khreibga.env import KhreibagaEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(src: int, dst: int) -> int:
    """Encode a (src, dst) pair as a flat action index."""
    return src * NUM_SQUARES + dst


def _random_legal_action(mask, rng=None):
    """Pick a random legal action from a float32 action mask."""
    legal = np.nonzero(mask)[0]
    assert len(legal) > 0, "No legal actions in mask"
    if rng is None:
        rng = np.random.default_rng(42)
    return int(rng.choice(legal))


# ===================================================================
# A. Encoder Tests
# ===================================================================

class TestEncoder:
    """Tests for khreibga.encoder (encode_observation, encode_action_mask)."""

    # -- encode_observation ------------------------------------------------

    def test_encode_observation_shape(self):
        """Output tensor has shape (7, 5, 5)."""
        gs = GameState()
        obs = encode_observation(
            gs.board, gs.current_player, gs.history,
            gs.current_hash, gs.move_count,
        )
        assert obs.shape == (7, 5, 5)

    def test_encode_observation_dtype(self):
        """Output tensor has dtype float32."""
        gs = GameState()
        obs = encode_observation(
            gs.board, gs.current_player, gs.history,
            gs.current_hash, gs.move_count,
        )
        assert obs.dtype == np.float32

    def test_encode_observation_initial_men_count(self):
        """Plane 0 has 12 ones (current player men) and plane 2 has 12 ones
        (opponent men) at the start."""
        gs = GameState()
        obs = encode_observation(
            gs.board, gs.current_player, gs.history,
            gs.current_hash, gs.move_count,
        )
        # Plane 0: current player's men (BLACK men at start)
        assert obs[0].sum() == pytest.approx(12.0)
        # Plane 2: opponent's men (WHITE men at start)
        assert obs[2].sum() == pytest.approx(12.0)

    def test_encode_observation_no_kings_initially(self):
        """Planes 1 and 3 are all zeros at the start (no kings)."""
        gs = GameState()
        obs = encode_observation(
            gs.board, gs.current_player, gs.history,
            gs.current_hash, gs.move_count,
        )
        np.testing.assert_array_equal(obs[1], np.zeros((5, 5), dtype=np.float32))
        np.testing.assert_array_equal(obs[3], np.zeros((5, 5), dtype=np.float32))

    def test_encode_observation_repetition_plane_no_repeat(self):
        """Plane 4 is all zeros when the position has been seen only once."""
        gs = GameState()
        obs = encode_observation(
            gs.board, gs.current_player, gs.history,
            gs.current_hash, gs.move_count,
        )
        np.testing.assert_array_equal(obs[4], np.zeros((5, 5), dtype=np.float32))

    def test_encode_observation_repetition_plane_with_repeat(self):
        """Plane 4 is all ones when the current hash count > 1."""
        gs = GameState()
        gs.history[gs.current_hash] = 2  # simulate a repeat
        obs = encode_observation(
            gs.board, gs.current_player, gs.history,
            gs.current_hash, gs.move_count,
        )
        np.testing.assert_array_equal(obs[4], np.ones((5, 5), dtype=np.float32))

    def test_encode_observation_colour_plane_black(self):
        """Plane 5 is all ones when current player is BLACK."""
        gs = GameState()
        assert gs.current_player == BLACK
        obs = encode_observation(
            gs.board, gs.current_player, gs.history,
            gs.current_hash, gs.move_count,
        )
        np.testing.assert_array_equal(obs[5], np.ones((5, 5), dtype=np.float32))

    def test_encode_observation_colour_plane_white(self):
        """Plane 5 is all zeros when current player is WHITE."""
        gs = GameState()
        gs.current_player = WHITE
        gs.current_hash = compute_zobrist_hash(gs.board, WHITE)
        gs.history = {gs.current_hash: 1}
        obs = encode_observation(
            gs.board, gs.current_player, gs.history,
            gs.current_hash, gs.move_count,
        )
        np.testing.assert_array_equal(obs[5], np.zeros((5, 5), dtype=np.float32))

    def test_encode_observation_move_count_plane(self):
        """Plane 6 = move_count / 200.  At move_count=100 every cell is 0.5."""
        gs = GameState()
        gs.move_count = 100
        obs = encode_observation(
            gs.board, gs.current_player, gs.history,
            gs.current_hash, gs.move_count,
        )
        expected = np.full((5, 5), 100.0 / 200.0, dtype=np.float32)
        np.testing.assert_allclose(obs[6], expected)

    def test_encode_observation_move_count_capped(self):
        """Plane 6 is capped at 1.0 when move_count exceeds max_steps."""
        gs = GameState()
        gs.move_count = 999
        obs = encode_observation(
            gs.board, gs.current_player, gs.history,
            gs.current_hash, gs.move_count,
        )
        np.testing.assert_array_equal(obs[6], np.ones((5, 5), dtype=np.float32))

    def test_encode_observation_canonical_flip_for_white(self):
        """When WHITE is the current player the board is flipped (4-r, 4-c).

        Place WHITE_MAN at (4,4)=sq24.  After the canonical flip it should
        appear in plane 0 (current player's men) at observation position (0,0).
        """
        board = [EMPTY] * NUM_SQUARES
        board[rc_to_sq(4, 4)] = WHITE_MAN  # sq 24
        board[rc_to_sq(0, 0)] = BLACK_MAN  # sq 0 -- opponent
        current_player = WHITE
        h = compute_zobrist_hash(board, current_player)
        history = {h: 1}

        obs = encode_observation(board, current_player, history, h, 0)

        # WHITE_MAN at (4,4) -> flipped to (0,0) -> plane 0 (current player's men)
        assert obs[0][0][0] == pytest.approx(1.0)
        # BLACK_MAN at (0,0) -> flipped to (4,4) -> plane 2 (opponent's men)
        assert obs[2][4][4] == pytest.approx(1.0)

    # -- encode_action_mask ------------------------------------------------

    def test_encode_action_mask_shape(self):
        """Output has shape (625,)."""
        mask_list = [0] * ACTION_SPACE
        mask_list[0] = 1
        arr = encode_action_mask(mask_list)
        assert arr.shape == (625,)

    def test_encode_action_mask_dtype(self):
        """Output has dtype float32."""
        mask_list = [0] * ACTION_SPACE
        arr = encode_action_mask(mask_list)
        assert arr.dtype == np.float32

    def test_encode_action_mask_values(self):
        """The float array faithfully reflects the input list of 0s and 1s."""
        mask_list = [0] * ACTION_SPACE
        mask_list[7] = 1
        mask_list[42] = 1
        mask_list[624] = 1
        arr = encode_action_mask(mask_list)

        assert arr[7] == pytest.approx(1.0)
        assert arr[42] == pytest.approx(1.0)
        assert arr[624] == pytest.approx(1.0)
        assert arr[0] == pytest.approx(0.0)
        assert arr[100] == pytest.approx(0.0)
        assert arr.sum() == pytest.approx(3.0)


# ===================================================================
# B. Environment Tests
# ===================================================================

class TestEnv:
    """Tests for khreibga.env.KhreibagaEnv."""

    # -- reset -------------------------------------------------------------

    def test_reset_returns_tuple(self):
        """reset() returns a (obs, info) tuple."""
        env = KhreibagaEnv()
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_obs_shape_dtype(self):
        """Observation from reset() is (7,5,5) float32."""
        env = KhreibagaEnv()
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (7, 5, 5)
        assert obs.dtype == np.float32

    def test_reset_info_has_action_mask(self):
        """info dict from reset() contains 'action_mask' with shape (625,)
        float32."""
        env = KhreibagaEnv()
        _, info = env.reset()
        assert "action_mask" in info
        mask = info["action_mask"]
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (625,)
        assert mask.dtype == np.float32

    def test_reset_has_legal_moves(self):
        """After reset, action_mask has at least one legal move."""
        env = KhreibagaEnv()
        _, info = env.reset()
        assert info["action_mask"].sum() >= 1.0

    # -- step --------------------------------------------------------------

    def test_step_returns_five_tuple(self):
        """step() returns (obs, reward, terminated, truncated, info)."""
        env = KhreibagaEnv()
        obs, info = env.reset()
        action = _random_legal_action(info["action_mask"])
        result = env.step(action)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_obs_shape_dtype(self):
        """Observation from step() is (7,5,5) float32."""
        env = KhreibagaEnv()
        obs, info = env.reset()
        action = _random_legal_action(info["action_mask"])
        obs2, _, _, _, _ = env.step(action)
        assert isinstance(obs2, np.ndarray)
        assert obs2.shape == (7, 5, 5)
        assert obs2.dtype == np.float32

    def test_step_reward_is_float(self):
        """Reward from step() is a float."""
        env = KhreibagaEnv()
        _, info = env.reset()
        action = _random_legal_action(info["action_mask"])
        _, reward, _, _, _ = env.step(action)
        assert isinstance(reward, float)

    def test_step_terminated_is_bool(self):
        """terminated from step() is a bool."""
        env = KhreibagaEnv()
        _, info = env.reset()
        action = _random_legal_action(info["action_mask"])
        _, _, terminated, _, _ = env.step(action)
        assert isinstance(terminated, bool)

    def test_step_truncated_always_false(self):
        """truncated from step() is always False (no time limit in env)."""
        env = KhreibagaEnv()
        _, info = env.reset()
        action = _random_legal_action(info["action_mask"])
        _, _, _, truncated, _ = env.step(action)
        assert truncated is False

    def test_step_info_has_action_mask(self):
        """info dict from step() contains 'action_mask' with correct
        shape/dtype."""
        env = KhreibagaEnv()
        _, info = env.reset()
        action = _random_legal_action(info["action_mask"])
        _, _, _, _, step_info = env.step(action)
        assert "action_mask" in step_info
        mask = step_info["action_mask"]
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (625,)
        assert mask.dtype == np.float32

    def test_step_illegal_action_raises(self):
        """Stepping with an illegal action raises ValueError."""
        env = KhreibagaEnv()
        _, info = env.reset()
        mask = info["action_mask"]
        # Find an action that is NOT legal
        illegal_actions = np.where(mask == 0.0)[0]
        assert len(illegal_actions) > 0, "Expected at least one illegal action"
        illegal = int(illegal_actions[0])
        with pytest.raises(ValueError):
            env.step(illegal)

    # -- properties --------------------------------------------------------

    def test_current_player_initial(self):
        """current_player is BLACK after reset."""
        env = KhreibagaEnv()
        env.reset()
        assert env.current_player == BLACK

    def test_current_player_switches_after_simple_move(self):
        """current_player becomes WHITE after a non-capture simple move.

        Use the canonical opening move sq11->sq12 (BLACK_MAN at (2,1) moves
        orthogonally forward to (2,2) EMPTY) which is always a simple move
        in the initial position.
        """
        env = KhreibagaEnv()
        _, info = env.reset()
        action = _action(11, 12)
        mask = info["action_mask"]
        if mask[action] == 1.0:
            env.step(action)
            assert env.current_player == WHITE
        else:
            # If captures are mandatory from the very start, take any legal
            # action and skip the assertion (edge case).
            pytest.skip(
                "Captures mandatory from start; cannot test simple move switch"
            )

    def test_done_initial(self):
        """done is False after reset."""
        env = KhreibagaEnv()
        env.reset()
        assert env.done is False

    def test_winner_initial(self):
        """winner is None after reset."""
        env = KhreibagaEnv()
        env.reset()
        assert env.winner is None

    # -- get_action_mask ---------------------------------------------------

    def test_get_action_mask_shape_dtype(self):
        """get_action_mask() returns (625,) float32."""
        env = KhreibagaEnv()
        env.reset()
        mask = env.get_action_mask()
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (625,)
        assert mask.dtype == np.float32

    # -- clone -------------------------------------------------------------

    def test_clone_independence(self):
        """Modifying a clone does not affect the original environment."""
        env = KhreibagaEnv()
        obs_orig, info_orig = env.reset()
        clone = env.clone()

        # Take a step in the clone
        action = _random_legal_action(info_orig["action_mask"])
        clone.step(action)

        # The original should be unaffected
        assert env.current_player == BLACK
        assert env.done is False
        np.testing.assert_array_equal(
            env.get_action_mask(), info_orig["action_mask"]
        )

    # -- render ------------------------------------------------------------

    def test_render_returns_string(self):
        """render() returns a string representation of the board."""
        env = KhreibagaEnv()
        env.reset()
        result = env.render()
        assert isinstance(result, str)
        assert len(result) > 0


# ===================================================================
# C. Integration / Rollout Tests
# ===================================================================

class TestRollout:
    """Play full or partial random games to check end-to-end integration."""

    MAX_STEPS = 500  # safety cap to avoid infinite loops in broken code

    def _play_random_game(self, seed=0):
        """Play a full random game and return (env, step_count, last_reward)."""
        rng = np.random.default_rng(seed)
        env = KhreibagaEnv()
        _, info = env.reset()
        step_count = 0
        last_reward = 0.0

        while not env.done and step_count < self.MAX_STEPS:
            mask = info["action_mask"]
            action = _random_legal_action(mask, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            last_reward = reward
            step_count += 1
            if terminated:
                break

        return env, step_count, last_reward

    def test_random_game_completes(self):
        """A random game terminates within the step cap."""
        env, step_count, _ = self._play_random_game(seed=1)
        assert env.done is True, (
            "Game did not terminate within %d steps" % self.MAX_STEPS
        )

    def test_random_game_winner_valid(self):
        """The winner is BLACK, WHITE, or None (draw)."""
        env, _, _ = self._play_random_game(seed=2)
        assert env.winner in (BLACK, WHITE, None)

    def test_random_game_reward_consistency(self):
        """On the terminal step, reward is +1, -1, or 0."""
        env, _, last_reward = self._play_random_game(seed=3)
        assert env.done is True
        assert last_reward in (1.0, -1.0, 0.0)

    def test_action_mask_always_has_moves_until_done(self):
        """During a game, the mask always has at least 1 legal action
        until done."""
        rng = np.random.default_rng(4)
        env = KhreibagaEnv()
        _, info = env.reset()
        step_count = 0

        while not env.done and step_count < self.MAX_STEPS:
            mask = info["action_mask"]
            assert mask.sum() >= 1.0, (
                "No legal actions at step %d but game is not done" % step_count
            )
            action = _random_legal_action(mask, rng)
            _, _, terminated, _, info = env.step(action)
            step_count += 1
            if terminated:
                break

    def test_multiple_random_games(self):
        """Play 5 random games; all should terminate within a reasonable
        number of moves."""
        for seed in range(5):
            env, step_count, _ = self._play_random_game(seed=seed + 100)
            assert env.done is True, (
                "Game %d did not terminate within %d steps"
                % (seed, self.MAX_STEPS)
            )
            assert step_count <= self.MAX_STEPS


# ===================================================================
# D. Consistency Tests
# ===================================================================

class TestConsistency:
    """Cross-validate encoder output against env output and raw GameState."""

    def test_encoder_matches_env_observation(self):
        """encode_observation() output matches the observation from
        env.reset()."""
        env = KhreibagaEnv()
        obs_env, _ = env.reset()

        # Manually create the same initial state and encode
        gs = GameState()
        obs_enc = encode_observation(
            gs.board, gs.current_player, gs.history,
            gs.current_hash, gs.move_count,
        )

        np.testing.assert_array_equal(obs_env, obs_enc)

    def test_env_obs_matches_game_obs(self):
        """The env observation is the numpy version of
        GameState.get_observation()."""
        env = KhreibagaEnv()
        obs_env, _ = env.reset()

        gs = GameState()
        raw_obs = gs.get_observation()  # nested Python lists
        obs_from_raw = np.array(raw_obs, dtype=np.float32)

        np.testing.assert_array_equal(obs_env, obs_from_raw)

    def test_encoder_matches_env_after_step(self):
        """After one step, encoder and env observations still agree."""
        env = KhreibagaEnv()
        _, info = env.reset()
        action = _random_legal_action(info["action_mask"])
        obs_env, _, _, _, _ = env.step(action)

        # Replicate the same step in a raw GameState
        gs = GameState()
        gs.step(action)
        obs_enc = encode_observation(
            gs.board, gs.current_player, gs.history,
            gs.current_hash, gs.move_count,
        )

        np.testing.assert_array_equal(obs_env, obs_enc)

    def test_action_mask_encoder_matches_env(self):
        """encode_action_mask(gs.get_action_mask()) matches
        env.get_action_mask()."""
        env = KhreibagaEnv()
        env.reset()

        gs = GameState()
        mask_raw = gs.get_action_mask()
        mask_enc = encode_action_mask(mask_raw)

        mask_env = env.get_action_mask()

        np.testing.assert_array_equal(mask_env, mask_enc)
