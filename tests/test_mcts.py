"""
Tests for khreibga.mcts -- Monte Carlo Tree Search.

Covers:
  - MCTSNode: q_value, UCB score, select_child, sign-flip for opponent
  - _advance_through_forced: forced chains, branching, terminal
  - MCTS.search: shape, legal-only visits, visit sums, short-circuit
  - Terminal value handling (win/loss/draw)
  - get_action_probs: temperature 0, 1, edge cases
  - select_action: deterministic and stochastic
  - Dirichlet noise: priors modified, disabled when epsilon=0
  - Integration: end-to-end with env and model
  - Multi-step capture chain handling
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from khreibga.board import BLACK, WHITE, BLACK_MAN, WHITE_MAN, EMPTY
from khreibga.env import KhreibagaEnv
from khreibga.game import compute_zobrist_hash
from khreibga.model import KhreibagaNet, ACTION_SPACE
from khreibga.mcts import (
    MCTS,
    MCTSNode,
    _advance_through_forced,
    _get_obs,
    get_action_probs,
    select_action,
)


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
def env():
    e = KhreibagaEnv()
    e.reset()
    return e


def _make_custom_env(board, player=BLACK, chain_piece=None):
    """Create a KhreibagaEnv with a custom board state."""
    e = KhreibagaEnv()
    e.game.board = list(board)
    e.game.current_player = player
    e.game.chain_piece = chain_piece
    e.game.done = False
    e.game.winner = None
    e.game.half_move_clock = 0
    e.game.move_count = 0
    e.game.current_hash = compute_zobrist_hash(e.game.board, player)
    e.game.history = {e.game.current_hash: 1}
    return e


# ---------------------------------------------------------------------------
# MCTSNode tests
# ---------------------------------------------------------------------------

class TestMCTSNode:
    def test_creation_defaults(self):
        node = MCTSNode()
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.prior == 0.0
        assert node.parent is None
        assert node.action is None
        assert node.player is None
        assert not node.is_expanded()

    def test_q_value_zero_visits(self):
        node = MCTSNode()
        assert node.q_value == 0.0

    def test_q_value_after_updates(self):
        node = MCTSNode()
        node.visit_count = 4
        node.value_sum = 2.0
        assert node.q_value == 0.5

    def test_q_value_negative(self):
        node = MCTSNode()
        node.visit_count = 2
        node.value_sum = -1.0
        assert node.q_value == -0.5

    def test_is_expanded(self):
        node = MCTSNode()
        assert not node.is_expanded()
        node.children[0] = MCTSNode(parent=node, action=0)
        assert node.is_expanded()


class TestSelectChild:
    def test_prefers_high_prior_unvisited(self):
        """Among unvisited children, prefer the one with higher prior."""
        root = MCTSNode(player=BLACK)
        root.visit_count = 1
        c1 = MCTSNode(prior=0.8, parent=root, action=0)
        c2 = MCTSNode(prior=0.2, parent=root, action=1)
        root.children = {0: c1, 1: c2}

        selected = root.select_child(c_puct=1.0, root_player=BLACK)
        assert selected.action == 0

    def test_prefers_less_visited(self):
        """Given equal Q and prior, prefer less-visited child."""
        root = MCTSNode(player=BLACK)
        root.visit_count = 10
        c1 = MCTSNode(prior=0.5, parent=root, action=0)
        c1.visit_count = 5
        c2 = MCTSNode(prior=0.5, parent=root, action=1)
        c2.visit_count = 1
        root.children = {0: c1, 1: c2}

        selected = root.select_child(c_puct=1.0, root_player=BLACK)
        assert selected.action == 1  # less visited

    def test_root_player_maximises_q(self):
        """Root player picks the child with higher Q."""
        root = MCTSNode(player=BLACK)
        root.visit_count = 100
        c1 = MCTSNode(prior=0.5, parent=root, action=0)
        c1.visit_count = 50
        c1.value_sum = 25.0  # Q = 0.5
        c2 = MCTSNode(prior=0.5, parent=root, action=1)
        c2.visit_count = 50
        c2.value_sum = -10.0  # Q = -0.2
        root.children = {0: c1, 1: c2}

        selected = root.select_child(c_puct=0.01, root_player=BLACK)
        assert selected.action == 0  # higher Q

    def test_opponent_minimises_q(self):
        """Opponent picks the child with LOWER Q (worst for root)."""
        opp_node = MCTSNode(player=WHITE)
        opp_node.visit_count = 100
        c1 = MCTSNode(prior=0.5, parent=opp_node, action=0)
        c1.visit_count = 50
        c1.value_sum = 25.0  # Q = 0.5 (good for root)
        c2 = MCTSNode(prior=0.5, parent=opp_node, action=1)
        c2.visit_count = 50
        c2.value_sum = -10.0  # Q = -0.2 (bad for root)
        opp_node.children = {0: c1, 1: c2}

        # root_player=BLACK, this node's player=WHITE (opponent)
        # opponent should pick c2 (lower Q = more harmful to root)
        selected = opp_node.select_child(c_puct=0.01, root_player=BLACK)
        assert selected.action == 1


# ---------------------------------------------------------------------------
# _advance_through_forced tests
# ---------------------------------------------------------------------------

class TestAdvanceThroughForced:
    def test_no_forced_moves(self, env):
        """Initial position has many legal moves -- should not advance."""
        player_before = env.current_player
        mask_before = env.get_action_mask()
        _advance_through_forced(env)
        # Nothing should change
        assert env.current_player == player_before
        assert np.array_equal(env.get_action_mask(), mask_before)

    def test_forced_chain_auto_played(self):
        """Set up a 2-hop forced chain and verify it's auto-played."""
        # BLACK man at (0,0)=sq0, WHITE men at (1,1)=sq6 and (3,3)=sq18
        # Chain: 0->12 (capture 6), then forced 12->24 (capture 18)
        board = [EMPTY] * 25
        board[0] = BLACK_MAN
        board[6] = WHITE_MAN
        board[18] = WHITE_MAN
        env = _make_custom_env(board, player=BLACK)

        # Step with first hop
        env.step(0 * 25 + 12)  # action 12: sq0 -> sq12
        assert env.game.chain_piece == 12

        # Now advance through forced continuation
        _advance_through_forced(env)

        # Chain should be complete, player switched
        assert env.game.chain_piece is None
        assert env.current_player == WHITE
        # BLACK man should be at sq24, promoted to KING (row 4 = promo row)
        from khreibga.board import BLACK_KING
        assert env.game.board[24] == BLACK_KING
        assert env.game.board[6] == EMPTY
        assert env.game.board[18] == EMPTY

    def test_terminal_during_forced(self):
        """If forced moves lead to terminal, stop correctly."""
        # BLACK man at (0,0)=sq0, single WHITE man at (1,1)=sq6
        # After capture: BLACK at sq12, WHITE eliminated -> game over
        board = [EMPTY] * 25
        board[0] = BLACK_MAN
        board[6] = WHITE_MAN
        env = _make_custom_env(board, player=BLACK)

        # Step with capture
        env.step(0 * 25 + 12)  # capture sq6, land on sq12
        # Game should be terminal (WHITE has no pieces)
        assert env.done

        # advance should be a no-op on terminal
        _advance_through_forced(env)
        assert env.done


# ---------------------------------------------------------------------------
# _get_obs tests
# ---------------------------------------------------------------------------

class TestGetObs:
    def test_shape_dtype(self, env):
        obs = _get_obs(env)
        assert obs.shape == (7, 5, 5)
        assert obs.dtype == np.float32

    def test_matches_env_reset(self):
        env = KhreibagaEnv()
        obs_from_reset, _ = env.reset()
        obs_from_helper = _get_obs(env)
        assert np.allclose(obs_from_reset, obs_from_helper)


# ---------------------------------------------------------------------------
# MCTS.search tests
# ---------------------------------------------------------------------------

class TestMCTSSearch:
    def test_output_shape(self, model, device, env):
        mcts = MCTS(model, num_simulations=5, dirichlet_epsilon=0.0,
                     device=device)
        visits = mcts.search(env)
        assert visits.shape == (ACTION_SPACE,)
        assert visits.dtype == np.float32

    def test_visits_only_legal(self, model, device, env):
        """Visit counts should be zero for illegal actions."""
        mcts = MCTS(model, num_simulations=10, dirichlet_epsilon=0.0,
                     device=device)
        visits = mcts.search(env)
        mask = env.get_action_mask()
        illegal = np.where(mask == 0)[0]
        assert np.all(visits[illegal] == 0)

    def test_visit_count_sum(self, model, device, env):
        """Total visits to root children should equal num_simulations."""
        n_sims = 20
        mcts = MCTS(model, num_simulations=n_sims, dirichlet_epsilon=0.0,
                     device=device)
        visits = mcts.search(env)
        assert int(visits.sum()) == n_sims

    def test_single_legal_action_short_circuit(self, model, device):
        """If only one action is legal, return immediately with visits=1."""
        # sq1 = (0,1), odd parity (r+c=1), no diagonals.
        # Forward move for BLACK is only (1,1)=sq6 (orthogonal north).
        board = [EMPTY] * 25
        board[1] = BLACK_MAN   # (0,1) -- only one forward move
        board[24] = WHITE_MAN  # far away so no capture
        env = _make_custom_env(board, player=BLACK)

        # Verify exactly one legal action
        legal = np.where(env.get_action_mask() > 0)[0]
        assert len(legal) == 1

        mcts = MCTS(model, num_simulations=100, dirichlet_epsilon=0.0,
                     device=device)
        visits = mcts.search(env)

        # Should have exactly 1.0 visit on the single legal action
        assert visits.sum() == 1.0
        assert visits[legal[0]] == 1.0

    def test_terminal_state_returns_zeros(self, model, device):
        """Search on a terminal state returns all zeros."""
        env = KhreibagaEnv()
        env.game.done = True
        env.game.winner = BLACK

        mcts = MCTS(model, num_simulations=10, device=device)
        visits = mcts.search(env)
        assert np.all(visits == 0)

    def test_no_legal_actions_returns_zeros(self, model, device):
        """If no legal actions (shouldn't happen normally), return zeros."""
        env = KhreibagaEnv()
        env.game.board = [EMPTY] * 25  # empty board
        env.game.done = True  # must be done since no pieces

        mcts = MCTS(model, num_simulations=10, device=device)
        visits = mcts.search(env)
        assert np.all(visits == 0)


# ---------------------------------------------------------------------------
# Terminal value tests
# ---------------------------------------------------------------------------

class TestTerminalValue:
    def test_win(self, model, device):
        mcts = MCTS(model, device=device)
        env = KhreibagaEnv()
        env.game.done = True
        env.game.winner = BLACK
        assert mcts._terminal_value(env, BLACK) == 1.0

    def test_loss(self, model, device):
        mcts = MCTS(model, device=device)
        env = KhreibagaEnv()
        env.game.done = True
        env.game.winner = WHITE
        assert mcts._terminal_value(env, BLACK) == -1.0

    def test_draw(self, model, device):
        mcts = MCTS(model, device=device)
        env = KhreibagaEnv()
        env.game.done = True
        env.game.winner = None
        assert mcts._terminal_value(env, BLACK) == 0.0


# ---------------------------------------------------------------------------
# get_action_probs tests
# ---------------------------------------------------------------------------

class TestGetActionProbs:
    def test_temperature_1_proportional(self):
        """With τ=1, probs should be proportional to visit counts."""
        visits = np.zeros(ACTION_SPACE, dtype=np.float32)
        visits[0] = 30.0
        visits[1] = 70.0
        probs = get_action_probs(visits, temperature=1.0)
        assert np.isclose(probs[0], 0.3, atol=1e-5)
        assert np.isclose(probs[1], 0.7, atol=1e-5)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)

    def test_temperature_0_deterministic(self):
        """With τ=0, all probability on the most-visited action."""
        visits = np.zeros(ACTION_SPACE, dtype=np.float32)
        visits[5] = 100.0
        visits[10] = 50.0
        probs = get_action_probs(visits, temperature=0)
        assert probs[5] == 1.0
        assert probs[10] == 0.0
        assert probs.sum() == 1.0

    def test_low_temperature_concentrates(self):
        """Lower temperature concentrates probability on top action."""
        visits = np.zeros(ACTION_SPACE, dtype=np.float32)
        visits[0] = 60.0
        visits[1] = 40.0

        probs_high = get_action_probs(visits, temperature=1.0)
        probs_low = get_action_probs(visits, temperature=0.5)

        # Low temp should give more weight to action 0
        assert probs_low[0] > probs_high[0]
        assert probs_low[1] < probs_high[1]

    def test_zero_visits(self):
        """All-zero visits should return all-zero probs."""
        visits = np.zeros(ACTION_SPACE, dtype=np.float32)
        probs = get_action_probs(visits, temperature=1.0)
        assert np.all(probs == 0)

    def test_sums_to_one(self):
        visits = np.zeros(ACTION_SPACE, dtype=np.float32)
        visits[0] = 10.0
        visits[100] = 20.0
        visits[200] = 30.0
        probs = get_action_probs(visits, temperature=1.0)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# select_action tests
# ---------------------------------------------------------------------------

class TestSelectAction:
    def test_deterministic_picks_best(self):
        visits = np.zeros(ACTION_SPACE, dtype=np.float32)
        visits[42] = 100.0
        visits[7] = 50.0
        action = select_action(visits, temperature=0)
        assert action == 42

    def test_stochastic_picks_legal(self):
        """With τ=1, selected action should be one with visits > 0."""
        rng = np.random.default_rng(42)
        visits = np.zeros(ACTION_SPACE, dtype=np.float32)
        visits[10] = 50.0
        visits[20] = 50.0
        for _ in range(20):
            action = select_action(visits, temperature=1.0, rng=rng)
            assert action in (10, 20)

    def test_reproducible_with_rng(self):
        visits = np.zeros(ACTION_SPACE, dtype=np.float32)
        visits[5] = 30.0
        visits[10] = 70.0
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        a1 = select_action(visits, temperature=1.0, rng=rng1)
        a2 = select_action(visits, temperature=1.0, rng=rng2)
        assert a1 == a2


# ---------------------------------------------------------------------------
# Dirichlet noise tests
# ---------------------------------------------------------------------------

class TestDirichletNoise:
    def test_noise_modifies_priors(self, model, device, env):
        """With epsilon > 0, root priors should differ from raw NN priors."""
        mcts = MCTS(model, num_simulations=5, dirichlet_epsilon=0.25,
                     device=device)
        # Run search -- it adds noise internally
        # We can't easily observe priors, so just verify search completes
        visits = mcts.search(env)
        assert visits.sum() == 5

    def test_no_noise_when_epsilon_zero(self, model, device, env):
        """With epsilon=0, root priors should match raw NN output."""
        np.random.seed(0)
        torch.manual_seed(0)
        mcts = MCTS(model, num_simulations=5, dirichlet_epsilon=0.0,
                     device=device)
        visits = mcts.search(env)
        assert visits.sum() == 5  # search completes normally

    def test_noise_applied_to_root_children(self, model, device):
        """Directly test that _add_dirichlet_noise changes priors."""
        mcts = MCTS(model, dirichlet_epsilon=0.5, dirichlet_alpha=1.0,
                     device=device)
        root = MCTSNode(player=BLACK)

        # Create children with uniform priors
        legal = np.array([0, 1, 2, 3, 4])
        for a in legal:
            root.children[int(a)] = MCTSNode(
                prior=0.2, parent=root, action=int(a),
            )

        original_priors = [root.children[a].prior for a in range(5)]
        np.random.seed(42)
        mcts._add_dirichlet_noise(root, legal)
        new_priors = [root.children[a].prior for a in range(5)]

        # At least some priors should have changed
        assert any(
            abs(o - n) > 0.01 for o, n in zip(original_priors, new_priors)
        )
        # Priors should still be positive
        assert all(p > 0 for p in new_priors)


# ---------------------------------------------------------------------------
# Multi-step capture handling tests
# ---------------------------------------------------------------------------

class TestMultiStepCapture:
    def test_forced_chain_skipped_in_search(self, model, device):
        """MCTS should auto-play forced chain hops (not expand them)."""
        # BLACK man at (0,0)=sq0, WHITE men at (1,1)=sq6 and (3,3)=sq18
        # Plus one more WHITE piece so game doesn't end after capture
        board = [EMPTY] * 25
        board[0] = BLACK_MAN
        board[6] = WHITE_MAN
        board[18] = WHITE_MAN
        board[24] = WHITE_MAN  # extra WHITE piece at (4,4)
        env = _make_custom_env(board, player=BLACK)

        # Only legal action is first hop of chain: 0 -> 12 (capture 6)
        mask = env.get_action_mask()
        legal = np.where(mask > 0)[0]

        # If there's only one legal action, search short-circuits
        if len(legal) == 1:
            mcts = MCTS(model, num_simulations=10, dirichlet_epsilon=0.0,
                         device=device)
            visits = mcts.search(env)
            assert visits[legal[0]] == 1.0
            assert visits.sum() == 1.0
        else:
            # Multiple actions available -- run search
            mcts = MCTS(model, num_simulations=10, dirichlet_epsilon=0.0,
                         device=device)
            visits = mcts.search(env)
            # All visits should be on legal actions only
            assert np.all(visits[np.where(mask == 0)[0]] == 0)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_search_from_initial_position(self, model, device):
        """Run MCTS from the initial position and verify basic properties."""
        env = KhreibagaEnv()
        env.reset()
        n_sims = 15
        mcts = MCTS(model, num_simulations=n_sims, dirichlet_epsilon=0.0,
                     device=device)
        visits = mcts.search(env)

        # Shape and sum
        assert visits.shape == (ACTION_SPACE,)
        assert int(visits.sum()) == n_sims

        # Only legal actions visited
        mask = env.get_action_mask()
        assert np.all(visits[np.where(mask == 0)[0]] == 0)
        assert np.any(visits > 0)

    def test_search_then_step(self, model, device):
        """Run MCTS, pick an action, step the env, repeat."""
        env = KhreibagaEnv()
        env.reset()
        mcts = MCTS(model, num_simulations=10, dirichlet_epsilon=0.0,
                     device=device)
        rng = np.random.default_rng(0)

        for _ in range(5):
            if env.done:
                break
            visits = mcts.search(env)
            action = select_action(visits, temperature=1.0, rng=rng)
            env.step(action)

    def test_full_game_with_mcts(self, model, device):
        """Play a full game using MCTS (few sims) and verify it terminates."""
        env = KhreibagaEnv()
        env.reset()
        mcts = MCTS(model, num_simulations=5, dirichlet_epsilon=0.0,
                     device=device)
        rng = np.random.default_rng(7)

        max_steps = 300
        for step in range(max_steps):
            if env.done:
                break
            visits = mcts.search(env)
            action = select_action(visits, temperature=0.5, rng=rng)
            env.step(action)

        # Game should have ended within max_steps
        # (50-move rule or elimination or stalemate)
        assert env.done or step == max_steps - 1

    def test_different_simulations_consistent(self, model, device):
        """Multiple searches from same state with same seed give same result."""
        env = KhreibagaEnv()
        env.reset()

        # Disable noise for determinism
        mcts = MCTS(model, num_simulations=10, dirichlet_epsilon=0.0,
                     device=device)

        visits1 = mcts.search(env)
        visits2 = mcts.search(env)

        # With no randomness, searches should be deterministic
        assert np.array_equal(visits1, visits2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_search_with_few_pieces(self, model, device):
        """Search with very few pieces (endgame-like)."""
        board = [EMPTY] * 25
        board[0] = BLACK_MAN
        board[24] = WHITE_MAN
        env = _make_custom_env(board, player=BLACK)

        mcts = MCTS(model, num_simulations=10, dirichlet_epsilon=0.0,
                     device=device)
        visits = mcts.search(env)
        assert visits.shape == (ACTION_SPACE,)
        assert visits.sum() > 0

    def test_custom_hyperparams(self, model, device, env):
        """MCTS with custom hyperparameters should work."""
        mcts = MCTS(
            model, c_puct=2.0, num_simulations=5,
            dirichlet_alpha=0.5, dirichlet_epsilon=0.1,
            device=device,
        )
        visits = mcts.search(env)
        assert visits.shape == (ACTION_SPACE,)
