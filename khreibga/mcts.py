"""
Monte Carlo Tree Search (MCTS) for Khreibaga Zero.

Implements AlphaZero-style MCTS with:
  - UCB selection with neural network priors
  - Expansion with NN evaluation at leaf nodes
  - Multi-step capture handling (forced moves auto-played)
  - Action masking (illegal actions get zero probability)
  - Dirichlet noise at root for exploration
  - Temperature-based action selection

Values are stored from the ROOT player's perspective throughout the tree.
At root-player nodes, UCB maximises Q; at opponent nodes, UCB maximises -Q,
so the opponent effectively minimises root's value.
"""

from __future__ import annotations

import math
import numpy as np
import torch

from khreibga.env import KhreibagaEnv
from khreibga.model import KhreibagaNet, predict, ACTION_SPACE

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_C_PUCT: float = 1.0
DEFAULT_NUM_SIMULATIONS: int = 200
DEFAULT_DIRICHLET_ALPHA: float = 0.3
DEFAULT_DIRICHLET_EPSILON: float = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_obs(env: KhreibagaEnv) -> np.ndarray:
    """Return the current observation as a (7, 5, 5) float32 numpy array."""
    return np.array(env.game.get_observation(), dtype=np.float32)


def _advance_through_forced(env: KhreibagaEnv) -> None:
    """Auto-play positions with exactly one legal action.

    This handles multi-step capture chains where only one continuation
    exists (Task 15).  The NN is **not** queried at these intermediate
    states -- they are internal environment steps, not MCTS expansions.

    Stops when:
      - the game is over, OR
      - there are 0 or 2+ legal actions (a true decision point).
    """
    while not env.done:
        mask = env.get_action_mask()
        legal = np.where(mask > 0)[0]
        if len(legal) != 1:
            break
        env.step(int(legal[0]))


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------

class MCTSNode:
    """A single node in the MCTS search tree.

    All value statistics (``value_sum``, ``q_value``) are stored from the
    **root player's** perspective so that back-propagation is a simple
    addition and UCB selection only needs a sign flip at opponent nodes.
    """

    __slots__ = (
        "parent", "action", "prior", "children",
        "visit_count", "value_sum", "player",
    )

    def __init__(
        self,
        prior: float = 0.0,
        parent: MCTSNode | None = None,
        action: int | None = None,
        player: int | None = None,
    ) -> None:
        self.prior: float = prior
        self.parent: MCTSNode | None = parent
        self.action: int | None = action
        self.player: int | None = player  # current_player at this node
        self.children: dict[int, MCTSNode] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0

    # -- properties --

    @property
    def q_value(self) -> float:
        """Mean value from root player's perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """True if this node has been expanded (children created)."""
        return len(self.children) > 0

    # -- selection --

    def select_child(self, c_puct: float, root_player: int) -> MCTSNode:
        """Select the child with the highest UCB score.

        At a root-player node we *maximise* ``Q + exploration``.
        At an opponent node we *maximise* ``-Q + exploration``
        (i.e. the opponent picks the action worst for root).
        """
        sqrt_parent = math.sqrt(max(1, self.visit_count))
        sign = 1.0 if self.player == root_player else -1.0

        best_score = -math.inf
        best_child: MCTSNode | None = None

        for child in self.children.values():
            exploration = (
                c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            )
            q = child.q_value if child.visit_count > 0 else 0.0
            score = sign * q + exploration
            if score > best_score:
                best_score = score
                best_child = child

        assert best_child is not None
        return best_child


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTS:
    """AlphaZero-style Monte Carlo Tree Search for Khreibaga.

    Parameters
    ----------
    model : KhreibagaNet
        Neural network (should already be on *device* and in eval mode).
    c_puct : float
        Exploration constant for the UCB formula.
    num_simulations : int
        Number of MCTS simulations per ``search()`` call.
    dirichlet_alpha : float
        Alpha for Dirichlet noise added to root priors.
    dirichlet_epsilon : float
        Mixing weight of noise vs. NN prior at root.
        Set to ``0.0`` to disable noise (e.g. during evaluation).
    device : torch.device or None
        Device for NN inference.  Defaults to model's device.
    rng : np.random.Generator or None
        Random generator used for Dirichlet root noise.
    """

    def __init__(
        self,
        model: KhreibagaNet,
        c_puct: float = DEFAULT_C_PUCT,
        num_simulations: int = DEFAULT_NUM_SIMULATIONS,
        dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
        dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPSILON,
        device: torch.device | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device or next(model.parameters()).device
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def search(self, env: KhreibagaEnv) -> np.ndarray:
        """Run MCTS from the current game state.

        Parameters
        ----------
        env : KhreibagaEnv
            The game state to search from.  **Not** modified.

        Returns
        -------
        np.ndarray
            Shape ``(625,)``.  Raw visit counts for each action.
        """
        if env.done:
            return np.zeros(ACTION_SPACE, dtype=np.float32)

        root_player = env.current_player
        root = MCTSNode(prior=1.0, player=root_player)

        # Identify legal actions at root
        mask = env.get_action_mask()
        legal_actions = np.where(mask > 0)[0]

        if len(legal_actions) == 0:
            return np.zeros(ACTION_SPACE, dtype=np.float32)

        # Short-circuit: only one legal action -- no decision to make
        if len(legal_actions) == 1:
            visits = np.zeros(ACTION_SPACE, dtype=np.float32)
            visits[int(legal_actions[0])] = 1.0
            return visits

        # Evaluate root with NN
        obs = _get_obs(env)
        probs, _ = predict(
            self.model, [obs], [mask], device=self.device,
        )

        # Create root children
        for a in legal_actions:
            root.children[int(a)] = MCTSNode(
                prior=float(probs[0, a]),
                parent=root,
                action=int(a),
            )

        # Add Dirichlet noise to root priors
        if self.dirichlet_epsilon > 0:
            self._add_dirichlet_noise(root, legal_actions)

        # ---- Simulation loop ----
        for _ in range(self.num_simulations):
            node = root
            sim_env = env.clone()

            # --- Selection ---
            while node.is_expanded() and not sim_env.done:
                node = node.select_child(self.c_puct, root_player)
                sim_env.step(node.action)
                _advance_through_forced(sim_env)
                # Set player on first visit
                if node.player is None:
                    node.player = sim_env.current_player

            # --- Evaluation ---
            if sim_env.done:
                leaf_value = self._terminal_value(sim_env, root_player)
            else:
                leaf_obs = _get_obs(sim_env)
                leaf_mask = sim_env.get_action_mask()
                leaf_probs, leaf_val = predict(
                    self.model, [leaf_obs], [leaf_mask],
                    device=self.device,
                )
                nn_value = float(leaf_val[0])

                # Convert NN value (from current player's perspective)
                # to root player's perspective.
                if sim_env.current_player == root_player:
                    leaf_value = nn_value
                else:
                    leaf_value = -nn_value

                # --- Expansion ---
                leaf_legal = np.where(leaf_mask > 0)[0]
                for a in leaf_legal:
                    node.children[int(a)] = MCTSNode(
                        prior=float(leaf_probs[0, a]),
                        parent=node,
                        action=int(a),
                    )

            # --- Back-propagation ---
            while node is not None:
                node.visit_count += 1
                node.value_sum += leaf_value
                node = node.parent

        # ---- Collect visit counts ----
        visits = np.zeros(ACTION_SPACE, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        return visits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _terminal_value(env: KhreibagaEnv, root_player: int) -> float:
        """Compute the terminal value from *root_player*'s perspective."""
        if env.winner is None:
            return 0.0        # draw
        if env.winner == root_player:
            return 1.0
        return -1.0

    def _add_dirichlet_noise(
        self, root: MCTSNode, legal_actions: np.ndarray,
    ) -> None:
        """Mix Dirichlet noise into root-child priors for exploration."""
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(legal_actions))
        eps = self.dirichlet_epsilon
        for i, a in enumerate(legal_actions):
            child = root.children[int(a)]
            child.prior = (1 - eps) * child.prior + eps * float(noise[i])


# ---------------------------------------------------------------------------
# Temperature-based action selection
# ---------------------------------------------------------------------------

def get_action_probs(
    visit_counts: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Convert raw MCTS visit counts to a probability distribution.

    Parameters
    ----------
    visit_counts : np.ndarray
        Shape ``(625,)``.  Raw visit counts from :meth:`MCTS.search`.
    temperature : float
        Controls exploration vs. exploitation:
        - ``temperature = 1.0`` : proportional to visit counts.
        - ``temperature -> 0``  : deterministic (pick highest visits).

    Returns
    -------
    np.ndarray
        Shape ``(625,)``.  Probability distribution summing to 1.
    """
    total = visit_counts.sum()
    if total == 0:
        return np.zeros_like(visit_counts)

    if temperature == 0:
        probs = np.zeros_like(visit_counts)
        probs[np.argmax(visit_counts)] = 1.0
        return probs

    adjusted = visit_counts ** (1.0 / temperature)
    adj_total = adjusted.sum()
    if adj_total == 0:
        return np.zeros_like(visit_counts)
    return adjusted / adj_total


def select_action(
    visit_counts: np.ndarray,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample an action from the visit-count distribution.

    Parameters
    ----------
    visit_counts : np.ndarray
        Shape ``(625,)``.  Raw visit counts.
    temperature : float
        Temperature for action selection.
    rng : np.random.Generator or None
        RNG for reproducibility.  If ``None``, a default one is created.

    Returns
    -------
    int
        Selected action index in ``[0, 625)``.
    """
    probs = get_action_probs(visit_counts, temperature)

    if rng is None:
        rng = np.random.default_rng()

    if temperature == 0:
        return int(np.argmax(probs))

    total = probs.sum()
    if total == 0:
        return 0
    return int(rng.choice(len(probs), p=probs))
