"""
Tests for khreibga.model -- KhreibagaNet architecture and input pipeline.

Covers:
  - Output shapes (policy B x 625, value B x 1)
  - Value head range [-1, 1] via tanh
  - Policy head outputs raw logits (no softmax)
  - Residual block identity shortcut
  - Various batch sizes (1, 8, 32)
  - Masked inference via predict()
  - Batching utilities (batch_observations, batch_action_masks)
  - Integration with KhreibagaEnv
  - Parameter count sanity check
  - Device handling
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from khreibga.model import (
    ACTION_SPACE,
    BOARD_H,
    BOARD_W,
    INPUT_PLANES,
    NUM_FILTERS,
    NUM_RES_BLOCKS,
    ConvBlock,
    KhreibagaNet,
    ResBlock,
    batch_action_masks,
    batch_observations,
    get_device,
    predict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    """Use CPU for deterministic testing."""
    return torch.device("cpu")


@pytest.fixture
def model(device):
    """A freshly initialised KhreibagaNet on CPU in eval mode."""
    net = KhreibagaNet().to(device)
    net.eval()
    return net


@pytest.fixture
def random_obs():
    """A single random (7, 5, 5) float32 observation."""
    return np.random.default_rng(42).random((INPUT_PLANES, BOARD_H, BOARD_W)).astype(np.float32)


@pytest.fixture
def random_mask():
    """A random (625,) float32 mask with ~10% of actions legal."""
    rng = np.random.default_rng(42)
    m = (rng.random(ACTION_SPACE) > 0.9).astype(np.float32)
    # Ensure at least one legal action
    if m.sum() == 0:
        m[0] = 1.0
    return m


# ---------------------------------------------------------------------------
# ConvBlock tests
# ---------------------------------------------------------------------------

class TestConvBlock:
    def test_output_shape(self, device):
        block = ConvBlock(7, 64).to(device)
        x = torch.randn(2, 7, 5, 5, device=device)
        out = block(x)
        assert out.shape == (2, 64, 5, 5)

    def test_output_non_negative(self, device):
        """ReLU activation means all outputs >= 0."""
        block = ConvBlock(7, 64).to(device)
        block.eval()
        x = torch.randn(4, 7, 5, 5, device=device)
        out = block(x)
        assert (out >= 0).all()


# ---------------------------------------------------------------------------
# ResBlock tests
# ---------------------------------------------------------------------------

class TestResBlock:
    def test_output_shape(self, device):
        block = ResBlock(64).to(device)
        x = torch.randn(2, 64, 5, 5, device=device)
        out = block(x)
        assert out.shape == (2, 64, 5, 5)

    def test_skip_connection(self, device):
        """With zero-initialised conv weights, output should equal ReLU(input)."""
        block = ResBlock(64).to(device)
        # Zero out all conv weights so residual path adds zero
        with torch.no_grad():
            block.conv1.weight.zero_()
            block.conv2.weight.zero_()
            block.bn1.weight.fill_(1)
            block.bn1.bias.zero_()
            block.bn1.running_mean.zero_()
            block.bn1.running_var.fill_(1)
            block.bn2.weight.fill_(1)
            block.bn2.bias.zero_()
            block.bn2.running_mean.zero_()
            block.bn2.running_var.fill_(1)
        block.eval()
        x = torch.randn(2, 64, 5, 5, device=device)
        out = block(x)
        expected = torch.relu(x)  # skip + 0 -> relu(x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_output_non_negative(self, device):
        block = ResBlock(64).to(device)
        block.eval()
        x = torch.randn(4, 64, 5, 5, device=device)
        out = block(x)
        assert (out >= -1e-6).all()


# ---------------------------------------------------------------------------
# KhreibagaNet tests -- shapes and ranges
# ---------------------------------------------------------------------------

class TestKhreibagaNetShapes:
    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_output_shapes(self, model, device, batch_size):
        x = torch.randn(batch_size, INPUT_PLANES, BOARD_H, BOARD_W, device=device)
        policy, value = model(x)
        assert policy.shape == (batch_size, ACTION_SPACE)
        assert value.shape == (batch_size, 1)

    def test_policy_is_raw_logits(self, model, device):
        """Policy should NOT be softmaxed (values can be negative and > 1)."""
        torch.manual_seed(0)
        x = torch.randn(16, INPUT_PLANES, BOARD_H, BOARD_W, device=device)
        policy, _ = model(x)
        # Raw logits will have some negative values (very likely with random init)
        assert (policy < 0).any(), "Expected some negative logits"
        # And row sums should NOT be 1.0 (they would be if softmax was applied)
        row_sums = policy.sum(dim=1)
        assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.1)

    def test_value_range(self, model, device):
        """Value output should be in [-1, 1] due to tanh."""
        torch.manual_seed(1)
        x = torch.randn(64, INPUT_PLANES, BOARD_H, BOARD_W, device=device)
        _, value = model(x)
        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_value_extremes(self, device):
        """With extreme inputs, value should approach +-1 but stay bounded."""
        model = KhreibagaNet().to(device)
        model.eval()
        x_pos = torch.full(
            (4, INPUT_PLANES, BOARD_H, BOARD_W), 100.0, device=device,
        )
        x_neg = torch.full(
            (4, INPUT_PLANES, BOARD_H, BOARD_W), -100.0, device=device,
        )
        _, v_pos = model(x_pos)
        _, v_neg = model(x_neg)
        assert (v_pos >= -1.0).all() and (v_pos <= 1.0).all()
        assert (v_neg >= -1.0).all() and (v_neg <= 1.0).all()


# ---------------------------------------------------------------------------
# KhreibagaNet tests -- architecture details
# ---------------------------------------------------------------------------

class TestKhreibagaNetArchitecture:
    def test_default_hyperparams(self):
        net = KhreibagaNet()
        assert net.conv_block.conv.in_channels == INPUT_PLANES
        assert net.conv_block.conv.out_channels == NUM_FILTERS
        assert len(net.res_blocks) == NUM_RES_BLOCKS

    def test_custom_hyperparams(self):
        net = KhreibagaNet(in_channels=3, num_filters=32, num_res_blocks=2)
        assert net.conv_block.conv.in_channels == 3
        assert net.conv_block.conv.out_channels == 32
        assert len(net.res_blocks) == 2

    def test_parameter_count(self):
        """Sanity check: model should have a reasonable number of params."""
        net = KhreibagaNet()
        total = sum(p.numel() for p in net.parameters())
        # With 64 filters and 6 res blocks, expect roughly 300k-600k params
        assert 100_000 < total < 1_000_000, f"Unexpected param count: {total}"

    def test_policy_fc_dimensions(self):
        net = KhreibagaNet()
        assert net.policy_fc.in_features == 2 * BOARD_H * BOARD_W  # 50
        assert net.policy_fc.out_features == ACTION_SPACE            # 625

    def test_value_fc_dimensions(self):
        net = KhreibagaNet()
        assert net.value_fc1.in_features == BOARD_H * BOARD_W       # 25
        assert net.value_fc1.out_features == 64
        assert net.value_fc2.in_features == 64
        assert net.value_fc2.out_features == 1

    def test_no_bias_in_conv_layers(self):
        """Conv layers before BN should have bias=False."""
        net = KhreibagaNet()
        assert net.conv_block.conv.bias is None
        assert net.policy_conv.bias is None
        assert net.value_conv.bias is None
        for block in net.res_blocks:
            assert block.conv1.bias is None
            assert block.conv2.bias is None


# ---------------------------------------------------------------------------
# Batching utility tests
# ---------------------------------------------------------------------------

class TestBatchObservations:
    def test_list_of_arrays(self, device):
        obs_list = [
            np.random.randn(INPUT_PLANES, BOARD_H, BOARD_W).astype(np.float32)
            for _ in range(4)
        ]
        t = batch_observations(obs_list, device=device)
        assert t.shape == (4, INPUT_PLANES, BOARD_H, BOARD_W)
        assert t.dtype == torch.float32
        assert t.device == device

    def test_single_array(self, device):
        """A pre-stacked (B, 7, 5, 5) array should also work."""
        arr = np.random.randn(3, INPUT_PLANES, BOARD_H, BOARD_W).astype(np.float32)
        t = batch_observations(arr, device=device)
        assert t.shape == (3, INPUT_PLANES, BOARD_H, BOARD_W)

    def test_values_preserved(self, device):
        obs = np.ones((1, INPUT_PLANES, BOARD_H, BOARD_W), dtype=np.float32) * 0.5
        t = batch_observations(obs, device=device)
        assert torch.allclose(t, torch.full_like(t, 0.5))


class TestBatchActionMasks:
    def test_list_of_arrays(self, device):
        masks = [np.zeros(ACTION_SPACE, dtype=np.float32) for _ in range(4)]
        masks[0][0] = 1.0
        t = batch_action_masks(masks, device=device)
        assert t.shape == (4, ACTION_SPACE)
        assert t.dtype == torch.float32

    def test_single_array(self, device):
        arr = np.ones((2, ACTION_SPACE), dtype=np.float32)
        t = batch_action_masks(arr, device=device)
        assert t.shape == (2, ACTION_SPACE)


# ---------------------------------------------------------------------------
# predict() tests
# ---------------------------------------------------------------------------

class TestPredict:
    def test_output_shapes(self, model, device, random_obs):
        obs_list = [random_obs, random_obs]
        probs, values = predict(model, obs_list, device=device)
        assert probs.shape == (2, ACTION_SPACE)
        assert values.shape == (2,)

    def test_probs_sum_to_one(self, model, device, random_obs):
        """Without masking, probs should sum to ~1.0."""
        probs, _ = predict(model, [random_obs], device=device)
        assert np.isclose(probs[0].sum(), 1.0, atol=1e-5)

    def test_probs_non_negative(self, model, device, random_obs):
        probs, _ = predict(model, [random_obs], device=device)
        assert (probs >= 0).all()

    def test_masked_probs(self, model, device, random_obs, random_mask):
        """Illegal actions should have 0 probability."""
        probs, _ = predict(model, [random_obs], [random_mask], device=device)
        illegal_indices = np.where(random_mask == 0)[0]
        assert np.allclose(probs[0, illegal_indices], 0.0, atol=1e-7)

    def test_masked_probs_sum_to_one(self, model, device, random_obs, random_mask):
        """Masked probs should still sum to ~1.0."""
        probs, _ = predict(model, [random_obs], [random_mask], device=device)
        assert np.isclose(probs[0].sum(), 1.0, atol=1e-5)

    def test_value_range(self, model, device, random_obs):
        _, values = predict(model, [random_obs], device=device)
        assert values[0] >= -1.0 and values[0] <= 1.0

    def test_batch_predict(self, model, device):
        """Predict on a batch of 8 different observations."""
        rng = np.random.default_rng(123)
        obs_list = [
            rng.random((INPUT_PLANES, BOARD_H, BOARD_W)).astype(np.float32)
            for _ in range(8)
        ]
        mask_list = [
            (rng.random(ACTION_SPACE) > 0.8).astype(np.float32)
            for _ in range(8)
        ]
        # Ensure at least one legal action per mask
        for m in mask_list:
            if m.sum() == 0:
                m[0] = 1.0
        probs, values = predict(model, obs_list, mask_list, device=device)
        assert probs.shape == (8, ACTION_SPACE)
        assert values.shape == (8,)
        # Each row sums to 1
        for i in range(8):
            assert np.isclose(probs[i].sum(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Integration with KhreibagaEnv
# ---------------------------------------------------------------------------

class TestEnvIntegration:
    def test_env_obs_to_model(self, model, device):
        """Feed real env observations through the model."""
        from khreibga.env import KhreibagaEnv

        env = KhreibagaEnv()
        obs, info = env.reset()

        # obs shape should be (7, 5, 5) from env
        assert obs.shape == (INPUT_PLANES, BOARD_H, BOARD_W)
        mask = info["action_mask"]
        assert mask.shape == (ACTION_SPACE,)

        probs, values = predict(model, [obs], [mask], device=device)
        assert probs.shape == (1, ACTION_SPACE)
        assert values.shape == (1,)
        assert np.isclose(probs[0].sum(), 1.0, atol=1e-5)
        assert -1.0 <= values[0] <= 1.0

        # Illegal actions must have zero probability
        illegal = np.where(mask == 0)[0]
        assert np.allclose(probs[0, illegal], 0.0, atol=1e-7)

    def test_multi_step_game(self, model, device):
        """Play a few random steps and check model handles each state."""
        from khreibga.env import KhreibagaEnv

        rng = np.random.default_rng(7)
        env = KhreibagaEnv()
        obs, info = env.reset()

        for _ in range(20):
            if env.done:
                break
            mask = info["action_mask"]
            probs, values = predict(model, [obs], [mask], device=device)
            # Pick a legal action
            legal = np.where(mask > 0)[0]
            action = rng.choice(legal)
            obs, reward, terminated, truncated, info = env.step(int(action))

            assert probs.shape == (1, ACTION_SPACE)
            assert -1.0 <= values[0] <= 1.0


# ---------------------------------------------------------------------------
# get_device tests
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_returns_torch_device(self):
        d = get_device()
        assert isinstance(d, torch.device)

    def test_cpu_always_available(self):
        """At minimum, CPU should be returned."""
        d = get_device()
        # Should be one of: cpu, cuda, mps
        assert d.type in ("cpu", "cuda", "mps")


# ---------------------------------------------------------------------------
# Gradient flow test
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_backward_pass(self, device):
        """Verify gradients flow through both heads."""
        model = KhreibagaNet().to(device)
        model.train()
        x = torch.randn(4, INPUT_PLANES, BOARD_H, BOARD_W, device=device)

        policy, value = model(x)

        # Dummy loss combining both heads
        target_policy = torch.zeros(4, ACTION_SPACE, device=device)
        target_policy[:, 0] = 1.0
        target_value = torch.ones(4, 1, device=device) * 0.5

        loss_p = torch.nn.functional.cross_entropy(policy, target_policy)
        loss_v = torch.nn.functional.mse_loss(value, target_value)
        loss = loss_p + loss_v

        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


# ---------------------------------------------------------------------------
# Determinism / reproducibility
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self, model, device):
        """Same input should produce identical output in eval mode."""
        x = torch.randn(2, INPUT_PLANES, BOARD_H, BOARD_W, device=device)
        p1, v1 = model(x)
        p2, v2 = model(x)
        assert torch.equal(p1, p2)
        assert torch.equal(v1, v2)
