"""
Neural network for Khreibaga Zero (AlphaZero-style dual-head ResNet).

Architecture (from spec Section 3.2)
-------------------------------------
  Input : (B, 7, 5, 5)
  Conv block : Conv2d(7 -> 64, 3x3, pad=1) -> BN -> ReLU
  Residual tower : 6 ResBlocks (64 filters each)
  Policy head : Conv(2, 1x1) -> BN -> ReLU -> Flatten -> Linear(50 -> 625)
  Value head  : Conv(1, 1x1) -> BN -> ReLU -> Flatten -> Linear(25 -> 64)
                -> ReLU -> Linear(64 -> 1) -> Tanh

The policy head outputs raw logits (no softmax).
The value head outputs a scalar in [-1, 1] via tanh.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_PLANES = 7
BOARD_H = 5
BOARD_W = 5
NUM_SQUARES = BOARD_H * BOARD_W          # 25
ACTION_SPACE = NUM_SQUARES * NUM_SQUARES  # 625
NUM_FILTERS = 64
NUM_RES_BLOCKS = 6


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Initial convolutional block: Conv -> BN -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ResBlock(nn.Module):
    """Residual block: Conv -> BN -> ReLU -> Conv -> BN -> skip -> ReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out, inplace=True)


# ---------------------------------------------------------------------------
# Dual-head network
# ---------------------------------------------------------------------------

class KhreibagaNet(nn.Module):
    """AlphaZero-style dual-head convolutional network for Khreibaga.

    Parameters
    ----------
    in_channels : int
        Number of input feature planes (default 7).
    num_filters : int
        Number of convolutional filters in the trunk (default 64).
    num_res_blocks : int
        Number of residual blocks in the trunk (default 6).
    """

    def __init__(
        self,
        in_channels: int = INPUT_PLANES,
        num_filters: int = NUM_FILTERS,
        num_res_blocks: int = NUM_RES_BLOCKS,
    ) -> None:
        super().__init__()

        # --- Shared trunk ---
        self.conv_block = ConvBlock(in_channels, num_filters)
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # --- Policy head ---
        self.policy_conv = nn.Conv2d(
            num_filters, 2, kernel_size=1, bias=False,
        )
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_H * BOARD_W, ACTION_SPACE)

        # --- Value head ---
        self.value_conv = nn.Conv2d(
            num_filters, 1, kernel_size=1, bias=False,
        )
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_H * BOARD_W, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of observations, shape ``(B, 7, 5, 5)``.

        Returns
        -------
        policy_logits : torch.Tensor
            Raw logits over the 625-dim action space, shape ``(B, 625)``.
            No softmax is applied (it is part of the loss function).
        value : torch.Tensor
            Board value estimate in ``[-1, 1]``, shape ``(B, 1)``.
        """
        # Shared trunk
        s = self.conv_block(x)
        s = self.res_blocks(s)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(s)), inplace=False)
        p = p.view(p.size(0), -1)          # (B, 2*5*5) = (B, 50)
        p = self.policy_fc(p)              # (B, 625)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(s)), inplace=False)
        v = v.view(v.size(0), -1)          # (B, 1*5*5) = (B, 25)
        v = F.relu(self.value_fc1(v), inplace=True)  # (B, 64)
        v = torch.tanh(self.value_fc2(v))  # (B, 1)

        return p, v


# ---------------------------------------------------------------------------
# Canonical input pipeline (Task 13)
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batch_observations(
    observations: list[np.ndarray] | np.ndarray,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Stack a list of (7, 5, 5) numpy arrays into a GPU-ready batch tensor.

    Parameters
    ----------
    observations : list of np.ndarray or np.ndarray
        Each element has shape ``(7, 5, 5)`` and dtype ``float32``.
        A single array of shape ``(B, 7, 5, 5)`` is also accepted.
    device : torch.device or None
        Target device.  If ``None``, uses :func:`get_device`.

    Returns
    -------
    torch.Tensor
        Shape ``(B, 7, 5, 5)`` on the target device.
    """
    if device is None:
        device = get_device()
    if isinstance(observations, np.ndarray) and observations.ndim == 4:
        arr = observations
    else:
        arr = np.stack(observations, axis=0)
    return torch.from_numpy(arr).to(device)


def batch_action_masks(
    masks: list[np.ndarray] | np.ndarray,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Stack a list of (625,) numpy masks into a GPU-ready batch tensor.

    Parameters
    ----------
    masks : list of np.ndarray or np.ndarray
        Each element has shape ``(625,)`` and dtype ``float32``.
        A single array of shape ``(B, 625)`` is also accepted.
    device : torch.device or None
        Target device.  If ``None``, uses :func:`get_device`.

    Returns
    -------
    torch.Tensor
        Shape ``(B, 625)`` on the target device.
    """
    if device is None:
        device = get_device()
    if isinstance(masks, np.ndarray) and masks.ndim == 2:
        arr = masks
    else:
        arr = np.stack(masks, axis=0)
    return torch.from_numpy(arr).to(device)


@torch.no_grad()
def predict(
    model: KhreibagaNet,
    observations: list[np.ndarray] | np.ndarray,
    action_masks: list[np.ndarray] | np.ndarray | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run batched inference and return numpy results.

    Parameters
    ----------
    model : KhreibagaNet
        The neural network (should already be on *device* and in eval mode).
    observations : list of np.ndarray or np.ndarray
        Observation tensors, each ``(7, 5, 5)`` float32.
    action_masks : list of np.ndarray or np.ndarray or None
        Optional ``(625,)`` float32 masks.  When provided, illegal actions
        are zeroed out in the returned probability distribution.
    device : torch.device or None
        Device to run on.  If ``None``, uses :func:`get_device`.

    Returns
    -------
    policy_probs : np.ndarray
        Shape ``(B, 625)``.  If *action_masks* is provided, illegal actions
        have probability 0 and the legal actions are re-normalised.
    values : np.ndarray
        Shape ``(B,)`` in ``[-1, 1]``.
    """
    if device is None:
        device = next(model.parameters()).device

    obs_t = batch_observations(observations, device=device)
    logits, values_t = model(obs_t)

    # Apply action mask: set illegal logits to -inf before softmax
    if action_masks is not None:
        mask_t = batch_action_masks(action_masks, device=device)
        logits = logits.masked_fill(mask_t == 0, float("-inf"))

    probs = F.softmax(logits, dim=1)

    # Handle fully-masked rows (all -inf -> NaN after softmax) by replacing
    # with uniform over the mask. This is a safety net; should not happen
    # if the game engine is correct (there's always at least one legal action
    # when the game is not over).
    if action_masks is not None:
        nan_rows = torch.isnan(probs).any(dim=1)
        if nan_rows.any():
            mask_t_safe = batch_action_masks(action_masks, device=device)
            uniform = mask_t_safe / mask_t_safe.sum(dim=1, keepdim=True).clamp(min=1)
            probs[nan_rows] = uniform[nan_rows]

    return probs.cpu().numpy(), values_t.squeeze(-1).cpu().numpy()
