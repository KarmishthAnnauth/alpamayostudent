"""Teacher emits (curvature, accel) per waypoint; the tokenizer uses (accel, curvature)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.teacher.wrapper import swap_action_dims  # noqa: E402


def test_swaps_within_waypoints_not_across():
    """[w0d0 w0d1 w1d0 w1d1 ...] -> [w0d1 w0d0 w1d1 w1d0 ...]."""
    t = torch.arange(8)
    assert swap_action_dims(t).tolist() == [1, 0, 3, 2, 5, 4, 7, 6]


def test_is_its_own_inverse():
    """Swapping twice is the identity."""
    t = torch.randint(0, 3000, (128,))
    assert torch.equal(swap_action_dims(swap_action_dims(t)), t)


def test_is_not_the_dim_major_transpose():
    """Interleaved swap differs from the dim-major transpose."""
    t = torch.arange(8)
    transposed = t.reshape(2, -1).t().reshape(-1)
    assert transposed.tolist() == [0, 4, 1, 5, 2, 6, 3, 7]
    assert not torch.equal(swap_action_dims(t), transposed)


def test_preserves_shape_and_batches_over_leading_dims():
    t = torch.randint(0, 3000, (4, 128))
    out = swap_action_dims(t)
    assert out.shape == t.shape
    for row in range(4):
        assert torch.equal(out[row], swap_action_dims(t[row]))


def test_dim_identity_is_actually_exchanged():
    """Pinned and varying dims swap places."""
    waypoints = 64
    pinned = torch.full((waypoints,), 1500)
    varying = torch.arange(waypoints) + 1400
    encode_order = torch.stack([varying, pinned], dim=-1).reshape(-1)

    emitted = swap_action_dims(encode_order)
    assert torch.equal(emitted.reshape(-1, 2)[:, 0], pinned)
    assert torch.equal(emitted.reshape(-1, 2)[:, 1], varying)
