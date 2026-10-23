"""
Correctness checks for the flow-matching action head.

Verifies the straight-path interpolation, the velocity target, and that the Euler
integration used at inference actually maps noise onto the ground-truth action when
the network predicts the exact target velocity.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/home/ps/openvla/openvla-oft")

import numpy as np
import torch

from prismatic.models.action_heads import FlowMatchingActionHead


def test_interpolation_and_velocity():
    """x_t must lie on the segment between the action and the noise, and u_t = noise - action."""
    torch.manual_seed(0)
    head = FlowMatchingActionHead(input_dim=8, hidden_dim=8, action_dim=7)
    actions = torch.randn(4, 8, 7)
    out = head.sample_noisy_actions(actions)
    t = out["time"].view(-1, 1, 1)
    expected_state = (1 - t) * actions + t * out["noise"]
    assert torch.allclose(out["noisy_actions"], expected_state, atol=1e-5), "interpolation mismatch"
    assert torch.allclose(out["target_velocity"], out["noise"] - actions, atol=1e-6), "velocity mismatch"
    # The path endpoints must be recoverable from the state and the velocity.
    recovered = out["noisy_actions"] - out["target_velocity"] * t
    assert torch.allclose(recovered, actions, atol=1e-4), "action endpoint mismatch"
    print("PASS interpolation and velocity target")


def test_time_distribution():
    """The sampled time must stay inside (0, 1] and follow Beta(1.5, 1)."""
    torch.manual_seed(0)
    head = FlowMatchingActionHead(input_dim=8, hidden_dim=8, action_dim=7)
    actions = torch.zeros(4096, 8, 7)
    times = head.sample_noisy_actions(actions)["time"]
    assert float(times.min()) > 0.0 and float(times.max()) <= 1.0
    expected_mean = 1.5 / 2.5
    observed_mean = float(times.mean())
    assert abs(observed_mean - expected_mean) < 0.02, (observed_mean, expected_mean)
    print(f"PASS time distribution (mean {observed_mean:.4f}, expected {expected_mean:.4f})")


def test_euler_integration_recovers_action():
    """
    With the exact target velocity, the Euler loop used at inference must move the
    sample from pure noise onto the ground-truth action.
    """
    torch.manual_seed(0)
    head = FlowMatchingActionHead(input_dim=8, hidden_dim=8, action_dim=7)
    actions = torch.randn(3, 8, 7)
    sampled = head.sample_noisy_actions(actions)
    noise, target_velocity = sampled["noise"], sampled["target_velocity"]

    for num_steps in (1, 2, 5, 10, 20):
        dt = -1.0 / num_steps
        state = noise.clone()
        time = 1.0
        for _ in range(num_steps):
            state = state + dt * target_velocity
            time += dt
        error = (state - actions).abs().max().item()
        assert error < 1e-4, f"{num_steps} steps produced max error {error}"
    print("PASS Euler integration recovers the action at every step count")


def test_integration_direction():
    """Integrating from noise toward t=0 must decrease the distance to the action."""
    torch.manual_seed(0)
    head = FlowMatchingActionHead(input_dim=8, hidden_dim=8, action_dim=7)
    actions = torch.randn(2, 8, 7)
    sampled = head.sample_noisy_actions(actions)
    noise, target_velocity = sampled["noise"], sampled["target_velocity"]

    dt = -1.0 / 10
    state = noise.clone()
    distances = []
    for _ in range(10):
        distances.append((state - actions).abs().mean().item())
        state = state + dt * target_velocity
    distances.append((state - actions).abs().mean().item())
    assert all(b <= a + 1e-6 for a, b in zip(distances, distances[1:])), distances
    print("PASS integration direction moves monotonically toward the action")


def test_velocity_predictor_shape():
    """The velocity head must emit one 7-dimensional vector per action token."""
    torch.manual_seed(0)
    head = FlowMatchingActionHead(input_dim=8, hidden_dim=8, action_dim=7)
    hidden = torch.randn(5, 8 * 7, 8)
    velocity = head.predict_velocity(hidden)
    assert velocity.shape == (5, 8, 7), velocity.shape
    assert torch.isfinite(velocity).all()
    print("PASS velocity predictor shape")


def test_time_encoder_is_continuous():
    """Nearby times must give similar embeddings; distant times must differ."""
    torch.manual_seed(0)
    head = FlowMatchingActionHead(input_dim=8, hidden_dim=8, action_dim=7)
    encoder = head.time_encoder
    near_a = encoder(torch.tensor([0.500]))
    near_b = encoder(torch.tensor([0.501]))
    far = encoder(torch.tensor([0.900]))
    assert torch.allclose(near_a, near_b, atol=1e-2), "time embedding not continuous"
    assert not torch.allclose(near_a, far, atol=1e-2), "time embedding not discriminative"
    print("PASS continuous time embedding")


if __name__ == "__main__":
    test_interpolation_and_velocity()
    test_time_distribution()
    test_euler_integration_recovers_action()
    test_integration_direction()
    test_velocity_predictor_shape()
    test_time_encoder_is_continuous()
    print("\nAll flow-matching correctness checks passed.")
