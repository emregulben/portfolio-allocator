import os
import sys
import torch
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.replay_buffer import AWRReplayBuffer
from scripts.run_rl_finetuning import compute_daily_utility


def test_daily_utility_shape_and_zero_risk_cash():
    """Verifies shapes and confirms 100% Cash incurs zero risk penalty."""
    T = 100
    window = 60
    n_assets = 10
    action_dim = 11

    states = torch.randn(T, window, n_assets)
    returns = torch.randn(T, action_dim) * 0.01

    cash_actions = torch.zeros(T, action_dim)
    cash_actions[:, -1] = 1.0

    utility = compute_daily_utility(cash_actions, returns, states, gamma=2.0, cost_c=0.0005)

    assert utility.shape == (T,), f"Expected shape ({T},), got {utility.shape}"
    expected_cash_return = returns[:, -1]
    assert torch.allclose(utility, expected_cash_return, atol=1e-6), "Cash utility mismatch."


def test_daily_utility_turnover_penalty():
    """Verifies that static portfolios incur zero fees, while rebalancing incurs exact fees."""
    T = 10
    window = 60
    n_assets = 10
    action_dim = 11

    states = torch.zeros(T, window, n_assets)
    returns = torch.zeros(T, action_dim)

    # Static portfolio -> 0 turnover
    static_actions = torch.zeros(T, action_dim)
    static_actions[:, 0] = 1.0
    u_static = compute_daily_utility(static_actions, returns, states, gamma=0.0, cost_c=0.0010)

    # Alternating portfolio -> L1 turnover = 2.0 per step
    alt_actions = torch.zeros(T, action_dim)
    for t in range(T):
        alt_actions[t, t % 2] = 1.0

    cost_c = 0.0010
    u_alt = compute_daily_utility(alt_actions, returns, states, gamma=0.0, cost_c=cost_c)

    assert u_static[1:].abs().max().item() == 0.0, "Static portfolio should pay zero fees."
    expected_penalty = -2.0 * cost_c
    assert torch.allclose(u_alt[1:], torch.full((T - 1,), expected_penalty), atol=1e-6), "Turnover fee mismatch."


def test_daily_utility_ground_truth_manual_math():
    """
    Verifies that the vectorized PyTorch utility matches an unvectorized,
    manual step-by-step mathematical loop on arbitrary mixed portfolios.
    """
    torch.manual_seed(42)
    T = 20
    window = 60
    n_assets = 10
    action_dim = 11
    gamma = 2.5
    cost_c = 0.0008

    states = torch.randn(T, window, n_assets)
    returns = torch.randn(T, action_dim) * 0.02
    raw_actions = torch.rand(T, action_dim)
    actions = raw_actions / raw_actions.sum(dim=1, keepdim=True)

    # Vectorized function
    computed_utility = compute_daily_utility(actions, returns, states, gamma=gamma, cost_c=cost_c)

    # Manual ground-truth calculation step-by-step
    manual_utility = torch.zeros(T)
    for t in range(T):
        # 1. Return
        ret = torch.dot(actions[t], returns[t])
        
        # 2. Risk (unbiased sample covariance of window)
        window_data = states[t]  # (window, 10)
        mu = window_data.mean(dim=0)
        centered = window_data - mu
        manual_cov = torch.matmul(centered.T, centered) / (window - 1)
        w_risky = actions[t, :-1]
        var = torch.dot(w_risky, torch.mv(manual_cov, w_risky))
        
        # 3. Turnover fee
        turnover = torch.norm(actions[t] - actions[t - 1], p=1) if t > 0 else torch.tensor(0.0)
        
        manual_utility[t] = ret - (gamma / 2.0) * var - cost_c * turnover

    assert torch.allclose(computed_utility, manual_utility, atol=1e-5), "Vectorized utility differs from manual math."


def test_daily_utility_risk_penalty_monotonicity():
    """
    Verifies that for identical returns, allocating to high-volatility assets
    incurs a strictly lower utility than allocating to low-volatility assets.
    """
    T = 5
    window = 60
    n_assets = 10
    action_dim = 11

    # Asset 0 is low volatility, Asset 1 is high volatility
    states = torch.zeros(T, window, n_assets)
    states[:, :, 0] = torch.randn(T, window) * 0.005  # low vol
    states[:, :, 1] = torch.randn(T, window) * 0.050  # high vol

    returns = torch.zeros(T, action_dim)  # identical 0 return

    actions_low_vol = torch.zeros(T, action_dim)
    actions_low_vol[:, 0] = 1.0

    actions_high_vol = torch.zeros(T, action_dim)
    actions_high_vol[:, 1] = 1.0

    u_low = compute_daily_utility(actions_low_vol, returns, states, gamma=2.0, cost_c=0.0)
    u_high = compute_daily_utility(actions_high_vol, returns, states, gamma=2.0, cost_c=0.0)

    # High vol penalty must be strictly greater -> utility strictly lower
    assert (u_low > u_high).all(), "High-volatility portfolio must have strictly lower utility."


def test_paired_advantage_identity():
    """Verifies that identical policies produce strictly zero advantage."""
    T = 50
    window = 60
    n_assets = 10
    action_dim = 11

    states = torch.randn(T, window, n_assets)
    returns = torch.randn(T, action_dim)
    actions = torch.rand(T, action_dim)
    actions = actions / actions.sum(dim=1, keepdim=True)

    u1 = compute_daily_utility(actions, returns, states, gamma=2.0, cost_c=0.0005)
    u2 = compute_daily_utility(actions, returns, states, gamma=2.0, cost_c=0.0005)

    delta = u1 - u2
    assert torch.allclose(delta, torch.zeros(T), atol=1e-7), "Identical policies must yield zero advantage."


def test_replay_buffer_daily_advantage_tensor_and_normalization():
    """
    Verifies that AWRReplayBuffer accepts daily advantage tensors,
    and that normalizing advantages results in mean=0 and std=1.
    """
    buffer = AWRReplayBuffer()
    T = 252
    window = 60
    n_assets = 10
    action_dim = 11

    states = torch.randn(T, window, n_assets)
    actions = torch.randn(T, action_dim)
    daily_advantages = torch.randn(T) * 0.005 + 0.01  # Arbitrary mean and scale

    buffer.add_trajectory(states, actions, daily_advantages)

    # Perform advantage normalization as done in run_rl_finetuning
    all_adv = torch.cat(buffer.advantages, dim=0)
    adv_mean = all_adv.mean()
    adv_std = all_adv.std() + 1e-8
    for i in range(len(buffer.advantages)):
        buffer.advantages[i] = (buffer.advantages[i] - adv_mean) / adv_std

    buffer.compile_buffer()

    assert len(buffer) == T
    assert buffer.compiled_advantages.shape == (T, 1)
    assert abs(buffer.compiled_advantages.mean().item()) < 1e-5, "Normalized mean should be 0.0."
    assert abs(buffer.compiled_advantages.std().item() - 1.0) < 1e-3, "Normalized std should be 1.0."