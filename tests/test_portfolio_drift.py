import os
import sys
import pytest
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.final_evaluation import compute_portfolio_metrics
from scripts.run_rl_finetuning import compute_daily_utility


def test_hand_calculated_rebalancing_drift():
    """
    Verifies portfolio drift against hand-calculated values:
    Day 0: Allocation w_0 = [0.50, 0.50]. Returns r_0 = [+0.10, -0.10].
    Portfolio Return = 0.5 * 0.10 + 0.5 * (-0.10) = 0.00.
    Drifted Weights w_0^+ = [0.50 * 1.10 / 1.0, 0.50 * 0.90 / 1.0] = [0.55, 0.45].
    Day 1: Target w_1 = [0.50, 0.50].
    Turnover at Day 1 = |0.50 - 0.55| + |0.50 - 0.45| = 0.05 + 0.05 = 0.10.
    """
    weights = np.array([
        [0.50, 0.50],
        [0.50, 0.50],
    ])
    returns = np.array([
        [0.10, -0.10],
        [0.00, 0.00],
    ])

    metrics = compute_portfolio_metrics(weights, returns, cost_c=0.001)

    assert metrics["mean_turnover"] == pytest.approx(0.10, abs=1e-7), (
        "Rebalancing back to 50/50 after +10%/-10% returns must incur exactly 0.10 turnover."
    )


def test_passive_buy_and_hold_zero_turnover():
    """
    Verifies that a passive buy-and-hold portfolio (where target equals drifted weights)
    incurs exactly 0.0 turnover and zero transaction cost drag.
    """
    T = 10
    action_dim = 4
    np.random.seed(42)

    returns = np.random.randn(T, action_dim) * 0.02

    # Construct pure buy-and-hold weights by letting w_0 drift without rebalancing
    weights = np.zeros((T, action_dim))
    weights[0] = np.array([0.40, 0.30, 0.20, 0.10])

    for t in range(1, T):
        w_prev = weights[t - 1]
        r_prev = returns[t - 1]
        gross_ret = np.sum(w_prev * r_prev)
        weights[t] = (w_prev * (1.0 + r_prev)) / (1.0 + gross_ret)

    metrics = compute_portfolio_metrics(weights, returns, cost_c=0.0005)

    assert metrics["mean_turnover"] == pytest.approx(0.0, abs=1e-7), (
        "Passive buy-and-hold must incur exactly 0.0 turnover."
    )
    assert metrics["gross_sharpe"] == pytest.approx(metrics["net_sharpe"], abs=1e-7), (
        "Gross and Net Sharpe must be identical for buy-and-hold."
    )


def test_single_asset_zero_drift_invariance():
    """Verifies that 100% single-asset portfolios have zero drift turnover across varying returns."""
    T = 15
    action_dim = 5
    weights = np.zeros((T, action_dim))
    weights[:, 0] = 1.0  # 100% in asset 0

    np.random.seed(123)
    returns = np.random.randn(T, action_dim) * 0.03 + 0.001

    metrics = compute_portfolio_metrics(weights, returns, cost_c=0.0005)

    assert metrics["mean_turnover"] == pytest.approx(0.0, abs=1e-7), (
        "A single-asset portfolio cannot drift relative to its own target."
    )


def test_pytorch_numpy_turnover_consistency():
    """Verifies numerical equivalence between PyTorch and NumPy drift turnover calculations."""
    T = 20
    action_dim = 4
    n_assets = action_dim - 1
    window_size = 30
    np.random.seed(999)

    raw_weights = np.random.rand(T, action_dim)
    weights_np = raw_weights / raw_weights.sum(axis=1, keepdims=True)
    returns_np = np.random.randn(T, action_dim) * 0.015

    # 1. NumPy Turnover
    gross_returns = np.sum(weights_np * returns_np, axis=1)
    w_prev = weights_np[:-1]
    r_prev = returns_np[:-1]
    gross_prev = gross_returns[:-1, np.newaxis]
    w_prev_drifted = (w_prev * (1.0 + r_prev)) / (1.0 + gross_prev)
    expected_turnover = np.sum(np.abs(weights_np[1:] - w_prev_drifted), axis=1)

    # 2. PyTorch Turnover via compute_daily_utility
    actions_th = torch.tensor(weights_np, dtype=torch.float32)
    returns_th = torch.tensor(returns_np, dtype=torch.float32)
    states_th = torch.randn(T, window_size, n_assets, dtype=torch.float32) * 0.02

    # Set cost_c = 1.0, gamma = 0.0 to isolate the turnover penalty directly
    utility = compute_daily_utility(actions_th, returns_th, states_th, gamma=0.0, cost_c=1.0)
    gross_pt = (actions_th * returns_th).sum(dim=1)
    pt_turnover = (gross_pt - utility)[1:].numpy()

    np.testing.assert_allclose(
        pt_turnover,
        expected_turnover,
        rtol=1e-5,
        atol=1e-6,
        err_msg="PyTorch and NumPy drift turnover implementations must yield identical results."
    )