import os
import sys
import pytest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.final_evaluation import compute_portfolio_metrics


def test_metrics_static_portfolio_zero_turnover():
    """Verifies that static portfolios incur zero turnover and gross == net Sharpe."""
    T = 100
    action_dim = 11

    # Static portfolio (100% in asset 0)
    weights = np.zeros((T, action_dim))
    weights[:, 0] = 1.0

    np.random.seed(42)
    returns = np.random.randn(T, action_dim) * 0.01 + 0.001

    metrics = compute_portfolio_metrics(weights, returns, cost_c=0.0005)

    assert metrics["mean_turnover"] == pytest.approx(0.0, abs=1e-6), "Static portfolio turnover should be 0.0."
    assert metrics["gross_sharpe"] == pytest.approx(metrics["net_sharpe"], abs=1e-6), "Gross and Net Sharpe must be identical when turnover is 0."


def test_metrics_turnover_penalty():
    """Verifies that rebalancing incurs the exact transaction fee penalty."""
    T = 20
    action_dim = 11
    cost_c = 0.0010  # 10 bps

    # Alternating portfolio: shifts 100% between asset 0 and asset 1 every day
    # L1 turnover per step = |1 - 0| + |0 - 1| = 2.0
    weights = np.zeros((T, action_dim))
    for t in range(T):
        weights[t, t % 2] = 1.0

    returns = np.zeros((T, action_dim))
    returns[:, 0] = 0.005
    returns[:, 1] = 0.005

    metrics = compute_portfolio_metrics(weights, returns, cost_c=cost_c)

    assert metrics["mean_turnover"] == pytest.approx(2.0, abs=1e-6), "Turnover should be exactly 2.0 per day."
    assert metrics["net_sharpe"] < metrics["gross_sharpe"], "Net Sharpe must be strictly less than Gross Sharpe."
    
    # Day 0 incurs 0 fees (return = 0.005), remaining T-1 days incur 2.0 * cost_c fees (return = 0.003)
    expected_daily_net = (0.005 + (T - 1) * (0.005 - cost_c * 2.0)) / T
    expected_ann_net_return = expected_daily_net * 252.0
    assert metrics["net_return"] == pytest.approx(expected_ann_net_return, abs=1e-5), "Net return fee deduction mismatch."

def test_metrics_exact_maximum_drawdown():
    """
    Verifies Maximum Drawdown calculation against a hand-calculated path:
    Day 0: +10% (Wealth = 1.10)
    Day 1: -20% (Wealth = 1.10 * 0.80 = 0.88)
    Peak = 1.10, Trough = 0.88 -> Drawdown = (0.88 - 1.10) / 1.10 = -0.20 (-20%)
    """
    weights = np.ones((2, 1))  # 1 asset, 100% allocation
    returns = np.array([[0.10], [-0.20]])  # Day 0: +10%, Day 1: -20%

    metrics = compute_portfolio_metrics(weights, returns, cost_c=0.0)

    expected_mdd = -0.20
    assert metrics["max_drawdown"] == pytest.approx(expected_mdd, abs=1e-6), f"Expected MDD {expected_mdd}, got {metrics['max_drawdown']}"


def test_metrics_sortino_downside_isolation():
    """
    Verifies that Sortino ratio does not penalize upside volatility.
    If all returns are strictly positive, downside deviation is zero.
    """
    T = 50
    weights = np.ones((T, 1))
    
    # Returns alternate between +1% and +5% (volatile, but strictly positive)
    returns = np.array([[0.01 if t % 2 == 0 else 0.05] for t in range(T)])

    metrics = compute_portfolio_metrics(weights, returns, cost_c=0.0)

    # Sortino should be massive because there are zero downside returns
    assert metrics["net_sortino"] > metrics["net_sharpe"] * 10, "Sortino should be vastly higher than Sharpe for purely positive returns."