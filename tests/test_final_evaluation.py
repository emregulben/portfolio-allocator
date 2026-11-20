import os
import sys
import pytest
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.final_evaluation import compute_portfolio_metrics, sample_unconditioned_path, PAIRED_COMPARISONS
from agents.diffusion import MeanCovMLPDenoiser, PortDiff


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


def _small_uncond_model():
    torch.manual_seed(0)
    denoiser = MeanCovMLPDenoiser(n_assets=4, action_dim=5, hidden_dim=32, condition_on_prev_weights=False)
    model = PortDiff(denoiser=denoiser, num_timesteps=10, beta_schedule="linear")
    return model.eval()


def test_unconditioned_path_same_market_every_day_gives_same_portfolio():
    """
    Sampler confound check: if the market looks the same on every day, a deterministic
    policy must pick the same portfolio every day (zero turnover). Before the fix, each
    day got a different starting noise, so the portfolio jittered with no market reason.
    """
    model = _small_uncond_model()
    one_day = torch.randn(1, 60, 4) * 0.02
    states_i = one_day.repeat(6, 1, 1)           # 6 days, identical market

    with torch.no_grad():
        w = sample_unconditioned_path(model, states_i, seed_index=0, num_seeds=3, sampler="ddim")
    assert torch.allclose(w, w[:1].expand_as(w), atol=1e-6)


def test_unconditioned_path_uses_same_noise_as_conditioned_rollout():
    """
    Fair comparison: market seed i must get the same starting noise as in the conditioned
    rollout, which samples all seeds in one batch (row i = seed i) every day.
    """
    model = _small_uncond_model()
    num_seeds = 3
    one_day_per_seed = torch.randn(num_seeds, 60, 4) * 0.02

    with torch.no_grad():
        batch_style = model.sample(one_day_per_seed, sampler="ddim")   # like rollout_conditioned_batch
        for i in range(num_seeds):
            path_style = sample_unconditioned_path(
                model, one_day_per_seed[i:i + 1], seed_index=i, num_seeds=num_seeds, sampler="ddim"
            )
            assert torch.allclose(path_style[0], batch_style[i], atol=1e-6)


@pytest.mark.parametrize("bad, why", [
    ("zeros", "a failed GPU step can return all-zero weights"),
    ("nan", "a failed GPU step can return NaN weights"),
    ("negative", "short positions are not allowed"),
])
def test_metrics_refuse_invalid_weights(bad, why):
    """Invalid portfolios must stop the evaluation instead of being saved as results."""
    T, action_dim = 10, 3
    weights = np.full((T, action_dim), 1.0 / action_dim)
    if bad == "zeros":
        weights[4] = 0.0
    elif bad == "nan":
        weights[4, 1] = np.nan
    else:
        weights[4] = [1.2, -0.2, 0.0]
    returns = np.zeros((T, action_dim))

    with pytest.raises(ValueError, match="Invalid portfolio weights"):
        compute_portfolio_metrics(weights, returns, cost_c=0.0005)



def test_wallet_effect_is_tested_directly():
    """Contribution 1 (the wallet input) needs Cond vs Uncond tests for both IL and RL."""
    pairs = {(a, b) for _, a, b in PAIRED_COMPARISONS}
    assert ("il_conditioned_det", "il_unconditioned_det") in pairs
    assert ("rl_conditioned_det", "rl_unconditioned_det") in pairs
