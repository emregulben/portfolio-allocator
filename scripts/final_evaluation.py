import os
import sys
import yaml
import torch
import logging
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.loader import MarketDataLoader
from simulator.hmm import HybridJumpsHMM
from simulator.single_index import SingleIndexModel
from experts.markowitz import MarkowitzExpert
from agents.diffusion import MeanCovMLPDenoiser, PortDiff, map_to_feasible_portfolio


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def compute_portfolio_metrics(weights_arr, returns_arr, cost_c=0.0005):
    """
    Computes institutional risk, return, and execution metrics from daily allocations,
    accounting for asset price drift between consecutive rebalancing periods.

    Args:
        weights_arr (np.ndarray): Target portfolio allocations of shape (T, action_dim).
        returns_arr (np.ndarray): Realized asset returns of shape (T, action_dim).
        cost_c (float): Proportional transaction cost parameter per unit turnover.

    Returns:
        dict: Dictionary containing gross/net returns, Sharpe, Sortino, MDD, and turnover.
    """
    T = len(weights_arr)

    # 1. Daily Gross Returns: w_t^T r_{t+1}
    gross_returns = np.sum(weights_arr * returns_arr, axis=1)

    # 2. Daily Turnover Accounting for Portfolio Drift
    turnover = np.zeros(T)
    if T > 1:
        w_prev = weights_arr[:-1]
        r_prev = returns_arr[:-1]
        gross_ret_prev = gross_returns[:-1, np.newaxis]

        # Drifted weights at the end of period t-1 prior to rebalancing
        w_prev_drifted = (w_prev * (1.0 + r_prev)) / np.maximum(1.0 + gross_ret_prev, 1e-8)

        # Turnover required to move from drifted weights to new target weights
        turnover[1:] = np.sum(np.abs(weights_arr[1:] - w_prev_drifted), axis=1)

    net_returns = gross_returns - cost_c * turnover

    # 3. Annualized Returns & Volatilities
    ann_net_return = np.mean(net_returns) * 252.0
    ann_net_vol = np.std(net_returns) * np.sqrt(252.0) + 1e-9
    ann_gross_vol = np.std(gross_returns) * np.sqrt(252.0) + 1e-9

    # 4. Gross and Net Sharpe Ratios
    gross_sharpe = (np.mean(gross_returns) * 252.0) / ann_gross_vol
    net_sharpe = ann_net_return / ann_net_vol

    # 5. Net Sortino Ratio (downside deviation only)
    downside_returns = np.minimum(net_returns, 0.0)
    downside_vol = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252.0) + 1e-9
    net_sortino = ann_net_return / downside_vol

    # 6. Maximum Drawdown (MDD) from cumulative wealth
    cum_wealth = np.cumprod(1.0 + net_returns)
    running_max = np.maximum.accumulate(cum_wealth)
    drawdowns = (cum_wealth - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    # 7. Average Daily Turnover
    mean_turnover = np.mean(turnover[1:]) if T > 1 else 0.0

    return {
        "net_return": float(ann_net_return),
        "gross_sharpe": float(gross_sharpe),
        "net_sharpe": float(net_sharpe),
        "net_sortino": float(net_sortino),
        "max_drawdown": float(max_drawdown),
        "mean_turnover": float(mean_turnover),
    }


def compute_paired_stats(metric_a, metric_b):
    """
    Computes paired difference statistics, 95% confidence intervals, and p-values
    across matched market seeds.

    Args:
        metric_a (list[float]): Metric values for model A (e.g., RL Cond).
        metric_b (list[float]): Metric values for model B (e.g., IL Cond).

    Returns:
        dict: Mean difference, standard error, 95% CI bounds, t-statistic, and p-value.
    """
    diffs = np.array(metric_a) - np.array(metric_b)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    se_diff = std_diff / np.sqrt(n)

    # Two-sided critical t value for 95% confidence interval
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Paired two-sided t-test
    t_stat = mean_diff / se_diff if se_diff > 1e-12 else 0.0
    p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=n - 1)))

    return {
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "ci_95": (ci_lower, ci_upper),
        "t_stat": t_stat,
        "p_value": p_val,
    }


def load_evaluation_environment():
    """
    Initializes market data, HMM simulator, classical and cost-aware experts,
    and both unconditioned and conditioned neural policies.

    Returns:
        tuple: (config, hmm, sim_model, expert, expert_tc, il_uncond, rl_uncond, il_cond, rl_cond, device)
    """
    logger = setup_logger("Final_Evaluation")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(root_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Evaluation running on device: {device}")

    # 1. Market Data & HMM Simulator Setup
    market_ticker = config["data"]["market_ticker"]
    stock_tickers = config["data"]["stock_tickers"]

    loader = MarketDataLoader(
        tickers=[market_ticker] + stock_tickers,
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"]
    )
    data = loader.fetch_data()

    hmm = HybridJumpsHMM(n_states=config["hmm"]["n_states"], emission_dist="gaussian")
    hmm.fit(data[market_ticker])
    hmm.epsilon = 0.0025
    hmm.lambd = 85.0
    logger.info(f"HMM Simulator configured with crash parameters: epsilon={hmm.epsilon}, lambda={hmm.lambd}")

    sim_model = SingleIndexModel(tickers=stock_tickers, emission_dist="gaussian")
    sim_model.fit(pd.DataFrame(data[stock_tickers]), data[market_ticker])

    # 2. Classical Markowitz and Transaction-Cost-Aware Markowitz Experts
    cost_c = config["expert"]["transaction_cost"]
    expert = MarkowitzExpert(
        risk_aversion=config["expert"]["risk_aversion"],
        rolling_window=config["expert"]["rolling_window"],
        max_weight=config["expert"]["max_weight"],
        annual_risk_free_rate=config["expert"]["annual_risk_free_rate"],
        transaction_cost=0.0,
    )
    expert_tc = MarkowitzExpert(
        risk_aversion=config["expert"]["risk_aversion"],
        rolling_window=config["expert"]["rolling_window"],
        max_weight=config["expert"]["max_weight"],
        annual_risk_free_rate=config["expert"]["annual_risk_free_rate"],
        transaction_cost=cost_c,
    )

    action_dim = len(stock_tickers) + 1

    # 3. Unconditioned Policies Setup
    il_uncond_denoiser = MeanCovMLPDenoiser(n_assets=len(stock_tickers), action_dim=action_dim, hidden_dim=1024, condition_on_prev_weights=False)
    il_uncond = PortDiff(denoiser=il_uncond_denoiser, num_timesteps=100, beta_schedule="linear")
    il_path = os.path.join(root_dir, "il_winner.pth")
    il_uncond.load_state_dict(torch.load(il_path, map_location=device))
    il_uncond.to(device).eval()
    logger.info(f"Loaded unconditioned IL model from {il_path}")

    rl_uncond_denoiser = MeanCovMLPDenoiser(n_assets=len(stock_tickers), action_dim=action_dim, hidden_dim=1024, condition_on_prev_weights=False)
    rl_uncond = PortDiff(denoiser=rl_uncond_denoiser, num_timesteps=100, beta_schedule="linear")
    rl_path = os.path.join(root_dir, "rl_winner.pth")
    rl_uncond.load_state_dict(torch.load(rl_path, map_location=device))
    rl_uncond.to(device).eval()
    logger.info(f"Loaded unconditioned RL model from {rl_path}")

    # 4. Conditioned Policies Setup
    il_cond_denoiser = MeanCovMLPDenoiser(n_assets=len(stock_tickers), action_dim=action_dim, hidden_dim=1024, condition_on_prev_weights=True)
    il_cond = PortDiff(denoiser=il_cond_denoiser, num_timesteps=100, beta_schedule="linear")
    il_cond_path = os.path.join(root_dir, "il_conditioned.pth")
    il_cond.load_state_dict(torch.load(il_cond_path, map_location=device))
    il_cond.to(device).eval()
    logger.info(f"Loaded conditioned IL model from {il_cond_path}")

    rl_cond_denoiser = MeanCovMLPDenoiser(n_assets=len(stock_tickers), action_dim=action_dim, hidden_dim=1024, condition_on_prev_weights=True)
    rl_cond = PortDiff(denoiser=rl_cond_denoiser, num_timesteps=100, beta_schedule="linear")
    rl_cond_path = os.path.join(root_dir, "rl_conditioned.pth")
    rl_cond.load_state_dict(torch.load(rl_cond_path, map_location=device))
    rl_cond.to(device).eval()
    logger.info(f"Loaded conditioned RL model from {rl_cond_path}")

    return config, hmm, sim_model, expert, expert_tc, il_uncond, rl_uncond, il_cond, rl_cond, device


def rollout_conditioned_batch(model, batch_states, batch_returns, device, sampler: str = "ddim"):
    """
    Executes closed-loop autoregressive rollout in parallel across independent market seeds.
    Tracks intra-period portfolio price drift prior to each rebalancing step.

    Args:
        model (PortDiff): Conditioned diffusion policy model.
        batch_states (torch.Tensor): Market return window tensor of shape (B, T, window, n_assets).
        batch_returns (torch.Tensor): Realized returns of shape (B, T, action_dim).
        device (torch.device): Computation device.
        sampler (str): Reverse sampling strategy ("ddim", "ddpm_det", "ddpm_stoch"). Defaults to "ddim".

    Returns:
        np.ndarray: Portfolio weights array of shape (B, T, action_dim).
    """
    num_seeds, T_steps, window, n_assets = batch_states.shape
    action_dim = batch_returns.shape[2]

    # Initialize Day 0 portfolio at 100% Cash
    prev_w = torch.zeros((num_seeds, action_dim), device=device)
    prev_w[:, -1] = 1.0

    all_weights = []
    model.eval()

    for t in range(T_steps):
        st_t = batch_states[:, t]   # (B, window, n_assets)
        pw_t = prev_w.clone()        # (B, action_dim)

        with torch.no_grad():
            w_t = model.sample(st_t, prev_weights=pw_t, sampler=sampler)
            all_weights.append(w_t)

            # Intra-period price drift to the end of period t
            r_t = batch_returns[:, t]
            gross_ret_t = (w_t * r_t).sum(dim=1, keepdim=True)
            denom = torch.clamp(1.0 + gross_ret_t, min=1e-8)
            prev_w = (w_t * (1.0 + r_t)) / denom

    weights_tensor = torch.stack(all_weights, dim=1)  # (B, T, action_dim)
    return weights_tensor.cpu().numpy()


def run_final_evaluation(num_seeds=30, start_seed=9000):
    """
    Executes an out-of-sample statistical comparison across 30 unseen market seeds
    evaluating Classical Markowitz, Cost-Aware Markowitz, Unconditioned IL/RL,
    and Conditioned IL/RL policies with paired confidence intervals.

    Args:
        num_seeds (int): Number of independent market seeds to evaluate.
        start_seed (int): Base random seed ensuring unseen evaluation paths.
    """
    config, hmm, sim_model, expert, expert_tc, il_uncond, rl_uncond, il_cond, rl_cond, device = load_evaluation_environment()
    logger = setup_logger("Final_Evaluation")

    cost_c = config["expert"]["transaction_cost"]
    window = config["expert"]["rolling_window"]
    daily_rf = config["expert"]["annual_risk_free_rate"] / 252.0

    policy_keys = [
        "markowitz",
        "markowitz_tc",
        "il_unconditioned_det",
        "rl_unconditioned_det",
        "il_conditioned_det",
        "rl_conditioned_det",
    ]
    policy_headers = {
        "markowitz": "Markowitz",
        "markowitz_tc": "Markowitz (TC)",
        "il_unconditioned_det": "IL Uncond (Det)",
        "rl_unconditioned_det": "RL Uncond (Det)",
        "il_conditioned_det": "IL Cond (Det)",
        "rl_conditioned_det": "RL Cond (Det)",
    }

    results = {k: [] for k in policy_keys}
    results["il_cond_ddpm_det"] = []

    logger.info(f"Simulating {num_seeds} unseen market paths (seeds {start_seed} to {start_seed + num_seeds - 1})...")

    all_sim_stocks = []
    all_actual_returns = []
    all_states = []

    for i in tqdm(range(num_seeds), desc="Simulating Markets"):
        seed = start_seed + i
        np.random.seed(seed)

        sim_states = hmm.simulate_states(
            n_steps=config["hmm"]["simulation"]["n_steps"],
            epsilon=hmm.epsilon,
            lambd=hmm.lambd
        )
        sim_stocks = sim_model.simulate(hmm.decode_states(sim_states), random_seed=seed)

        sim_stocks_with_cash = sim_stocks.copy()
        sim_stocks_with_cash['Cash'] = daily_rf
        actual_returns_arr = sim_stocks_with_cash.to_numpy()[window:]

        state_returns_arr = sim_stocks.to_numpy()
        X = [state_returns_arr[t - window : t] for t in range(window, len(state_returns_arr))]

        all_sim_stocks.append(sim_stocks)
        all_actual_returns.append(actual_returns_arr)
        all_states.append(X)

    batch_states = torch.tensor(np.array(all_states), dtype=torch.float32).to(device)
    batch_returns = torch.tensor(np.array(all_actual_returns), dtype=torch.float32).to(device)

    # 1. Classical Markowitz and Cost-Aware Markowitz Benchmarks
    logger.info("Evaluating Classical and Cost-Aware Markowitz Experts...")
    for i in tqdm(range(num_seeds), desc="Markowitz Optimization"):
        markowitz_df = expert.generate_labels(all_sim_stocks[i], cost_aware=False)
        w_markowitz = markowitz_df.to_numpy()
        results["markowitz"].append(compute_portfolio_metrics(w_markowitz, all_actual_returns[i], cost_c=cost_c))

        markowitz_tc_df = expert_tc.generate_labels(all_sim_stocks[i], cost_aware=True)
        w_markowitz_tc = markowitz_tc_df.to_numpy()
        results["markowitz_tc"].append(compute_portfolio_metrics(w_markowitz_tc, all_actual_returns[i], cost_c=cost_c))

    # 2. Unconditioned Policies (Deterministic DDIM)
    logger.info("Evaluating Unconditioned IL and RL Policies...")
    for i in range(num_seeds):
        states_i = batch_states[i]
        with torch.no_grad():
            w_il_det = il_uncond.sample(states_i, sampler="ddim").cpu().numpy()
            w_rl_det = rl_uncond.sample(states_i, sampler="ddim").cpu().numpy()
        results["il_unconditioned_det"].append(compute_portfolio_metrics(w_il_det, all_actual_returns[i], cost_c=cost_c))
        results["rl_unconditioned_det"].append(compute_portfolio_metrics(w_rl_det, all_actual_returns[i], cost_c=cost_c))

    # 3. Conditioned Policies (Closed-Loop Autoregressive Rollout via DDIM)
    logger.info("Evaluating Conditioned IL Policy (Closed-Loop Autoregressive, DDIM)...")
    il_cond_weights = rollout_conditioned_batch(il_cond, batch_states, batch_returns, device, sampler="ddim")
    for i in range(num_seeds):
        results["il_conditioned_det"].append(compute_portfolio_metrics(il_cond_weights[i], all_actual_returns[i], cost_c=cost_c))

    logger.info("Evaluating Conditioned RL Policy (Closed-Loop Autoregressive, DDIM)...")
    rl_cond_weights = rollout_conditioned_batch(rl_cond, batch_states, batch_returns, device, sampler="ddim")
    for i in range(num_seeds):
        results["rl_conditioned_det"].append(compute_portfolio_metrics(rl_cond_weights[i], all_actual_returns[i], cost_c=cost_c))

    # 4. Sampler Robustness Check (Conditioned IL under zero-noise DDPM-Det)
    logger.info("Evaluating Conditioned IL Policy under DDPM-Det Sampler (Robustness Check)...")
    il_cond_ddpm_weights = rollout_conditioned_batch(il_cond, batch_states, batch_returns, device, sampler="ddpm_det")
    for i in range(num_seeds):
        results["il_cond_ddpm_det"].append(compute_portfolio_metrics(il_cond_ddpm_weights[i], all_actual_returns[i], cost_c=cost_c))

    # 5. Statistical Aggregation (Mean +/- Std)
    metrics_list = ["net_sharpe", "gross_sharpe", "net_sortino", "net_return", "max_drawdown", "mean_turnover"]
    metric_labels = {
        "net_sharpe": "Net Sharpe Ratio",
        "gross_sharpe": "Gross Sharpe Ratio",
        "net_sortino": "Net Sortino Ratio",
        "net_return": "Annual Net Return",
        "max_drawdown": "Maximum Drawdown",
        "mean_turnover": "Daily Turnover"
    }

    all_keys = policy_keys + ["il_cond_ddpm_det"]
    summary = {}
    for model_key in all_keys:
        summary[model_key] = {}
        for m in metrics_list:
            vals = [r[m] for r in results[model_key]]
            summary[model_key][m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals))
            }

    # 6. Formatted Results Table Display
    header_cols = [f"{'Metric':<20}"] + [f"{policy_headers[k]:<20}" for k in policy_keys]
    header = " | ".join(header_cols)
    divider = "-" * len(header)

    print("\n" + divider)
    print("OUT-OF-SAMPLE EVALUATION RESULTS (30 UNSEEN MARKET SEEDS)")
    print(divider)
    print(header)
    print(divider)

    for m in metrics_list:
        label = metric_labels[m]
        row_cols = [f"{label:<20}"]
        for k in policy_keys:
            val_str = f"{summary[k][m]['mean']:.4f} +/- {summary[k][m]['std']:.4f}"
            row_cols.append(f"{val_str:<20}")
        print(" | ".join(row_cols))

    print(divider + "\n")

    # 7. Sampler Robustness Check Table (DDIM vs. DDPM-Det)
    rob_header = f"{'Metric':<20} | {'IL Cond (DDIM)':<20} | {'IL Cond (DDPM-Det)':<20} | {'Abs Diff (|Delta|)':<20}"
    rob_div = "-" * len(rob_header)
    print("\n" + rob_div)
    print("SAMPLER ROBUSTNESS CHECK: SONG ET AL. DDIM VS. HO ET AL. ZERO-NOISE DDPM-DET")
    print(rob_div)
    print(rob_header)
    print(rob_div)
    for m in metrics_list:
        val_ddim = summary["il_conditioned_det"][m]["mean"]
        val_ddpm = summary["il_cond_ddpm_det"][m]["mean"]
        diff = abs(val_ddim - val_ddpm)
        print(f"{metric_labels[m]:<20} | {val_ddim:<20.4f} | {val_ddpm:<20.4f} | {diff:<20.4f}")
    print(rob_div + "\n")

    # 8. Paired Differences & 95% Confidence Intervals
    comparisons = [
        ("RL Cond vs IL Cond", "rl_conditioned_det", "il_conditioned_det"),
        ("RL Cond vs Markowitz TC", "rl_conditioned_det", "markowitz_tc"),
        ("IL Cond vs Markowitz TC", "il_conditioned_det", "markowitz_tc"),
        ("IL Cond vs Markowitz", "il_conditioned_det", "markowitz"),
    ]

    paired_summary = {}
    print("=" * 95)
    print("PAIRED STATISTICAL DIFFERENCES & 95% CONFIDENCE INTERVALS (30 UNSEEN SEEDS)")
    print("=" * 95)

    for comp_name, key_a, key_b in comparisons:
        paired_summary[comp_name] = {}
        print(f"\n--- Comparison: {comp_name} [Model A: {key_a} - Model B: {key_b}] ---")
        comp_hdr = f"{'Metric':<20} | {'Mean Diff':<12} | {'95% Conf Interval':<24} | {'t-stat':<10} | {'p-value':<10}"
        print(comp_hdr)
        print("-" * len(comp_hdr))

        for m in metrics_list:
            vals_a = [r[m] for r in results[key_a]]
            vals_b = [r[m] for r in results[key_b]]
            p_stats = compute_paired_stats(vals_a, vals_b)
            paired_summary[comp_name][m] = p_stats

            ci_str = f"[{p_stats['ci_95'][0]:+.4f}, {p_stats['ci_95'][1]:+.4f}]"
            sig_marker = "***" if p_stats['p_value'] < 0.001 else ("**" if p_stats['p_value'] < 0.01 else ("*" if p_stats['p_value'] < 0.05 else "n.s."))
            print(f"{metric_labels[m]:<20} | {p_stats['mean_diff']:<+12.4f} | {ci_str:<24} | {p_stats['t_stat']:<10.3f} | {p_stats['p_value']:<7.4f} {sig_marker}")

    print("=" * 95 + "\n")

    # 9. Save Raw and Paired Results to File
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(root_dir, "final_evaluation_results.json")
    save_data = {
        "summary": summary,
        "paired_statistics": {
            comp_name: {
                m: {
                    "mean_diff": paired_summary[comp_name][m]["mean_diff"],
                    "se_diff": paired_summary[comp_name][m]["se_diff"],
                    "ci_95_lower": paired_summary[comp_name][m]["ci_95"][0],
                    "ci_95_upper": paired_summary[comp_name][m]["ci_95"][1],
                    "t_stat": paired_summary[comp_name][m]["t_stat"],
                    "p_value": paired_summary[comp_name][m]["p_value"],
                }
                for m in metrics_list
            }
            for comp_name, _, _ in comparisons
        }
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=4)

    logger.info(f"Saved aggregated evaluation results and paired statistics to {output_path}")


if __name__ == "__main__":
    run_final_evaluation()