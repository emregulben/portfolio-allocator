import os
import sys
import yaml
import torch
import logging
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

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
    Computes institutional risk, return, and execution metrics from daily allocations.

    Args:
        weights_arr (np.ndarray): Daily portfolio allocations of shape (T, action_dim).
        returns_arr (np.ndarray): Realized asset returns of shape (T, action_dim).
        cost_c (float): Proportional transaction cost parameter per unit turnover.

    Returns:
        dict: Dictionary containing gross/net returns, Sharpe, Sortino, MDD, and turnover.
    """
    T = len(weights_arr)
    
    # 1. Daily Gross Returns: w_t^T r_{t+1}
    gross_returns = np.sum(weights_arr * returns_arr, axis=1)
    
    # 2. Daily Turnover and Net Returns
    turnover = np.zeros(T)
    if T > 1:
        turnover[1:] = np.sum(np.abs(weights_arr[1:] - weights_arr[:-1]), axis=1)
        
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
    max_drawdown = np.min(drawdowns)  # negative percentage
    
    # 7. Average Daily Turnover
    mean_turnover = np.mean(turnover[1:]) if T > 1 else 0.0
    
    return {
        "net_return": float(ann_net_return),
        "gross_sharpe": float(gross_sharpe),
        "net_sharpe": float(net_sharpe),
        "net_sortino": float(net_sortino),
        "max_drawdown": float(max_drawdown),
        "mean_turnover": float(mean_turnover)
    }
    
def load_evaluation_environment():
    """
    Initializes market data, HMM simulator, classical expert, and neural policies.

    Returns:
        tuple: (config, hmm, sim_model, expert, il_model, rl_model, device)
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
    
    # 2. Classical Markowitz Benchmark Setup
    expert = MarkowitzExpert(
        risk_aversion=config["expert"]["risk_aversion"],
        rolling_window=config["expert"]["rolling_window"],
        max_weight=config["expert"]["max_weight"],
        annual_risk_free_rate=config["expert"]["annual_risk_free_rate"]
    )
    
    # 3. Neural Network Architectures Setup
    action_dim = len(stock_tickers) + 1
    
    # Frozen Imitation Learning Baseline
    il_denoiser = MeanCovMLPDenoiser(n_assets=len(stock_tickers), action_dim=action_dim, hidden_dim=1024)
    il_model = PortDiff(denoiser=il_denoiser, num_timesteps=100, beta_schedule="linear")
    il_path = os.path.join(root_dir, "il_winner.pth")
    il_model.load_state_dict(torch.load(il_path, map_location=device))
    il_model.to(device).eval()
    logger.info(f"Loaded frozen IL baseline model from {il_path}")
    
    # Fine-Tuned Reinforcement Learning Policy
    rl_denoiser = MeanCovMLPDenoiser(n_assets=len(stock_tickers), action_dim=action_dim, hidden_dim=1024)
    rl_model = PortDiff(denoiser=rl_denoiser, num_timesteps=100, beta_schedule="linear")
    rl_path = os.path.join(root_dir, "rl_winner.pth")
    rl_model.load_state_dict(torch.load(rl_path, map_location=device))
    rl_model.to(device).eval()
    logger.info(f"Loaded fine-tuned RL policy from {rl_path}")
    
    return config, hmm, sim_model, expert, il_model, rl_model, device

def run_final_evaluation(num_seeds=30, start_seed=9000):
    """
    Executes an out-of-sample statistical comparison of Markowitz, IL, and RL policies.

    Args:
        num_seeds (int): Number of independent market seeds to evaluate.
        start_seed (int): Base random seed ensuring unseen evaluation paths.
    """
    config, hmm, sim_model, expert, il_model, rl_model, device = load_evaluation_environment()
    logger = setup_logger("Final_Evaluation")
    
    cost_c = config["expert"]["transaction_cost"]
    window = config["expert"]["rolling_window"]
    daily_rf = config["expert"]["annual_risk_free_rate"] / 252.0
    stock_tickers = config["data"]["stock_tickers"]
    
    policy_keys = [
        "markowitz",
        "il_stochastic",
        "il_deterministic",
        "rl_stochastic",
        "rl_deterministic",
    ]
    policy_headers = {
        "markowitz": "Markowitz",
        "il_stochastic": "IL (Stochastic)",
        "il_deterministic": "IL (Deterministic)",
        "rl_stochastic": "RL (Stochastic)",
        "rl_deterministic": "RL (Deterministic)",
    }

    results = {k: [] for k in policy_keys}

    logger.info(f"Starting out-of-sample evaluation across {num_seeds} unseen seeds (seeds {start_seed} to {start_seed + num_seeds - 1})...")

    for i in tqdm(range(num_seeds), desc="Evaluating Seeds"):
        seed = start_seed + i
        np.random.seed(seed)

        # 1. Simulate Unseen 10-Year Market Path
        sim_states = hmm.simulate_states(
            n_steps=config["hmm"]["simulation"]["n_steps"],
            epsilon=hmm.epsilon,
            lambd=hmm.lambd
        )
        sim_stocks = sim_model.simulate(hmm.decode_states(sim_states), random_seed=seed)

        sim_stocks_with_cash = sim_stocks.copy()
        sim_stocks_with_cash['Cash'] = daily_rf
        actual_returns_arr = sim_stocks_with_cash.to_numpy()[window:]

        # Prepare historical input tensor for diffusion models
        state_returns_arr = sim_stocks.to_numpy()
        X = [state_returns_arr[t - window : t] for t in range(window, len(state_returns_arr))]
        states_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(device)

        # 2. Evaluate Classical Markowitz Expert
        markowitz_df = expert.generate_labels(sim_stocks)
        w_markowitz = markowitz_df.to_numpy()
        results["markowitz"].append(compute_portfolio_metrics(w_markowitz, actual_returns_arr, cost_c=cost_c))

        # 3. Evaluate Imitation Learning Baselines (Stochastic vs Deterministic)
        with torch.no_grad():
            w_il_stoch = il_model.sample(states_tensor, deterministic=False).cpu().numpy()
            w_il_det = il_model.sample(states_tensor, deterministic=True).cpu().numpy()
        results["il_stochastic"].append(compute_portfolio_metrics(w_il_stoch, actual_returns_arr, cost_c=cost_c))
        results["il_deterministic"].append(compute_portfolio_metrics(w_il_det, actual_returns_arr, cost_c=cost_c))

        # 4. Evaluate Reinforcement Learning Policies (Stochastic vs Deterministic)
        with torch.no_grad():
            w_rl_stoch = rl_model.sample(states_tensor, deterministic=False).cpu().numpy()
            w_rl_det = rl_model.sample(states_tensor, deterministic=True).cpu().numpy()
        results["rl_stochastic"].append(compute_portfolio_metrics(w_rl_stoch, actual_returns_arr, cost_c=cost_c))
        results["rl_deterministic"].append(compute_portfolio_metrics(w_rl_det, actual_returns_arr, cost_c=cost_c))

    # 5. Aggregate Statistical Results (Mean +/- Std)
    metrics_list = ["net_sharpe", "gross_sharpe", "net_sortino", "net_return", "max_drawdown", "mean_turnover"]
    metric_labels = {
        "net_sharpe": "Net Sharpe Ratio",
        "gross_sharpe": "Gross Sharpe Ratio",
        "net_sortino": "Net Sortino Ratio",
        "net_return": "Annual Net Return",
        "max_drawdown": "Maximum Drawdown",
        "mean_turnover": "Daily Turnover"
    }

    summary = {}
    for model_key in policy_keys:
        summary[model_key] = {}
        for m in metrics_list:
            vals = [r[m] for r in results[model_key]]
            summary[model_key][m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals))
            }

    # 6. Print Formatted Results Table
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

    # 7. Save Raw Results to File
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(root_dir, "final_evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    logger.info(f"Saved aggregated evaluation results to {output_path}")


if __name__ == "__main__":
    run_final_evaluation()