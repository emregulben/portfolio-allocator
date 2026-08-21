import pandas as pd
import numpy as np
import yaml
from simulator.loader import MarketDataLoader
from simulator.hmm import HybridJumpsHMM
from simulator.validation import MetricsEvaluator
from simulator.logger import setup_logger

logger = setup_logger(__name__)

def run_ablation(hist_returns: np.ndarray, config: dict, label: str):
    """Fits HMM, calibrates jump parameters, and runs a 1,000-path ablation study."""
    logger.info(f"[{label}] Fitting HMM...")
    hmm = HybridJumpsHMM(n_states=config["hmm"]["n_states"])
    hmm.fit(pd.Series(hist_returns))
    
    logger.info(f"[{label}] Running grid search...")
    best_eps, best_lambd = hmm.grid_search(
        pd.Series(hist_returns),
        max_lag=config["hmm"]["grid_search"]["max_lag"],
        n_paths=config["hmm"]["grid_search"]["n_paths"],
        w_K=config["hmm"]["grid_search"]["w_K"]
    )
    logger.info(f"[{label}] Best epsilon={best_eps}, lambda={best_lambd}")
    
    n_steps = config["hmm"]["simulation"]["n_steps"]
    n_paths = 1000
    results_jump = []
    results_no_jump = []
    
    for i in range(n_paths):
        if i > 0 and i % 200 == 0:
            logger.info(f"[{label}] Running path {i} / {n_paths}...")
            
        states_jump = hmm.simulate_states(n_steps, epsilon=best_eps, lambd=best_lambd)
        sim_jump = hmm.decode_states(states_jump)
        results_jump.append(MetricsEvaluator.evaluate_all(hist_returns, sim_jump))
        
        states_no_jump = hmm.simulate_states(n_steps, epsilon=0.0, lambd=best_lambd)
        sim_no_jump = hmm.decode_states(states_no_jump)
        results_no_jump.append(MetricsEvaluator.evaluate_all(hist_returns, sim_no_jump))
        
    return pd.DataFrame(results_jump), pd.DataFrame(results_no_jump)

def run_validation(mode: str = "both"):
    """
    Runs the Monte Carlo ablation study comparing HMM with and without jumps.
    
    Args:
        mode: Which return transformation to validate.
              "arithmetic" - daily arithmetic returns only
              "log"        - daily log returns only
              "both"       - side-by-side controlled comparison
    """
    valid_modes = ("arithmetic", "log", "both")
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")
    
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    market_ticker = config["data"]["market_ticker"]
    loader = MarketDataLoader(
        tickers=[market_ticker],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"]
    )
    arith_returns = loader.fetch_data()[market_ticker].dropna().to_numpy()
    log_returns = np.log(1 + arith_returns)
    
    if mode == "both":
        df_jump_a, df_no_jump_a = run_ablation(arith_returns, config, "Arithmetic")
        df_jump_l, df_no_jump_l = run_ablation(log_returns, config, "Log")
        
        logger.info(f"\n=== CONTROLLED COMPARISON: ARITHMETIC vs LOG RETURNS (1,000 Paths) ===")
        logger.info(f"{'Metric':<15} | {'Arith (No Jump)':<22} | {'Arith (With Jump)':<22} | {'Log (No Jump)':<22} | {'Log (With Jump)':<22}")
        logger.info("-" * 115)
        for col in df_jump_a.columns:
            logger.info(
                f"{col:<15} "
                f"| {df_no_jump_a[col].mean():7.4f} ± {df_no_jump_a[col].std():.4f}    "
                f"| {df_jump_a[col].mean():7.4f} ± {df_jump_a[col].std():.4f}    "
                f"| {df_no_jump_l[col].mean():7.4f} ± {df_no_jump_l[col].std():.4f}    "
                f"| {df_jump_l[col].mean():7.4f} ± {df_jump_l[col].std():.4f}"
            )
    else:
        returns = arith_returns if mode == "arithmetic" else log_returns
        label = mode.capitalize()
        
        df_jump, df_no_jump = run_ablation(returns, config, label)
        
        logger.info(f"\n=== ABLATION STUDY: {label.upper()} RETURNS (1,000 Paths) ===")
        logger.info(f"{'Metric':<15} | {'No Jumps':<22} | {'With Jumps':<22}")
        logger.info("-" * 65)
        for col in df_jump.columns:
            logger.info(
                f"{col:<15} "
                f"| {df_no_jump[col].mean():7.4f} ± {df_no_jump[col].std():.4f}    "
                f"| {df_jump[col].mean():7.4f} ± {df_jump[col].std():.4f}"
            )

if __name__ == "__main__":
    run_validation(mode="both")