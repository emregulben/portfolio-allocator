import pandas as pd
import numpy as np
import yaml
from simulator.loader import MarketDataLoader
from simulator.hmm import HybridJumpsHMM
from simulator.validation import MetricsEvaluator
from simulator.logger import setup_logger

logger = setup_logger(__name__)

def run_validation():
    # 1. Load configuration and fetch historical market benchmark
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    market_ticker = config["data"]["market_ticker"]
    logger.info(f"Fetching historical data for {market_ticker}...")
    
    loader = MarketDataLoader(
        tickers=[market_ticker],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"]
    )
    hist_returns = loader.fetch_data()[market_ticker].dropna().to_numpy()
    
    # 2. Fit the HMM and find optimal parameters
    logger.info(f"Fitting HMM-WJ on {market_ticker}...")
    hmm = HybridJumpsHMM(n_states=config["hmm"]["n_states"])
    hmm.fit(pd.Series(hist_returns))
    
    logger.info("Running grid search to calibrate jump parameters...")
    best_eps, best_lambd = hmm.grid_search(
        pd.Series(hist_returns), 
        max_lag=config["hmm"]["grid_search"]["max_lag"],
        n_paths=config["hmm"]["grid_search"]["n_paths"],
        w_K=config["hmm"]["grid_search"]["w_K"]
    )
    
    # 3. Setup Monte Carlo parameters
    n_steps = config["hmm"]["simulation"]["n_steps"]
    n_paths = 1000
    
    logger.info(f"Starting {n_paths}-path Monte Carlo Ablation Study...")
    results_jump = []
    results_no_jump = []
    
    for i in range(n_paths):
        if i > 0 and i % 100 == 0:
            logger.info(f"Running path {i} / {n_paths}...")
            
        # A. Simulate WITH Jumps (HMM-WJ)
        states_jump = hmm.simulate_states(n_steps, epsilon=best_eps, lambd=best_lambd)
        sim_jump = hmm.decode_states(states_jump)
        results_jump.append(MetricsEvaluator.evaluate_all(hist_returns, sim_jump))
        
        # B. Simulate WITHOUT Jumps (Standard HMM) by forcing epsilon = 0.0
        states_no_jump = hmm.simulate_states(n_steps, epsilon=0.0, lambd=best_lambd)
        sim_no_jump = hmm.decode_states(states_no_jump)
        results_no_jump.append(MetricsEvaluator.evaluate_all(hist_returns, sim_no_jump))
        
    # 4. Aggregate and print the final evaluation table
    df_jump = pd.DataFrame(results_jump)
    df_no_jump = pd.DataFrame(results_no_jump)
    
    logger.info(f"\n=== ABLATION STUDY RESULTS ({n_paths:,} Paths) ===")
    logger.info(f"{'Metric':<15} | {'HMM-WJ (With Jumps)':<25} | {'HMM (No Jumps)':<25}")
    logger.info("-" * 72)
    
    for col in df_jump.columns:
        mean_jump = df_jump[col].mean()
        std_jump = df_jump[col].std()
        
        mean_no = df_no_jump[col].mean()
        std_no = df_no_jump[col].std()
        
        logger.info(f"{col:<15} | {mean_jump:7.4f} ± {std_jump:.4f}       | {mean_no:7.4f} ± {std_no:.4f}")

if __name__ == "__main__":
    run_validation()