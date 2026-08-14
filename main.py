from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from simulator.loader import MarketDataLoader
from simulator.logger import setup_logger
from simulator.hmm import HybridJumpsHMM
from simulator.single_index import SingleIndexModel
from experts.markowitz import MarkowitzExpert

logger = setup_logger(__name__)

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(path, "r") as file:
        return yaml.safe_load(file)

def main() -> None:
    logger.info("Initializing Multi-Asset HMM-SIM pipeline...")
    
    # 1. Load configuration
    config = load_config("config.yaml")
    data_cfg = config["data"]
    hmm_cfg = config["hmm"]
    grid_cfg = hmm_cfg["grid_search"]
    sim_cfg = hmm_cfg["simulation"]
    
    market_ticker = data_cfg["market_ticker"]
    stock_tickers = data_cfg["stock_tickers"]
    all_tickers = [market_ticker] + stock_tickers
    
    # 2. Fetch historical market benchmark and stock data
    logger.info(f"Fetching data for market index '{market_ticker}' and {len(stock_tickers)} stock tickers...")
    loader = MarketDataLoader(
        tickers=all_tickers,
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"]
    )
    
    market_data = loader.fetch_data()
    market_returns = market_data[market_ticker]
    raw_stock_returns = market_data[stock_tickers]
    
    stock_returns = raw_stock_returns.to_frame() if isinstance(raw_stock_returns, pd.Series) else raw_stock_returns
    logger.info(f"Loaded {len(market_returns)} historical daily return rows.")
    
    # 3. Fit HMM model on market benchmark
    n_states = hmm_cfg["n_states"]
    logger.info(f"Fitting HybridJumpsHMM on market benchmark ({market_ticker}) with {n_states} states...")
    hmm_model = HybridJumpsHMM(n_states=n_states)
    hmm_model.fit(market_returns)
    
    # 4. Calibrate market jump parameters via grid search
    logger.info(f"Running multi-objective grid search for {market_ticker} jump parameters...")
    best_eps, best_lambd = hmm_model.grid_search(
        returns=market_returns,
        max_lag=grid_cfg["max_lag"],
        n_paths=grid_cfg["n_paths"],
        w_K=grid_cfg["w_K"]
    )
    logger.info(f"Optimal market parameters: epsilon* = {best_eps}, lambda* = {best_lambd}")
    
    # 5. Fit Single-Index Model for all stock tickers against market benchmark
    logger.info(f"Fitting SingleIndexModel for {len(stock_tickers)} stocks against {market_ticker}...")
    sim_model = SingleIndexModel(tickers=stock_tickers)
    sim_model.fit(stock_returns, market_returns)
    logger.info(f"Estimated Stock Betas (Min: {np.min(sim_model.betas):.2f}, Mean: {np.mean(sim_model.betas):.2f}, Max: {np.max(sim_model.betas):.2f})")
    
    # 6. Simulate synthetic market path and project across all stocks
    n_steps = sim_cfg["n_steps"]
    logger.info(f"Simulating {n_steps} trading days for {len(stock_tickers)} stocks...")
    
    sim_market_states = hmm_model.simulate_states(
        n_steps=n_steps,
        epsilon=best_eps,
        lambd=best_lambd
    )
    sim_market_returns = hmm_model.decode_states(sim_market_states)
    
    # Project market returns to multi-asset return DataFrame
    sim_stock_returns_df = sim_model.simulate(market_sim_returns=sim_market_returns, random_seed=42)
    
    # 7. Log summary statistics
    hist_avg_vol = np.mean(np.std(stock_returns.to_numpy(), axis=0))
    sim_avg_vol = np.mean(np.std(sim_stock_returns_df.to_numpy(), axis=0))
    
    logger.info("--- MULTI-ASSET STATISTICAL SUMMARY ---")
    logger.info(f"Synthetic Return Matrix Shape : {sim_stock_returns_df.shape} (Days x Stocks)")
    logger.info(f"Average Stock Volatility      | Hist: {hist_avg_vol:.6f} | Sim: {sim_avg_vol:.6f}")

    # 8. Generate target weights using Markowitz expert
    expert_cfg = config["expert"]
    logger.info("Running rolling Markowitz expert to generate portfolio labels...")
    
    expert = MarkowitzExpert(
        risk_aversion=expert_cfg["risk_aversion"],
        rolling_window=expert_cfg["rolling_window"],
        max_weight=expert_cfg["max_weight"],
        annual_risk_free_rate=expert_cfg["annual_risk_free_rate"]
    )
    
    target_weights = expert.generate_labels(sim_stock_returns_df)
    
    logger.info(f"Generated Markowitz labels matrix shape: {target_weights.shape} (Days x Stocks)")
    # Sanity-check the very first day of weights
    first_day_weights = target_weights.iloc[0]
    logger.info("--- Sanity Check (First Day Weights) ---")
    logger.info(f"Sum of weights : {first_day_weights.sum():.4f} (Should be 1.0)")
    logger.info(f"Max weight     : {first_day_weights.max():.4f} (Should be <= {expert_cfg['max_weight']})")
    logger.info(f"Min weight     : {first_day_weights.min():.4f} (Should be >= 0.0)")

    logger.info("Multi-Asset pipeline execution complete!")

if __name__ == "__main__":
    main()