from pathlib import Path
import numpy as np
import yaml

from simulator.loader import MarketDataLoader
from simulator.logger import setup_logger
from simulator.hmm import HybridJumpsHMM

logger = setup_logger(__name__)

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(path, "r") as file:
        return yaml.safe_load(file)

def main() -> None:
    logger.info("Initializing HMM-WJ pipeline...")
    
    # 1. Load configuration directly
    config = load_config("config.yaml")
    data_cfg = config["data"]
    hmm_cfg = config["hmm"]
    grid_cfg = hmm_cfg["grid_search"]
    sim_cfg = hmm_cfg["simulation"]
    
    # 2. Fetch historical data
    tickers = data_cfg["tickers"]
    logger.info(f"Loading market data for: {tickers}")
    
    loader = MarketDataLoader(
        tickers=tickers,
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"]
    )
    
    market_data = loader.fetch_data()
    ticker = tickers[0]
    returns = market_data[ticker].dropna()
    logger.info(f"Loaded {len(returns)} historical returns for {ticker}.")
    
    # 3. Fit HMM model
    n_states = hmm_cfg["n_states"]
    logger.info(f"Fitting HybridJumpsHMM with {n_states} states...")
    model = HybridJumpsHMM(n_states=n_states)
    model.fit(returns)
    
    # 4. Calibrate jump parameters via grid search
    logger.info("Running multi-objective grid search for jump parameters...")
    best_eps, best_lambd = model.grid_search(
        returns=returns,
        max_lag=grid_cfg["max_lag"],
        n_paths=grid_cfg["n_paths"],
        w_K=grid_cfg["w_K"]
    )
    logger.info(f"Optimal parameters found: epsilon* = {best_eps}, lambda* = {best_lambd}")
    
    # 5. Simulate synthetic returns
    n_steps = sim_cfg["n_steps"]
    logger.info(f"Simulating {n_steps} synthetic trading days...")
    
    sim_states = model.simulate_states(
        n_steps=n_steps,
        epsilon=best_eps,
        lambd=best_lambd
    )
    sim_returns = model.decode_states(sim_states)
    
    # 6. Calculate comparative summary statistics
    ret_data = returns.to_numpy()
    hist_mean, sim_mean = np.mean(ret_data), np.mean(sim_returns)
    hist_std, sim_std = np.std(ret_data), np.std(sim_returns)
    
    hist_var, sim_var = np.var(ret_data), np.var(sim_returns)
    hist_kurt = (np.mean((ret_data - hist_mean)**4) / (hist_var**2) - 3.0) if hist_var > 0 else 0.0
    sim_kurt = (np.mean((sim_returns - sim_mean)**4) / (sim_var**2) - 3.0) if sim_var > 0 else 0.0
    
    logger.info("--- COMPARATIVE STATISTICAL SUMMARY ---")
    logger.info(f"Mean Return    | Hist: {hist_mean:.6f} | Sim: {sim_mean:.6f}")
    logger.info(f"Volatility (Std)| Hist: {hist_std:.6f} | Sim: {sim_std:.6f}")
    logger.info(f"Kurtosis (Tails)| Hist: {hist_kurt:.4f}   | Sim: {sim_kurt:.4f}")
    logger.info("Pipeline execution complete!")

if __name__ == "__main__":
    main()