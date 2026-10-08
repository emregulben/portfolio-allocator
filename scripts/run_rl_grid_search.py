import os
import yaml
import torch
import pandas as pd
import numpy as np
import logging
import itertools
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.loader import MarketDataLoader
from simulator.hmm import HybridJumpsHMM
from simulator.single_index import SingleIndexModel
from agents.diffusion import MLPDenoiser, MeanCovMLPDenoiser, PortDiff, map_to_feasible_portfolio
from agents.replay_buffer import AWRReplayBuffer
from agents.awr_trainer import PortDiffAWRTrainer

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def run_rl_grid_search():
    logger = setup_logger("RL_GridSearch")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    with open(os.path.join(root_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Setup Environment
    logger.info("Initializing Simulator B (Stress-Augmented)...")
    market_ticker = config["data"]["market_ticker"]
    stock_tickers = config["data"]["stock_tickers"]
    
    loader = MarketDataLoader(tickers=[market_ticker] + stock_tickers, 
                              start_date=config["data"]["start_date"], 
                              end_date=config["data"]["end_date"])
    data = loader.fetch_data()
    
    hmm = HybridJumpsHMM(n_states=config["hmm"]["n_states"], emission_dist="gaussian")
    hmm.fit(data[market_ticker])
    # The Canonical Stress Environment
    hmm.epsilon = 0.001
    hmm.lambd = 85.0
    
    sim_model = SingleIndexModel(tickers=stock_tickers, emission_dist="gaussian")
    sim_model.fit(pd.DataFrame(data[stock_tickers]), data[market_ticker])
    
    action_dim = len(stock_tickers) + 1
    window = config["expert"]["rolling_window"]
    daily_rf = config["expert"]["annual_risk_free_rate"] / 252.0
    
    # 2. Grid Search Parameters
    betas = [0.05, 0.2, 0.5]
    learning_rates = [1e-4, 5e-4]
    rl_cfg = config["rl"]
    
    results = []
    
    logger.info("Starting RL Grid Search...")
    
    for beta, lr in itertools.product(betas, learning_rates):
        logger.info(f"\n{'='*50}\nTesting Combo -> Beta: {beta}, LR: {lr}\n{'='*50}")
        
        # Fresh Model Load for every grid point!
        denoiser = MeanCovMLPDenoiser(n_assets=len(stock_tickers), action_dim=action_dim, hidden_dim=1024)
        model = PortDiff(denoiser=denoiser, num_timesteps=100, beta_schedule="linear")
        model.load_state_dict(torch.load(os.path.join(root_dir, "il_winner.pth"), map_location=device))
        model.to(device)
        
        buffer = AWRReplayBuffer()
        trainer = PortDiffAWRTrainer(
            model=model, replay_buffer=buffer, 
            learning_rate=lr, batch_size=rl_cfg["batch_size"], 
            beta=beta, weight_clip=rl_cfg["weight_clip"]
        )
        
        start_sharpe = 0.0
        end_sharpe = 0.0
        
        for iteration in range(1, rl_cfg["iterations"] + 1):
            iteration_clean_sharpes = []
            iteration_noisy_sharpes = []
            
            # EXPLORATION
            for traj in tqdm(range(rl_cfg["trajectories_per_iteration"]), desc=f"Iter {iteration} Trajectories"):
                seed = iteration * 1000 + traj
                np.random.seed(seed)
                
                sim_states = hmm.simulate_states(n_steps=config["hmm"]["simulation"]["n_steps"], epsilon=hmm.epsilon, lambd=hmm.lambd)
                sim_stocks = sim_model.simulate(hmm.decode_states(sim_states), random_seed=seed)
                
                state_returns_arr = sim_stocks.to_numpy()
                sim_stocks_with_cash = sim_stocks.copy()
                sim_stocks_with_cash['Cash'] = daily_rf
                actual_returns_arr = sim_stocks_with_cash.to_numpy()
                
                X, actual_returns = [], []
                for t in range(window, len(state_returns_arr)):
                    X.append(state_returns_arr[t - window : t])
                    actual_returns.append(actual_returns_arr[t])
                    
                states = torch.tensor(np.array(X), dtype=torch.float32).to(device)
                returns_tensor = torch.tensor(np.array(actual_returns), dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    clean_actions = model.sample(states)
                    
                clean_portfolio_returns = (clean_actions * returns_tensor).sum(dim=1)
                clean_sharpe = (clean_portfolio_returns.mean() / clean_portfolio_returns.std()).item() * np.sqrt(252)
                iteration_clean_sharpes.append(clean_sharpe)
                
                noise = torch.randn_like(clean_actions) * 0.05
                noisy_actions = map_to_feasible_portfolio(clean_actions + noise, max_weight=0.20)
                
                portfolio_returns = (noisy_actions * returns_tensor).sum(dim=1)
                sharpe = (portfolio_returns.mean() / portfolio_returns.std()).item() * np.sqrt(252)
                
                buffer.add_trajectory(states, noisy_actions, sharpe)
                iteration_noisy_sharpes.append(sharpe)
                
            mean_clean_sharpe = np.mean(iteration_clean_sharpes)
            mean_noisy_sharpe = np.mean(iteration_noisy_sharpes)
            
            logger.info(f"Iter {iteration} | Clean Sharpe: {mean_clean_sharpe:.4f} | Noisy Mean: {mean_noisy_sharpe:.4f}")
            
            # Log Start and End Sharpe for this Combo
            if iteration == 1:
                start_sharpe = mean_clean_sharpe
            if iteration == rl_cfg["iterations"]:
                end_sharpe = mean_clean_sharpe
                
            # FILTERING & TRAINING
            for i in range(len(buffer.advantages)):
                buffer.advantages[i] = buffer.advantages[i] - mean_noisy_sharpe
                
            buffer.compile_buffer()
            trainer.train(epochs=rl_cfg["epochs_per_iteration"])
            
            buffer = AWRReplayBuffer()
            trainer.replay_buffer = buffer
            
        improvement = end_sharpe - start_sharpe
        logger.info(f"Combo Finished! Improvement: {improvement:.4f} ({start_sharpe:.4f} -> {end_sharpe:.4f})")
        
        results.append({
            "Beta": beta,
            "LR": lr,
            "Start_Sharpe": start_sharpe,
            "End_Sharpe": end_sharpe,
            "Improvement": improvement
        })
        
    logger.info("\n" + "="*50 + "\nGRID SEARCH RESULTS\n" + "="*50)
    results_df = pd.DataFrame(results).sort_values(by="Improvement", ascending=False)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    run_rl_grid_search()