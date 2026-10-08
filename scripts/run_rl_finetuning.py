import os
import yaml
import torch
import pandas as pd
import numpy as np
import logging
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
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def run_rl():
    logger = setup_logger("RL_Finetuning")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    with open(os.path.join(root_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Setup Pure Gaussian Simulator
    logger.info("Initializing Pure Gaussian Simulator...")
    market_ticker = config["data"]["market_ticker"]
    stock_tickers = config["data"]["stock_tickers"]
    
    loader = MarketDataLoader(tickers=[market_ticker] + stock_tickers, 
                              start_date=config["data"]["start_date"], 
                              end_date=config["data"]["end_date"])
    data = loader.fetch_data()
    
    hmm = HybridJumpsHMM(n_states=config["hmm"]["n_states"], emission_dist="gaussian")
    hmm.fit(data[market_ticker])
    
    # TURN ON CRASHES
    hmm.epsilon = 0.0025
    hmm.lambd = 85.0
    logger.info(f"Market Crashes Enabled! Epsilon: {hmm.epsilon}, Lambda: {hmm.lambd}")
    
    sim_model = SingleIndexModel(tickers=stock_tickers, emission_dist="gaussian")
    # Wrap in pd.DataFrame to satisfy static type checkers
    sim_model.fit(pd.DataFrame(data[stock_tickers]), data[market_ticker])
    
    # 2. Load the Winner IL Model
    logger.info("Loading IL Winner Model...")
    action_dim = len(stock_tickers) + 1
    
    denoiser = MeanCovMLPDenoiser(n_assets=len(stock_tickers), action_dim=action_dim, hidden_dim=1024)
    model = PortDiff(denoiser=denoiser, num_timesteps=100, beta_schedule="linear")
    
    model_path = os.path.join(root_dir, "il_winner.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # 3. RL Configuration
    rl_cfg = config["rl"]
    buffer = AWRReplayBuffer()
    trainer = PortDiffAWRTrainer(
        model=model, 
        replay_buffer=buffer, 
        learning_rate=rl_cfg["learning_rate"], 
        batch_size=rl_cfg["batch_size"], 
        beta=rl_cfg["beta"], 
        weight_clip=rl_cfg["weight_clip"]
    )
    
    window = config["expert"]["rolling_window"]
    daily_rf = config["expert"]["annual_risk_free_rate"] / 252.0
    
    # 4. Main RL Loop
    logger.info(f"Starting AWR Reinforcement Learning for {rl_cfg['iterations']} Iterations...")
    
    for iteration in range(1, rl_cfg["iterations"] + 1):
        logger.info(f"--- Iteration {iteration} ---")
        
        iteration_clean_sharpes = []
        iteration_noisy_sharpes = []
        
        # EXPLORATION: Generate Trajectories
        for traj in tqdm(range(rl_cfg["trajectories_per_iteration"]), desc="Generating Trajectories"):
            # Set the seed for pure deterministic reproducibility
            seed = iteration * 1000 + traj
            np.random.seed(seed)
            
            # Simulate a completely new 5-year market
            sim_states = hmm.simulate_states(
                n_steps=config["hmm"]["simulation"]["n_steps"], 
                epsilon=hmm.epsilon, 
                lambd=hmm.lambd
            )
            sim_stocks = sim_model.simulate(hmm.decode_states(sim_states), random_seed=seed)
            
            # 1. State Input Data (Restricted to the 10 Risky Assets)
            state_returns_arr = sim_stocks.to_numpy()
            
            # 2. Reward Calculation Data (Must include Cash)
            sim_stocks_with_cash = sim_stocks.copy()
            sim_stocks_with_cash['Cash'] = daily_rf
            actual_returns_arr = sim_stocks_with_cash.to_numpy()
            
            X, actual_returns = [], []
            for t in range(window, len(state_returns_arr)):
                X.append(state_returns_arr[t - window : t])
                actual_returns.append(actual_returns_arr[t])
                
            states = torch.tensor(np.array(X), dtype=torch.float32).to(device)
            returns_tensor = torch.tensor(np.array(actual_returns), dtype=torch.float32).to(device)
            
            # Predict the clean actions using the frozen model
            with torch.no_grad():
                clean_actions = model.sample(states)
                
            # Inject Exploration Noise and Enforce Feasibility
            noise = torch.randn_like(clean_actions) * 0.05
            noisy_actions = map_to_feasible_portfolio(clean_actions + noise, max_weight=0.20)
            
            # ---------------------------------------------------------
            # 1-Year Chunking for Temporal Credit Assignment
            # ---------------------------------------------------------
            chunk_size = 252  # 1 Trading Year
            num_chunks = len(states) // chunk_size
            
            for c in range(num_chunks):
                start_idx = c * chunk_size
                end_idx = start_idx + chunk_size
                
                chunk_states = states[start_idx:end_idx]
                chunk_returns = returns_tensor[start_idx:end_idx]
                chunk_clean_actions = clean_actions[start_idx:end_idx]
                chunk_noisy_actions = noisy_actions[start_idx:end_idx]
                
                # 1. Calculate Clean Baseline for this specific year
                c_ret = (chunk_clean_actions * chunk_returns).sum(dim=1)
                # Note: We add 1e-9 to prevent division-by-zero if the agent holds 100% cash
                c_sharpe = (c_ret.mean() / (c_ret.std() + 1e-9)).item() * np.sqrt(252)
                iteration_clean_sharpes.append(c_sharpe)
                
                # 2. Calculate Noisy Exploration Sharpe for this specific year
                n_ret = (chunk_noisy_actions * chunk_returns).sum(dim=1)
                n_sharpe = (n_ret.mean() / (n_ret.std() + 1e-9)).item() * np.sqrt(252)
                iteration_noisy_sharpes.append(n_sharpe)
                
                # 3. Store this specific year as an independent memory
                buffer.add_trajectory(chunk_states, chunk_noisy_actions, n_sharpe)
        
        # FILTERING: Calculate Advantage
        mean_clean_sharpe = np.mean(iteration_clean_sharpes)
        mean_noisy_sharpe = np.mean(iteration_noisy_sharpes)
        max_noisy_sharpe = np.max(iteration_noisy_sharpes)
        
        logger.info(f"Baseline IL Sharpe: {mean_clean_sharpe:.4f} | RL Exploration Mean: {mean_noisy_sharpe:.4f} | Max: {max_noisy_sharpe:.4f}")
        
        # Calculate A = R - Mean(R) directly inside the buffer
        for i in range(len(buffer.advantages)):
            buffer.advantages[i] = buffer.advantages[i] - mean_noisy_sharpe
            
        # Compile and Train
        buffer.compile_buffer()
        trainer.train(epochs=rl_cfg["epochs_per_iteration"])
        
        # Clear the buffer so the next iteration starts fresh with the new, smarter brain!
        buffer = AWRReplayBuffer()
        trainer.replay_buffer = buffer
        
    # 5. Save the RL Winner
    rl_model_path = os.path.join(root_dir, "rl_winner.pth")
    torch.save(model.state_dict(), rl_model_path)
    logger.info(f"Saved RL Winner to {rl_model_path}!")

if __name__ == "__main__":
    run_rl()