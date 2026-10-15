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
from agents.replay_buffer import AWRReplayBuffer, ConditionedAWRReplayBuffer
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

def compute_daily_utility(actions, returns, states, gamma=2.0, cost_c=0.0005, prev_weights=None):
    """
    Computes localized daily mean-variance utility with drift-adjusted transaction costs:
        r_tilde_t = w_t^T r_{t+1} - (gamma / 2) * w_t^T Sigma_t w_t - c * ||w_t - w_{t-1}^+||_1

    Args:
        actions (torch.Tensor): Target portfolio weights of shape (T, action_dim).
        returns (torch.Tensor): Realized returns of shape (T, action_dim).
        states (torch.Tensor): Rolling return windows of shape (T, window_size, n_assets).
        gamma (float): Risk aversion coefficient.
        cost_c (float): Proportional transaction cost parameter.
        prev_weights (torch.Tensor, optional): Incumbent portfolio weights prior to rebalancing,
            of shape (T, action_dim). If provided, turnover at each step t is ||actions[t] - prev_weights[t]||_1.
            If None, evaluates drift recursively from consecutive actions.

    Returns:
        torch.Tensor: Daily utility values of shape (T,).
    """
    # 1. Daily Portfolio Return: w_t^T r_{t+1}
    portfolio_return = (actions * returns).sum(dim=1)

    # 2. Rolling Covariance Matrix: Sigma_t from states available at decision time
    mu = states.mean(dim=1, keepdim=True)
    centered = states - mu
    cov = torch.bmm(centered.transpose(1, 2), centered) / (states.shape[1] - 1)

    # Risky asset weights (excluding cash)
    w_risky = actions[:, :-1].unsqueeze(1)
    portfolio_variance = torch.bmm(torch.bmm(w_risky, cov), w_risky.transpose(1, 2)).view(-1)

    # 3. Transaction Costs Accounting for Portfolio Drift
    if prev_weights is not None:
        turnover = torch.norm(actions - prev_weights, p=1, dim=1)
    else:
        turnover = torch.zeros(len(actions), device=actions.device)
        if len(actions) > 1:
            w_prev = actions[:-1]
            r_prev = returns[:-1]
            gross_ret_prev = portfolio_return[:-1].unsqueeze(1)

            # Drifted weights at the end of period t-1 prior to rebalancing
            denom = torch.clamp(1.0 + gross_ret_prev, min=1e-8)
            w_prev_drifted = (w_prev * (1.0 + r_prev)) / denom

            # Turnover required to move from drifted weights to new target weights
            turnover[1:] = torch.norm(actions[1:] - w_prev_drifted, p=1, dim=1)

    return portfolio_return - (gamma / 2.0) * portfolio_variance - cost_c * turnover

def run_rl_unconditioned():
    logger = setup_logger("RL_Finetuning_Unconditioned")
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
    gamma = config["expert"]["risk_aversion"]
    cost_c = config["expert"]["transaction_cost"]
    
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
            
            # Simulate a completely new 10-year market
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
                
                # 3. Localized Daily Utility & Paired Advantage
                c_utility = compute_daily_utility(chunk_clean_actions, chunk_returns, chunk_states, gamma=gamma, cost_c=cost_c)
                n_utility = compute_daily_utility(chunk_noisy_actions, chunk_returns, chunk_states, gamma=gamma, cost_c=cost_c)
                daily_advantage = n_utility - c_utility
                
                buffer.add_trajectory(chunk_states, chunk_noisy_actions, daily_advantage)
        
        # Financial Evaluation Metrics
        mean_clean_sharpe = np.mean(iteration_clean_sharpes)
        mean_noisy_sharpe = np.mean(iteration_noisy_sharpes)
        max_noisy_sharpe = np.max(iteration_noisy_sharpes)
        mean_paired_sharpe = mean_noisy_sharpe - mean_clean_sharpe
        
        logger.info(
            f"Baseline IL Sharpe: {mean_clean_sharpe:.4f} | RL Exploration Mean: {mean_noisy_sharpe:.4f} | "
            f"Max: {max_noisy_sharpe:.4f} | Paired Sharpe Delta: {mean_paired_sharpe:.4f}"
        )
        
        # Advantage Normalization
        all_adv = torch.cat(buffer.advantages, dim=0)
        adv_mean = all_adv.mean()
        adv_std = all_adv.std() + 1e-8
        for i in range(len(buffer.advantages)):
            buffer.advantages[i] = (buffer.advantages[i] - adv_mean) / adv_std
            
        # ---------------------------------------------------------
        # Weight Diagnostics & Effective Sample Size
        # ---------------------------------------------------------
        # Convert list of tensors into one massive tensor for fast PyTorch math
        adv_tensor = torch.cat(buffer.advantages, dim=0)
        
        # Calculate theoretical weights before clipping
        raw_weights = torch.exp(adv_tensor / rl_cfg["beta"])
        
        # Calculate fraction of weights exceeding the maximum cap
        clip_max = rl_cfg["weight_clip"]
        clip_rate = (raw_weights >= clip_max).float().mean().item()
        
        # Apply the clip to calculate actual weights used by the neural net
        clipped_weights = torch.clamp(raw_weights, max=clip_max)
        
        # Effective Sample Size (N_eff) calculation
        n_eff = (clipped_weights.sum() ** 2) / (clipped_weights ** 2).sum()
        
        logger.info(f"Diagnostics | Clipping Rate: {clip_rate*100:.2f}% | N_eff: {n_eff.item():.2f} / {len(raw_weights)}")
        
        # Calculate quantiles (25th, 50th/Median, 75th, 90th, 99th)
        adv_q = torch.quantile(adv_tensor, torch.tensor([0.25, 0.5, 0.75, 0.90, 0.99]))
        w_q = torch.quantile(clipped_weights, torch.tensor([0.25, 0.5, 0.75, 0.90, 0.99]))
        
        logger.info(f"Advantage Quantiles [25, 50, 75, 90, 99]: {[round(x, 4) for x in adv_q.tolist()]}")
        logger.info(f"Weight Quantiles    [25, 50, 75, 90, 99]: {[round(x, 4) for x in w_q.tolist()]}")
        # ---------------------------------------------------------

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

def run_rl_conditioned():
    logger = setup_logger("RL_Conditioned_Finetuning")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    with open(os.path.join(root_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Setup Market Simulator
    logger.info("Initializing Pure Gaussian Simulator...")
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
    
    # Enable crash regimes
    hmm.epsilon = 0.0025
    hmm.lambd = 85.0
    logger.info(f"Market Crashes Enabled! Epsilon: {hmm.epsilon}, Lambda: {hmm.lambd}")
    
    sim_model = SingleIndexModel(tickers=stock_tickers, emission_dist="gaussian")
    sim_model.fit(pd.DataFrame(data[stock_tickers]), data[market_ticker])
    
    # 2. Load the Conditioned IL Checkpoint
    logger.info("Loading Conditioned IL Model...")
    action_dim = len(stock_tickers) + 1
    
    denoiser = MeanCovMLPDenoiser(
        n_assets=len(stock_tickers), 
        action_dim=action_dim, 
        hidden_dim=1024,
        condition_on_prev_weights=True
    )
    model = PortDiff(denoiser=denoiser, num_timesteps=100, beta_schedule="linear")
    
    model_path = os.path.join(root_dir, "il_conditioned.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # 3. RL Configuration
    rl_cfg = config["rl"]
    buffer = ConditionedAWRReplayBuffer()
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
    gamma = config["expert"]["risk_aversion"]
    cost_c = config["expert"]["transaction_cost"]
    
    total_trajs = rl_cfg["trajectories_per_iteration"]
    batch_trajs_size = 20  # Parallelize rollout across market trajectories on MPS/GPU
    chunk_size = 252       # 1 Trading Year
    
    # 4. Main Conditioned AWR Fine-Tuning Loop
    logger.info(f"Starting Conditioned AWR Reinforcement Learning for {rl_cfg['iterations']} Iterations...")
    
    for iteration in range(1, rl_cfg["iterations"] + 1):
        logger.info(f"--- Iteration {iteration} ---")
        
        iteration_clean_sharpes = []
        iteration_noisy_sharpes = []
        
        # Parallelized Trajectory Rollouts
        pbar = tqdm(total=total_trajs, desc="Generating Trajectories (Batched)")
        for b_start in range(0, total_trajs, batch_trajs_size):
            b_end = min(b_start + batch_trajs_size, total_trajs)
            curr_batch_trajs = b_end - b_start
            
            # Pre-simulate market data for this batch
            batch_states_list = []
            batch_returns_list = []
            for traj in range(b_start, b_end):
                seed = iteration * 1000 + traj
                np.random.seed(seed)
                
                sim_states = hmm.simulate_states(
                    n_steps=config["hmm"]["simulation"]["n_steps"], 
                    epsilon=hmm.epsilon, 
                    lambd=hmm.lambd
                )
                sim_stocks = sim_model.simulate(hmm.decode_states(sim_states), random_seed=seed)
                
                state_returns_arr = sim_stocks.to_numpy()
                sim_stocks_with_cash = sim_stocks.copy()
                sim_stocks_with_cash['Cash'] = daily_rf
                actual_returns_arr = sim_stocks_with_cash.to_numpy()
                
                X, actual_returns = [], []
                for t in range(window, len(state_returns_arr)):
                    X.append(state_returns_arr[t - window : t])
                    actual_returns.append(actual_returns_arr[t])
                    
                batch_states_list.append(torch.tensor(np.array(X), dtype=torch.float32))
                batch_returns_list.append(torch.tensor(np.array(actual_returns), dtype=torch.float32))
            
            # Stack batch: (B, T, window, n_assets) and (B, T, action_dim)
            batch_states = torch.stack(batch_states_list, dim=0).to(device)
            batch_returns = torch.stack(batch_returns_list, dim=0).to(device)
            T_steps = batch_states.shape[1]
            
            # Closed-Loop Autoregressive Rollout
            # Initial portfolio at t=0: 100% Cash
            prev_w = torch.zeros((curr_batch_trajs, action_dim), device=device)
            prev_w[:, -1] = 1.0
            
            all_clean_actions = []
            all_noisy_actions = []
            all_prev_weights = []
            
            model.eval()
            for t in range(T_steps):
                st_t = batch_states[:, t]     # (B, window, n_assets)
                pw_t = prev_w.clone()          # (B, action_dim)
                all_prev_weights.append(pw_t)
                
                with torch.no_grad():
                    # Deterministic clean policy baseline
                    clean_act_t = model.sample(st_t, prev_weights=pw_t, deterministic=True)
                    all_clean_actions.append(clean_act_t)
                    
                    # Inject exploration noise and map to feasible simplex
                    noise = torch.randn_like(clean_act_t) * 0.05
                    noisy_act_t = map_to_feasible_portfolio(clean_act_t + noise, max_weight=0.20)
                    all_noisy_actions.append(noisy_act_t)
                    
                    # Execute noisy action in the environment and account for portfolio drift
                    r_t = batch_returns[:, t]
                    gross_ret_t = (noisy_act_t * r_t).sum(dim=1, keepdim=True)
                    denom = torch.clamp(1.0 + gross_ret_t, min=1e-8)
                    prev_w = (noisy_act_t * (1.0 + r_t)) / denom
            
            # Reconstruct batch trajectories: (B, T, action_dim)
            b_clean_actions = torch.stack(all_clean_actions, dim=1)
            b_noisy_actions = torch.stack(all_noisy_actions, dim=1)
            b_prev_weights = torch.stack(all_prev_weights, dim=1)
            
            # 1-Year Temporal Chunking for Advantage Computation
            num_chunks = T_steps // chunk_size
            for i in range(curr_batch_trajs):
                traj_states = batch_states[i]
                traj_returns = batch_returns[i]
                traj_clean_actions = b_clean_actions[i]
                traj_noisy_actions = b_noisy_actions[i]
                traj_prev_weights = b_prev_weights[i]
                
                for c in range(num_chunks):
                    start_idx = c * chunk_size
                    end_idx = start_idx + chunk_size
                    
                    chunk_states = traj_states[start_idx:end_idx]
                    chunk_returns = traj_returns[start_idx:end_idx]
                    chunk_clean_actions = traj_clean_actions[start_idx:end_idx]
                    chunk_noisy_actions = traj_noisy_actions[start_idx:end_idx]
                    chunk_prev_weights = traj_prev_weights[start_idx:end_idx]
                    
                    # 1. Clean Baseline Sharpe
                    c_ret = (chunk_clean_actions * chunk_returns).sum(dim=1)
                    c_sharpe = (c_ret.mean() / (c_ret.std() + 1e-9)).item() * np.sqrt(252)
                    iteration_clean_sharpes.append(c_sharpe)
                    
                    # 2. Noisy Exploration Sharpe
                    n_ret = (chunk_noisy_actions * chunk_returns).sum(dim=1)
                    n_sharpe = (n_ret.mean() / (n_ret.std() + 1e-9)).item() * np.sqrt(252)
                    iteration_noisy_sharpes.append(n_sharpe)
                    
                    # 3. Localized Daily Utility & Paired Advantage
                    c_utility = compute_daily_utility(
                        chunk_clean_actions, chunk_returns, chunk_states, gamma=gamma, cost_c=cost_c, prev_weights=chunk_prev_weights
                    )
                    n_utility = compute_daily_utility(
                        chunk_noisy_actions, chunk_returns, chunk_states, gamma=gamma, cost_c=cost_c, prev_weights=chunk_prev_weights
                    )
                    daily_advantage = n_utility - c_utility
                    
                    buffer.add_trajectory(chunk_states, chunk_prev_weights, chunk_noisy_actions, daily_advantage)
            
            pbar.update(curr_batch_trajs)
        pbar.close()
        
        # Financial Evaluation Metrics
        mean_clean_sharpe = np.mean(iteration_clean_sharpes)
        mean_noisy_sharpe = np.mean(iteration_noisy_sharpes)
        max_noisy_sharpe = np.max(iteration_noisy_sharpes)
        mean_paired_sharpe = mean_noisy_sharpe - mean_clean_sharpe
        
        logger.info(
            f"Baseline Conditioned IL Sharpe: {mean_clean_sharpe:.4f} | RL Exploration Mean: {mean_noisy_sharpe:.4f} | "
            f"Max: {max_noisy_sharpe:.4f} | Paired Sharpe Delta: {mean_paired_sharpe:.4f}"
        )
        
        # Advantage Normalization
        all_adv = torch.cat(buffer.advantages, dim=0)
        adv_mean = all_adv.mean()
        adv_std = all_adv.std() + 1e-8
        for i in range(len(buffer.advantages)):
            buffer.advantages[i] = (buffer.advantages[i] - adv_mean) / adv_std
            
        # Diagnostics & Effective Sample Size
        adv_tensor = torch.cat(buffer.advantages, dim=0)
        raw_weights = torch.exp(adv_tensor / rl_cfg["beta"])
        clip_max = rl_cfg["weight_clip"]
        clip_rate = (raw_weights >= clip_max).float().mean().item()
        clipped_weights = torch.clamp(raw_weights, max=clip_max)
        n_eff = (clipped_weights.sum() ** 2) / (clipped_weights ** 2).sum()
        
        logger.info(f"Diagnostics | Clipping Rate: {clip_rate*100:.2f}% | N_eff: {n_eff.item():.2f} / {len(raw_weights)}")
        
        adv_q = torch.quantile(adv_tensor, torch.tensor([0.25, 0.5, 0.75, 0.90, 0.99]))
        w_q = torch.quantile(clipped_weights, torch.tensor([0.25, 0.5, 0.75, 0.90, 0.99]))
        
        logger.info(f"Advantage Quantiles [25, 50, 75, 90, 99]: {[round(x, 4) for x in adv_q.tolist()]}")
        logger.info(f"Weight Quantiles    [25, 50, 75, 90, 99]: {[round(x, 4) for x in w_q.tolist()]}")

        # Compile and Train
        buffer.compile_buffer()
        trainer.train(epochs=rl_cfg["epochs_per_iteration"])
        
        # Reset buffer for next iteration
        buffer = ConditionedAWRReplayBuffer()
        trainer.replay_buffer = buffer
        
    # 5. Save the Conditioned RL Winner
    rl_model_path = os.path.join(root_dir, "rl_conditioned.pth")
    torch.save(model.state_dict(), rl_model_path)
    logger.info(f"Saved Conditioned RL Winner to {rl_model_path}!")

# Backwards compatibility alias
run_rl = run_rl_unconditioned

if __name__ == "__main__":
    run_rl_conditioned()