import os
import sys
import yaml
import torch
import numpy as np

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.loader import MarketDataLoader
from simulator.hmm import HybridJumpsHMM
from simulator.single_index import SingleIndexModel
from experts.markowitz import MarkowitzExpert
from agents.dataset import PortfolioDataset
from agents.diffusion import MeanCovMLPDenoiser, PortDiff
from agents.trainer import PortDiffTrainer
from torch.utils.data import ConcatDataset, DataLoader
from simulator.logger import setup_logger

def setup_pure_gaussian_environment(config):
    """Initializes the market simulators strictly without jump crashes."""
    market_ticker = config["data"]["market_ticker"]
    stock_tickers = list(config["data"]["stock_tickers"])
    
    loader = MarketDataLoader(
        tickers=[market_ticker] + stock_tickers,
        start_date=config["data"]["start_date"], 
        end_date=config["data"]["end_date"]
    )
    data = loader.fetch_data()
    
    hmm = HybridJumpsHMM(n_states=config["hmm"]["n_states"], emission_dist="gaussian")
    hmm.fit(data[market_ticker])
    
    # Disable the jump mechanism to provide a pure Gaussian baseline for Markowitz evaluation
    hmm.lambd = 0.0
    hmm.epsilon = 0.0
    
    sim_model = SingleIndexModel(tickers=stock_tickers, emission_dist="gaussian")
    sim_model.fit(data[stock_tickers], data[market_ticker])
    
    return hmm, sim_model

def generate_combined_dataset(config, hmm, sim_model, n_seeds=5, seed_offset=0):
    """
    Simulates multiple independent market histories and concatenates them into a single dataset.
    """
    all_datasets = []
    
    for i in range(n_seeds):
        current_seed = i + seed_offset
        
        # Enforce strict determinism for the HMM state generation
        np.random.seed(current_seed)
        
        # 1. Simulate an independent market reality
        hmm_states = hmm.simulate_states(config["hmm"]["simulation"]["n_steps"], epsilon=hmm.epsilon, lambd=hmm.lambd)
        market_returns = hmm.decode_states(hmm_states)
        stock_returns = sim_model.simulate(market_returns, random_seed=current_seed)
        
        # 2. Extract corresponding Markowitz expert weights
        expert = MarkowitzExpert(
            rolling_window=config["expert"]["rolling_window"],
            risk_aversion=config["expert"]["risk_aversion"],
            max_weight=config["expert"]["max_weight"]
        )
        expert_weights = expert.generate_labels(stock_returns)
        
        # 3. Create dataset and append
        dataset = PortfolioDataset(
            returns_df=stock_returns,
            weights_df=expert_weights,
            window_size=config["expert"]["rolling_window"],
            # MeanCovMLPDenoiser expects an explicit 3D tensor shape (batch, window, assets) to calculate statistics
            flatten=False 
        )
        all_datasets.append(dataset)
        
    return ConcatDataset(all_datasets)

if __name__ == "__main__":
    logger = setup_logger("Final_IL_Evaluation")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    logger.info("Initializing Pure Gaussian Environment...")
    hmm, sim_model = setup_pure_gaussian_environment(config)
    
    # 1. Data Generation
    logger.info("Generating Combined 5-Market Training Curriculum...")
    train_dataset = generate_combined_dataset(config, hmm, sim_model, n_seeds=5, seed_offset=0)
    
    logger.info("Generating Combined 5-Market Validation Curriculum...")
    val_dataset = generate_combined_dataset(config, hmm, sim_model, n_seeds=5, seed_offset=5)
    
    n_assets = len(config["data"]["stock_tickers"])
    action_dim = n_assets + 1
    
    # 2. Model Initialization
    logger.info("Initializing the Winner Architecture: MeanCovMLPDenoiser (1024 Dims)...")
    denoiser = MeanCovMLPDenoiser(
        n_assets=n_assets,
        action_dim=action_dim,
        hidden_dim=1024
    )
    
    model = PortDiff(
        denoiser=denoiser, 
        num_timesteps=100, 
        beta_schedule="linear"
    )
    
    trainer = PortDiffTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset, 
        batch_size=64
    )
    
    # 3. Execution
    logger.info("Training the Final Model...")
    trained_model = trainer.train(epochs=15)
    
    # Freeze model weights to prevent parameter updates during evaluation
    logger.info("\nTraining Complete! Permanently freezing model weights for blind evaluation...")
    trained_model.eval()
    
    # Save the Winner IL Model to the hard drive so RL can load it later
    model_path = os.path.join(root_dir, "il_winner.pth")
    torch.save(trained_model.state_dict(), model_path)
    logger.info(f"Saved IL Winner to {model_path}")
    
    logger.info("Generating 5 Completely Unseen Test Markets (Seed Offset = 9999)...")
    test_dataset = generate_combined_dataset(config, hmm, sim_model, n_seeds=5, seed_offset=9999)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    logger.info("\n" + "="*50)
    logger.info("INITIATING BLIND GENERALIZATION TEST")
    logger.info("="*50)
    
    total_mae = 0.0
    n_diff_seeds = 3
    
    for diff_seed in range(n_diff_seeds):
        torch.manual_seed(diff_seed)
        seed_mae = 0.0
        
        with torch.no_grad():
            for state, action_clean in test_loader:
                state = state.to(trainer.device)
                action_clean = action_clean.to(trainer.device)
                
                sampled_actions = trained_model.sample(state)
                mae = torch.nn.functional.l1_loss(sampled_actions, action_clean)
                seed_mae += mae.item()
                
        avg_seed_mae = seed_mae / len(test_loader)
        total_mae += avg_seed_mae
        logger.info(f"Diffusion Seed {diff_seed} -> MAE: {avg_seed_mae:.4f}")
        
    final_blind_mae = total_mae / n_diff_seeds
    logger.info("="*50)
    logger.info(f"FINAL BLIND TEST MAE (5 UNSEEN MARKETS): {final_blind_mae:.4f}")
    logger.info("="*50)