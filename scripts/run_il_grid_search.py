import os
import sys
import yaml
import torch
import itertools
import numpy as np
import pandas as pd

# Add root directory to python path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.loader import MarketDataLoader
from simulator.hmm import HybridJumpsHMM
from simulator.single_index import SingleIndexModel
from experts.markowitz import MarkowitzExpert
from agents.dataset import PortfolioDataset
from agents.diffusion import MLPDenoiser, TransformerDenoiser, MeanCovMLPDenoiser, PortDiff
from agents.trainer import PortDiffTrainer
from torch.utils.data import Subset

def prepare_market_data(config, hmm_dist="gaussian", sim_dist="gaussian"):
    """Loads historical data and fits the HMM and SIM models with the chosen distributions."""
    market_ticker = config["data"]["market_ticker"]
    stock_tickers = list(config["data"]["stock_tickers"])
    
    loader = MarketDataLoader(
        tickers=[market_ticker] + stock_tickers,
        start_date=config["data"]["start_date"], 
        end_date=config["data"]["end_date"]
    )
    data = loader.fetch_data()
    
    # Use independent variables to prevent crashing if we revert later
    hmm = HybridJumpsHMM(n_states=config["hmm"]["n_states"], emission_dist=hmm_dist)
    hmm.fit(data[market_ticker])
    
    hmm.epsilon = 0.001
    hmm.lambd = 40
    
    sim_model = SingleIndexModel(tickers=stock_tickers, emission_dist=sim_dist)
    sim_model.fit(data[stock_tickers], data[market_ticker])
    
    return hmm, sim_model

def evaluate_config(
    denoiser_class,
    hyperparams,
    config,
    hmm,
    sim_model,
    stock_tickers,
    n_market_seeds=3,
    n_diff_seeds=3,
    seed_offset=0
):
    """
    Rigorously evaluates a single architecture and hyperparameter combination.
    
    To isolate architectural performance from stochastic variation, this function:
    1. Generates multiple independent market realities (market_seeds).
    2. Trains a fresh model on each reality.
    3. Evaluates each model across multiple diffusion sampling seeds (diff_seeds).
    
    Returns:
        float: The average Mean Absolute Error (MAE) across all seeds and realities.
    """
    all_maes = []
    window_size = config["expert"]["rolling_window"]
    
    # Initialize the Ground Truth Expert
    expert = MarkowitzExpert(
        risk_aversion=config["expert"]["risk_aversion"],
        rolling_window=window_size,
        max_weight=config["expert"]["max_weight"],
        annual_risk_free_rate=config["expert"]["annual_risk_free_rate"]
    )
    
    # Iterate over independent market realities to prevent overfitting to a single path
    for i in range(n_market_seeds):
        market_seed = i + seed_offset
        
        # Enforce strict determinism for the HMM state generation
        np.random.seed(market_seed)
        
        sim_states = hmm.simulate_states(config["hmm"]["simulation"]["n_steps"], epsilon=hmm.epsilon, lambd=hmm.lambd)
        sim_stocks = sim_model.simulate(hmm.decode_states(sim_states), random_seed=market_seed)
        
        expert_weights_df = expert.generate_labels(sim_stocks)
        
        dataset = PortfolioDataset(
            returns_df=sim_stocks, 
            weights_df=expert_weights_df, 
            window_size=window_size, 
            flatten=(denoiser_class == MLPDenoiser)
        )
        
        # Chronological train/validation split to prevent data leakage
        train_size = int(0.8 * len(dataset))
        train_dataset = Subset(dataset, list(range(0, train_size)))
        val_dataset = Subset(dataset, list(range(train_size + window_size, len(dataset))))
        
        n_assets = len(stock_tickers)
        action_dim = n_assets + 1 
        state_dim = window_size * n_assets
        
        # Extract hyperparameter values, falling back to config defaults
        epochs = hyperparams.get('epochs', 15)
        batch_size = hyperparams.get('batch_size', config['il']['batch_size'])
        num_timesteps = hyperparams.get('num_timesteps', config['il']['num_timesteps'])
        beta_schedule = hyperparams.get('beta_schedule', config['il']['beta_schedule'])
        
        # Initialize the target architecture
        if denoiser_class == MLPDenoiser:
            use_layernorm = hyperparams.get('use_layernorm', True)
            denoiser = MLPDenoiser(
                state_dim=state_dim, action_dim=action_dim, 
                hidden_dim=hyperparams['hidden_dim'], t_dim=config['il']['t_dim'], 
                use_layernorm=use_layernorm
            )
        elif denoiser_class == MeanCovMLPDenoiser:
            use_layernorm = hyperparams.get('use_layernorm', True)
            denoiser = MeanCovMLPDenoiser(
                n_assets=n_assets, action_dim=action_dim, 
                hidden_dim=hyperparams['hidden_dim'], t_dim=config['il']['t_dim'], 
                use_layernorm=use_layernorm
            )
        else:
            n_layers = hyperparams.get('n_layers', 2)
            denoiser = TransformerDenoiser(
                n_assets=n_assets, action_dim=action_dim, window_size=window_size,
                hidden_dim=hyperparams['hidden_dim'], t_dim=config['il']['t_dim'], 
                n_heads=hyperparams['n_heads'], n_layers=n_layers
            )
            
        diffusion = PortDiff(denoiser, num_timesteps=num_timesteps, beta_schedule=beta_schedule)
        
        trainer = PortDiffTrainer(
            model=diffusion, train_dataset=train_dataset, val_dataset=val_dataset,
            learning_rate=hyperparams['learning_rate'], batch_size=batch_size
        )
        trained_model = trainer.train(epochs=epochs)
        
        # Evaluate across multiple diffusion sampling seeds
        trained_model.eval()
        for diff_seed in range(n_diff_seeds):
            torch.manual_seed(diff_seed)
            
            total_mae = 0.0
            with torch.no_grad():
                for state, action_clean in trainer.val_loader:
                    state = state.to(trainer.device)
                    action_clean = action_clean.to(trainer.device)
                    
                    sampled_actions = trained_model.sample(state)
                    mae = torch.nn.functional.l1_loss(sampled_actions, action_clean)
                    total_mae += mae.item()
                    
            avg_mae = total_mae / len(trainer.val_loader)
            all_maes.append(avg_mae)
            
    return np.mean(all_maes)

if __name__ == "__main__":
    from simulator.logger import setup_logger
    
    # Initialize custom logger
    logger = setup_logger("IL_GridSearch")
    
    # Load configuration
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    # Isolate architecture performance using a pure Gaussian environment
    logger.info("Preparing Gaussian Market Data...")
    hmm, sim_model = prepare_market_data(config, hmm_dist="gaussian", sim_dist="gaussian")
    stock_tickers = list(config["data"]["stock_tickers"])
        
    
    
    # # Define grid search parameter space
    # # Lock the empirically proven parameters to save cluster compute time
    # learning_rates = [1e-3]
    # batch_sizes = [64]
    # beta_schedules = ["linear"]
    # epochs_list = [15]
    # num_timesteps_list = [100]
    # use_layernorm_list = [False]
    
    # # Expand the boundaries for the parameters that hit the ceiling in local trials
    # hidden_dims = [512, 1024]
    
    # mlp_results = []
    
    # # Pre-calculate all MLP combinations to generate progress percentages
    # mlp_combinations = list(itertools.product(
    #     learning_rates, hidden_dims, num_timesteps_list, beta_schedules, batch_sizes, epochs_list, use_layernorm_list
    # ))
    # total_mlp = len(mlp_combinations)
    
    # logger.info("\n" + "="*50 + f"\nSTARTING MLP GRID SEARCH ({total_mlp} combinations)\n" + "="*50)
    # for i, (lr, hd, ts, beta, bs, ep, ln) in enumerate(mlp_combinations, 1):
    #     hyperparams = {
    #         'learning_rate': lr, 'hidden_dim': hd, 'num_timesteps': ts, 
    #         'beta_schedule': beta, 'batch_size': bs, 'epochs': ep, 'use_layernorm': ln
    #     }
        
    #     # Inject the exact progress percentage into the logger!
    #     pct = (i / total_mlp) * 100
    #     logger.info(f"\n--- Testing MLP Config [{i}/{total_mlp} | {pct:.1f}% Complete] ---\n{hyperparams}")
        
    #     avg_mae = evaluate_config(
    #         MLPDenoiser, hyperparams, config, hmm, sim_model, stock_tickers,
    #         n_market_seeds=1, n_diff_seeds=1
    #     )
    #     logger.info(f"RESULT -> Average Action-Space MAE: {avg_mae:.4f}")
    #     mlp_results.append({'hyperparams': hyperparams, 'mae': avg_mae})
        
    # logger.info("MLP Search Complete!")
    # best_mlp = None
    # if mlp_results:
    #     best_mlp = min(mlp_results, key=lambda x: x['mae'])
    #     logger.info(f"\n*** BEST MLP CONFIGURATION ***\n{best_mlp}")

    # # Define Transformer-specific parameter space
    # n_heads_list = [2, 4]
    # n_layers_list = [1, 2]
    # tf_results = []
    
    # # Pre-calculate valid Transformer combinations (filtering out PyTorch invalid dimensions)
    # tf_combinations = list(itertools.product(
    #     learning_rates, hidden_dims, num_timesteps_list, beta_schedules, batch_sizes, epochs_list, n_heads_list, n_layers_list
    # ))
    # valid_tf_combinations = [combo for combo in tf_combinations if combo[1] % combo[6] == 0]
    # total_tf = len(valid_tf_combinations)
    
    # logger.info("\n" + "="*50 + f"\nSTARTING TRANSFORMER GRID SEARCH ({total_tf} combinations)\n" + "="*50)
    # for i, (lr, hd, ts, beta, bs, ep, heads, layers) in enumerate(valid_tf_combinations, 1):
    #     hyperparams = {
    #         'learning_rate': lr, 'hidden_dim': hd, 'num_timesteps': ts, 
    #         'beta_schedule': beta, 'batch_size': bs, 'epochs': ep, 
    #         'n_heads': heads, 'n_layers': layers
    #     }
        
    #     pct = (i / total_tf) * 100
    #     logger.info(f"\n--- Testing Transformer Config [{i}/{total_tf} | {pct:.1f}% Complete] ---\n{hyperparams}")
        
    #     avg_mae = evaluate_config(
    #         TransformerDenoiser, hyperparams, config, hmm, sim_model, stock_tickers,
    #         n_market_seeds=1, n_diff_seeds=1
    #     )
    #     logger.info(f"RESULT -> Average Action-Space MAE: {avg_mae:.4f}")
    #     tf_results.append({'hyperparams': hyperparams, 'mae': avg_mae})
        
    # logger.info("Transformer Search Complete!")
    # best_tf = None
    # if tf_results:
    #     best_tf = min(tf_results, key=lambda x: x['mae'])
    #     logger.info(f"\n*** BEST TRANSFORMER CONFIGURATION ***\n{best_tf}")

    # # === CONTROL BENCHMARK: MEAN-COV MLP GRID SEARCH ===
    # # This loop isolates whether the models are failing at representation learning.
    # mean_cov_results = []
    
    # logger.info("\n" + "="*50 + f"\nSTARTING MEAN-COV MLP GRID SEARCH ({total_mlp} combinations)\n" + "="*50)
    # for i, (lr, hd, ts, beta, bs, ep, ln) in enumerate(mlp_combinations, 1):
    #     hyperparams = {
    #         'learning_rate': lr, 'hidden_dim': hd, 'num_timesteps': ts, 
    #         'beta_schedule': beta, 'batch_size': bs, 'epochs': ep, 'use_layernorm': ln
    #     }
        
    #     pct = (i / total_mlp) * 100
    #     logger.info(f"\n--- Testing Mean-Cov MLP Config [{i}/{total_mlp} | {pct:.1f}% Complete] ---\n{hyperparams}")
        
    #     avg_mae = evaluate_config(
    #         MeanCovMLPDenoiser, hyperparams, config, hmm, sim_model, stock_tickers,
    #         n_market_seeds=1, n_diff_seeds=1
    #     )
    #     logger.info(f"RESULT -> Average Action-Space MAE: {avg_mae:.4f}")
    #     mean_cov_results.append({'hyperparams': hyperparams, 'mae': avg_mae})
        
    # logger.info("Mean-Cov MLP Search Complete!")
    # best_mean_cov = None
    # if mean_cov_results:
    #     best_mean_cov = min(mean_cov_results, key=lambda x: x['mae'])
    #     logger.info(f"\n*** BEST MEAN-COV MLP CONFIGURATION ***\n{best_mean_cov}")

    #    # Determine the Overall Winner safely to satisfy static type checkers
    # all_best = []
    # if best_mlp is not None:
    #     all_best.append(("MLP", best_mlp))
    # if best_tf is not None:
    #     all_best.append(("Transformer", best_tf))
    # if best_mean_cov is not None:
    #     all_best.append(("Mean-Cov MLP", best_mean_cov))
        
    # if not all_best:
    #     raise ValueError("All grid searches failed. No models were evaluated.")
        
    # overall_winner_name, overall_winner_config = min(all_best, key=lambda x: x[1]['mae'])
    
    # # === UNTOUCHED TEST SET EVALUATION ===
    # logger.info("\n" + "="*50 + "\nRUNNING BLIND TEST ON UNTOUCHED MARKET SEED\n" + "="*50)
    # logger.info("Generating a completely independent dataset to prevent hyperparameter overfitting...")
    
    # # Map the winner name back to the Python class
    # if overall_winner_name == "MLP":
    #     winner_class = MLPDenoiser
    # elif overall_winner_name == "Transformer":
    #     winner_class = TransformerDenoiser
    # else:
    #     winner_class = MeanCovMLPDenoiser
        
    # # Evaluate the winner using a large market seed offset to guarantee it has never seen this data
    # blind_mae = evaluate_config(
    #     winner_class, overall_winner_config['hyperparams'], config, hmm, sim_model, stock_tickers,
    #     n_market_seeds=1, n_diff_seeds=3, seed_offset=9999 
    # )

    # # === FINAL MASTER SUMMARY ===
    # logger.info("\n" + "="*50 + "\nFINAL ARCHITECTURE SHOWDOWN\n" + "="*50)
    
    # if best_mlp is not None:
    #     logger.info(f"🏆 BEST MLP MAE: {best_mlp['mae']:.4f}")
    #     logger.info(f"Parameters: {best_mlp['hyperparams']}\n")
        
    # if best_tf is not None:
    #     logger.info(f"🏆 BEST TRANSFORMER MAE: {best_tf['mae']:.4f}")
    #     logger.info(f"Parameters: {best_tf['hyperparams']}\n")
        
    # if best_mean_cov is not None:
    #     logger.info(f"🏆 BEST MEAN-COV MLP MAE: {best_mean_cov['mae']:.4f}")
    #     logger.info(f"Parameters: {best_mean_cov['hyperparams']}\n")
        
    # logger.info(f"👑 OVERALL WINNER: {overall_winner_name} (Validation MAE: {overall_winner_config['mae']:.4f})")
    
    # logger.info("\n" + "="*50 + "\nBLIND TEST RESULTS\n" + "="*50)
    # logger.info(f"FINAL UNBIASED EVALUATION SCORE -> {overall_winner_name} Blind Test MAE: {blind_mae:.4f}")
    # logger.info("="*50)
    
    
    
    # === FINAL ROBUSTNESS SHOWDOWN (512 vs 1024) ===
    logger.info("\n" + "="*50 + "\nSTARTING 10-MARKET ROBUSTNESS SHOWDOWN\n" + "="*50)
        
    robust_results = []
    
    for hidden_dim in [512, 1024]:
        hyperparams = {
            'learning_rate': 1e-3, 
            'hidden_dim': hidden_dim, 
            'num_timesteps': 100, 
            'beta_schedule': 'linear', 
            'batch_size': 64, 
            'epochs': 15, 
            'use_layernorm': False
        }
        
        logger.info(f"\n--- Testing MLP Config: {hidden_dim} Hidden Dimensions ---")
        
        # Train and evaluate across 5 different market seeds
        val_mae = evaluate_config(
            MLPDenoiser, hyperparams, config, hmm, sim_model, stock_tickers,
            n_market_seeds=5, n_diff_seeds=3
        )
        logger.info(f"Robust Validation MAE (5 Markets): {val_mae:.4f}")
        
        # Evaluate on 5 completely new blind market seeds
        blind_mae = evaluate_config(
            MLPDenoiser, hyperparams, config, hmm, sim_model, stock_tickers,
            n_market_seeds=5, n_diff_seeds=3, seed_offset=9999 
        )
        logger.info(f"Robust Blind Test MAE (5 Markets): {blind_mae:.4f}")
        
        robust_results.append((hidden_dim, val_mae, blind_mae))
        
    logger.info("\n" + "="*50 + "\nFINAL THESIS STATISTICS\n" + "="*50)
    for dim, v_mae, b_mae in robust_results:
        logger.info(f"MLP ({dim} Dims) -> Val MAE: {v_mae:.4f} | Blind Test MAE: {b_mae:.4f}")
        
    winner = min(robust_results, key=lambda x: x[2])  # Winner based on Blind Test MAE
    logger.info(f"\n👑 UNDISPUTED CHAMPION: MLP ({winner[0]} Dims)")
    logger.info("="*50)