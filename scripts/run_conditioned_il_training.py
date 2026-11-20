import argparse
import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.loader import MarketDataLoader
from simulator.hmm import HybridJumpsHMM
from simulator.single_index import SingleIndexModel
from experts.markowitz import MarkowitzExpert
from agents.dataset import ConditionedPortfolioDataset
from agents.diffusion import build_il_model
from agents.trainer import PortDiffTrainer
from simulator.logger import setup_logger

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_simulation_environment(config):
    """Initializes the market data loader, HMM simulator, and single-index model."""
    market_ticker = config["data"]["market_ticker"]
    stock_tickers = list(config["data"]["stock_tickers"])

    loader = MarketDataLoader(
        tickers=[market_ticker] + stock_tickers,
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        cache_path=os.path.join(ROOT_DIR, config["data"]["snapshot_path"])
    )
    data = loader.fetch_data()

    hmm = HybridJumpsHMM(n_states=config["hmm"]["n_states"], emission_dist="gaussian")
    hmm.fit(data[market_ticker])
    hmm.lambd = 0.0
    hmm.epsilon = 0.0

    sim_model = SingleIndexModel(tickers=stock_tickers, emission_dist="gaussian")
    sim_model.fit(data[stock_tickers], data[market_ticker])

    return hmm, sim_model


def generate_conditioned_dataset(config, hmm, sim_model, n_seeds=5, seed_offset=0):
    """Simulates independent market histories and packages them into ConditionedPortfolioDatasets."""
    all_datasets = []

    for i in range(n_seeds):
        current_seed = i + seed_offset
        np.random.seed(current_seed)

        hmm_states = hmm.simulate_states(
            config["hmm"]["simulation"]["n_steps"],
            epsilon=hmm.epsilon,
            lambd=hmm.lambd
        )
        market_returns = hmm.decode_states(hmm_states)
        stock_returns = sim_model.simulate(market_returns, random_seed=current_seed)

        expert = MarkowitzExpert(
            rolling_window=config["expert"]["rolling_window"],
            risk_aversion=config["expert"]["risk_aversion"],
            max_weight=config["expert"]["max_weight"],
            annual_risk_free_rate=config["expert"]["annual_risk_free_rate"]
        )
        expert_weights = expert.generate_labels(stock_returns)

        dataset = ConditionedPortfolioDataset(
            returns_df=stock_returns,
            weights_df=expert_weights,
            window_size=config["expert"]["rolling_window"],
            flatten=False
        )
        all_datasets.append(dataset)

    return ConcatDataset(all_datasets)


def train_conditioned_il(device_name: str = "cpu"):
    logger = setup_logger("Conditioned_IL_Training")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(root_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    logger.info("Setting up market simulation environment...")
    hmm, sim_model = setup_simulation_environment(config)

    # 1. Dataset Generation
    logger.info("Generating training dataset across 5 seeds...")
    train_dataset = generate_conditioned_dataset(config, hmm, sim_model, n_seeds=5, seed_offset=0)

    logger.info("Generating validation dataset across 5 seeds...")
    val_dataset = generate_conditioned_dataset(config, hmm, sim_model, n_seeds=5, seed_offset=5)

    n_assets = len(config["data"]["stock_tickers"])
    il_cfg = config["il"]

    # 2. Conditioned Model Architecture (same settings as the unconditioned winner)
    logger.info(f"Initializing conditioned MeanCovMLPDenoiser ({il_cfg['hidden_dim']} dims)...")
    torch.manual_seed(il_cfg["seed"])
    model = build_il_model(il_cfg, n_assets, conditioned=True)

    trainer = PortDiffTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=il_cfg["learning_rate"],
        batch_size=il_cfg["batch_size"],
        device=device_name
    )

    # 3. Training Loop
    logger.info(f"Training conditioned model for {il_cfg['epochs']} epochs...")
    trained_model = trainer.train(epochs=il_cfg["epochs"])
    trained_model.eval()

    # 4. Save Checkpoint
    output_path = os.path.join(root_dir, "il_conditioned.pth")
    torch.save(trained_model.state_dict(), output_path)
    logger.info(f"Saved conditioned IL checkpoint to {output_path}")

    # 5. Blind Generalization Test on 5 Unseen Markets
    logger.info("Evaluating on 5 unseen test markets (seeds 9999-10003)...")
    test_dataset = generate_conditioned_dataset(config, hmm, sim_model, n_seeds=5, seed_offset=9999)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    seed_maes = []
    with torch.no_grad():
        for state, prev_weights, action_clean in test_loader:
            state = state.to(trainer.device)
            prev_weights = prev_weights.to(trainer.device)
            action_clean = action_clean.to(trainer.device)

            sampled_actions = trained_model.sample(state, prev_weights=prev_weights, sampler="ddim")
            mae = torch.nn.functional.l1_loss(sampled_actions, action_clean)
            seed_maes.append(mae.item())

    avg_test_mae = float(np.mean(seed_maes))
    logger.info(f"Conditioned IL Test MAE (Deterministic DDIM): {avg_test_mae:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the conditioned IL policy (il_conditioned.pth).")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu",
                        help="CPU is the default: it is reproducible, and MPS fails after macOS 27.2.")
    train_conditioned_il(device_name=parser.parse_args().device)