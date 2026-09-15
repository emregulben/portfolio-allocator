import pytest
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from agents.dataset import PortfolioDataset
from agents.trainer import PortDiffTrainer
from agents.diffusion import project_portfolio_weights

def test_portfolio_dataset_temporal_alignment():
    """Proves mathematically that there is zero future data leakage in the dataset."""
    np.random.seed(42)
    # 1. Create dummy returns (100 days, 2 assets)
    returns_data = np.random.randn(100, 2)
    returns_df = pd.DataFrame(returns_data)
    
    # 2. Create dummy expert weights (generated for days 60-99)
    weights_data = np.random.rand(40, 3) 
    weights_df = pd.DataFrame(weights_data, index=range(60, 100))
    
    # 3. Initialize Dataset
    window_size = 60
    dataset = PortfolioDataset(returns_df, weights_df, window_size=window_size, flatten=True)
    
    # 4. Verify Temporal Alignment
    state, label = dataset[0] # Grab the very first sample
    
    # The label should be exactly weights_df.iloc[0] (which is for Day 60)
    assert np.allclose(label.numpy(), weights_data[0])
    
    # The state should be exactly the 60 days strictly BEFORE Day 60 (Days 0 to 59)
    expected_state = returns_data[0:60].flatten()
    assert np.allclose(state.numpy(), expected_state)
    
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    def forward(self, state, action):
        return torch.tensor(0.5, requires_grad=True)
    def sample(self, state):
        return torch.ones((state.shape[0], 2))

def test_trainer_tracks_best_model():
    """Proves that the trainer ignores overfitted final epochs and restores the best MAE checkpoint."""
    # 1. Initialize Dummy Data & Trainer
    returns_df = pd.DataFrame(np.random.randn(80, 2))
    weights_df = pd.DataFrame(np.random.rand(20, 2), index=range(60, 80))
    dataset = PortfolioDataset(returns_df, weights_df, window_size=60, flatten=True)
    
    model = DummyModel()
    trainer = PortDiffTrainer(model, dataset, dataset, learning_rate=1e-3, batch_size=4)
    initial_weight = model.linear.weight.clone()
    
    # 2. Mock the validation loop to simulate overfitting: MAEs go from 0.8 -> 0.2 -> 0.9
    mock_maes = [(0.5, 0.8), (0.4, 0.2), (0.3, 0.9)]
    state = {"idx": 0} 
    def mock_validate():
        res = mock_maes[state["idx"]]
        state["idx"] += 1
        return res
    
    # 3. Mock the training loop to artificially change the weights every epoch
    def mock_train():
        with torch.no_grad():
            model.linear.weight += 1.0 # Add 1.0 to the weights every epoch
        return 0.5
        
    trainer.validate_epoch = mock_validate
    trainer.train_epoch = mock_train
    
    # 4. Run training for 3 epochs
    trainer.train(epochs=3)
    
    # By Epoch 3, the weights reached `initial_weight + 3.0`.
    # But because Epoch 2 had the best MAE (0.2), it should have restored the weights to `initial_weight + 2.0`!
    expected_weight = initial_weight + 2.0
    assert torch.allclose(model.linear.weight, expected_weight), "The trainer failed to restore the weights from Epoch 2!"
    
def test_custom_projection_algorithm():
    """Proves the projection correctly enforces Markowitz constraints and dumps excess to cash."""
    # Create 3 raw dummy outputs from the model (Batch of 3)
    # Asset format: [StockA, StockB, StockC, Cash]
    raw_outputs = torch.tensor([
        [0.80, 0.10, 0.10, 0.00],  # Case 1: Bull market. Stock A is way over 20%.
        [-0.50, 0.40, 0.20, -0.1], # Case 2: Neural network outputs illegal negative numbers.
        [0.10, 0.10, 0.10, 0.70],  # Case 3: Completely valid, safe portfolio.
    ])
    
    projected = project_portfolio_weights(raw_outputs, max_weight=0.20)
    
    # Check Case 1: [0.8, 0.1, 0.1, 0.0] -> Stock A loses 0.6. Cash absorbs 0.6.
    assert torch.allclose(projected[0], torch.tensor([0.20, 0.10, 0.10, 0.60]))
    
    # Check Case 2: [-0.5, 0.4, 0.2, -0.1]
    # Clamped to >= 0: [0, 0.4, 0.2, 0]
    # L1 Normalized (Sum=0.6): [0, 0.666, 0.333, 0]
    # Capped to 0.20: Stock B loses 0.466, Stock C loses 0.133. Total excess = 0.60
    # Cash absorbs the 0.60 excess perfectly!
    assert torch.allclose(projected[1], torch.tensor([0.0, 0.20, 0.20, 0.60]), atol=1e-3)
    
    # Check Case 3: Already valid, the projection should leave it entirely untouched.
    assert torch.allclose(projected[2], torch.tensor([0.10, 0.10, 0.10, 0.70]))
    
    # Verify universal mathematical constraints on the whole batch
    assert torch.all(projected >= 0), "No weights can be negative"
    assert torch.allclose(projected.sum(dim=-1), torch.ones(3)), "All portfolios must sum to 1.0"
    assert torch.all(projected[:, :-1] <= 0.2001), "No risky asset can exceed 20%"