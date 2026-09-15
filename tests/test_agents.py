import pytest
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from agents.dataset import PortfolioDataset
from agents.trainer import PortDiffTrainer

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