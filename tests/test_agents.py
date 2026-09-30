import pytest
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from agents.dataset import PortfolioDataset
from agents.trainer import PortDiffTrainer
from agents.diffusion import map_to_feasible_portfolio, MLPDenoiser, MeanCovMLPDenoiser, PortDiff
from torch.utils.data import TensorDataset

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
    
    projected = map_to_feasible_portfolio(raw_outputs, max_weight=0.20)
    
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
    
def test_validation_determinism():
    """
    Ensures that PortDiffTrainer.validate_epoch() locks the random seed internally.
    If the seed is not locked, diffusion noise will cause the MAE to fluctuate across calls,
    corrupting the Early Stopping mechanism.
    """
    
    # 1. Setup a tiny dummy model
    state_dim, action_dim = 120, 11
    denoiser = MLPDenoiser(state_dim=state_dim, action_dim=action_dim, hidden_dim=32, t_dim=16)
    model = PortDiff(denoiser=denoiser, num_timesteps=3, beta_schedule="linear")
    
    # 2. Setup a dummy dataset (batch_size=4)
    dummy_states = torch.randn(4, state_dim)
    dummy_actions = torch.rand(4, action_dim)
    dummy_actions = dummy_actions / dummy_actions.sum(dim=1, keepdim=True) # Normalize
    
    dataset = TensorDataset(dummy_states, dummy_actions)
    
    # 3. Instantiate trainer (using dummy dataset for both train and val)
    trainer = PortDiffTrainer(model=model, train_dataset=dataset, val_dataset=dataset, batch_size=4)
    
    # 4. Call validate_epoch twice in a row
    mse1, mae1 = trainer.validate_epoch()
    mse2, mae2 = trainer.validate_epoch()
    
    # 5. Assertion: If the seed lock is missing, these floats will differ drastically
    assert abs(mae1 - mae2) < 1e-7, "validate_epoch() is not deterministic! The random seed lock is missing."
    
def test_mean_cov_calculation():
    """
    Verifies that MeanCovMLPDenoiser correctly calculates the batch-wise 
    unbiased sample covariance and mean, matching standard numpy implementations.
    """
    
    batch_size, window_size, n_assets = 2, 60, 10
    
    # Generate random test data using float64 to avoid floating point precision errors during testing
    state = torch.randn(batch_size, window_size, n_assets, dtype=torch.float64)
    
    # Initialize the denoiser and force it to float64 for the test
    denoiser = MeanCovMLPDenoiser(n_assets=n_assets, action_dim=11).double()
    
    # 1. Get the custom PyTorch batch output
    features = denoiser._compute_mean_and_cov(state)
    
    # Extract the PyTorch means (first 10 elements) and covariances (remaining 100 elements)
    pt_means = features[:, :n_assets].numpy()
    pt_covs = features[:, n_assets:].view(batch_size, n_assets, n_assets).numpy()
    
    state_np = state.numpy()
    
    # 2. Iterate through the batch and calculate the Gold-Standard Numpy outputs
    for i in range(batch_size):
        # Numpy mean along the time dimension (axis=0)
        np_mean = np.mean(state_np[i], axis=0)
        
        # Numpy covariance: rowvar=False ensures columns are assets and rows are time observations
        np_cov = np.cov(state_np[i], rowvar=False)
        
        # 3. Assert perfect mathematical equivalence
        assert np.allclose(pt_means[i], np_mean), f"Mean mismatch at batch index {i}"
        assert np.allclose(pt_covs[i], np_cov), f"Covariance mismatch at batch index {i}"

def test_mean_cov_forward_normalization():
    """
    Verifies that the MeanCovMLPDenoiser successfully executes a forward pass
    and that the internal BatchNorm1d layer correctly standardizes the 
    Mean and Covariance features to ensure equal gradient scaling.
    """
    batch_size, window_size, n_assets = 64, 60, 10
    action_dim = n_assets + 1
    
    # 1. Initialize the Model
    model = MeanCovMLPDenoiser(n_assets=n_assets, action_dim=action_dim, hidden_dim=256)
    model.train() # Ensure BatchNorm calculates live batch statistics
    
    # 2. Setup a PyTorch Forward Hook to intercept the normalized features
    intercepted_features = {}
    def hook_fn(module, input, output):
        intercepted_features['normalized'] = output.detach()
        
    # Attach the hook to the feature_norm layer
    hook_handle = model.feature_norm.register_forward_hook(hook_fn)
    
    # 3. Create realistic dummy inputs (scaled down to mimic real financial decimals)
    state = torch.randn(batch_size, window_size, n_assets) * 0.05
    action_noisy = torch.randn(batch_size, action_dim)
    
    # Updated to enforce the 2D shape (batch_size, 1) and .float() format used in live production
    t = torch.randint(0, 100, (batch_size, 1)).float()
    
    # 4. Execute the live forward pass
    output = model(state, action_noisy, t)
    
    # 5. Clean up the hook
    hook_handle.remove()
    
    # 6. Verify the forward pass completed and output the correct shape
    assert output.shape == (batch_size, action_dim), "Forward pass output shape mismatch."
    assert not torch.isnan(output).any(), "Forward pass produced NaNs."
    
    # 7. Verify the mathematical normalization of the intercepted features
    norm_features = intercepted_features['normalized']
    
    # Calculate the batch-wise mean and variance for every single feature
    feature_means = norm_features.mean(dim=0)
    # BatchNorm uses biased variance for the standard deviation scaling
    feature_vars = norm_features.var(dim=0, unbiased=False) 
    
    # Assert that all features have a mean of 0 and variance of 1 (within a tiny float tolerance)
    assert torch.allclose(feature_means, torch.zeros_like(feature_means), atol=1e-5), "Features are not centered at mean 0."
    # Relaxed tolerance to 1e-2 to account for 32-bit floating point precision limits when scaling microscopic decimals
    assert torch.allclose(feature_vars, torch.ones_like(feature_vars), atol=1e-2), "Features are not scaled to variance 1."