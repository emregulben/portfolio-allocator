import pytest
import numpy as np
import pandas as pd
from simulator.single_index import SingleIndexModel


def test_single_index_init_empty_tickers_raises_error():
    """Validates that initializing with an empty tickers list raises ValueError."""
    with pytest.raises(ValueError, match="Tickers list cannot be empty."):
        SingleIndexModel(tickers=[])


def test_single_index_unfitted_simulate_raises_error():
    """Validates that simulating before fitting raises ValueError."""
    model = SingleIndexModel(tickers=["AAPL", "MSFT"])
    dummy_market_sim = np.array([0.01, -0.01])
    with pytest.raises(ValueError, match="SingleIndexModel must be fitted before simulating."):
        model.simulate(dummy_market_sim)


def test_single_index_fit_dimension_mismatch_raises_error():
    """Validates that input length mismatches or ticker column mismatches raise ValueError."""
    model = SingleIndexModel(tickers=["AAPL", "MSFT"])
    
    stock_returns = pd.DataFrame({"AAPL": [0.01, 0.02], "MSFT": [0.005, -0.01]})
    mismatched_market = pd.Series([0.01, 0.02, 0.03])
    
    # 1. Length mismatch
    with pytest.raises(ValueError, match="stock_returns and market_returns must have identical length."):
        model.fit(stock_returns, mismatched_market)
        
    # 2. Ticker names mismatch
    wrong_columns_stock = pd.DataFrame({"AAPL": [0.01, 0.02], "TSLA": [0.005, -0.01]})
    valid_market = pd.Series([0.01, 0.02])
    with pytest.raises(ValueError, match="stock_returns columns must match model tickers exactly."):
        model.fit(wrong_columns_stock, valid_market)


def test_single_index_fit_exact_analytical_values():
    """
    Validates OLS alpha and beta estimation against exact analytical calculations.
    Stock A: Return = 0.0 + 2.0 * Market (Beta = 2.0, Alpha = 0.0)
    Stock B: Return = 0.001 + 0.5 * Market (Beta = 0.5, Alpha = 0.001)
    """
    tickers = ["Stock_A", "Stock_B"]
    model = SingleIndexModel(tickers=tickers)
    
    market_returns = pd.Series([0.01, -0.01, 0.02, -0.02])
    stock_a = 0.0 + 2.0 * market_returns
    stock_b = 0.001 + 0.5 * market_returns
    stock_returns = pd.DataFrame({"Stock_A": stock_a, "Stock_B": stock_b})
    
    model.fit(stock_returns, market_returns)
    
    np.testing.assert_allclose(model.betas, [2.0, 0.5], atol=1e-8)
    np.testing.assert_allclose(model.alphas, [0.0, 0.001], atol=1e-8)
    np.testing.assert_allclose(model.residuals, np.zeros((4, 2)), atol=1e-8)


def test_single_index_simulate_output_shape_and_reproducibility():
    """Validates simulation output shape, DataFrame columns, and random seed reproducibility."""
    tickers = ["AAPL", "MSFT", "NVDA"]
    model = SingleIndexModel(tickers=tickers)
    
    np.random.seed(42)
    t_days = 100
    market_hist = pd.Series(np.random.normal(0.0005, 0.01, t_days))
    
    stock_data = {
        "AAPL": 0.0002 + 1.2 * market_hist + np.random.normal(0, 0.005, t_days),
        "MSFT": 0.0001 + 0.9 * market_hist + np.random.normal(0, 0.004, t_days),
        "NVDA": 0.0005 + 1.6 * market_hist + np.random.normal(0, 0.008, t_days),
    }
    stock_returns = pd.DataFrame(stock_data)
    
    model.fit(stock_returns, market_hist)
    
    sim_steps = 250
    sim_market = np.random.normal(0.0005, 0.01, sim_steps)
    
    # Run simulation with seed 100
    df_sim1 = model.simulate(sim_market, random_seed=100)
    df_sim2 = model.simulate(sim_market, random_seed=100)
    
    # Assert shape and columns
    assert df_sim1.shape == (250, 3)
    assert list(df_sim1.columns) == tickers
    
    # Assert reproducibility
    pd.testing.assert_frame_equal(df_sim1, df_sim2)