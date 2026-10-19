import numpy as np
import pandas as pd
import pytest
from experts.markowitz import MarkowitzExpert

def test_markowitz_optimization_failure_raises_error():
    """
    Validates that providing mathematically corrupted data forces the 
    optimizer to fail and raises our explicit RuntimeError.
    """
    expert = MarkowitzExpert(risk_aversion=1.0, max_weight=0.20)
    
    # Injecting a NaN (Not a Number) to corrupt the objective function
    mu = np.array([np.nan, 0.05, 0.05])
    cov = np.eye(3)
    
    with pytest.raises(RuntimeError, match="Markowitz optimization failed"):
        expert.optimize(mu, cov)

def test_markowitz_constraints():
    """
    Tests that the optimizer perfectly obeys all constraints: 
    sum = 1.0, min >= 0.0, risky max <= max_weight, cash max <= 1.0.
    """
    expert = MarkowitzExpert(risk_aversion=1.0, max_weight=0.20)
    
    # Fake data for 10 assets
    mu = np.random.normal(0.0002, 0.01, 10)
    cov = np.diag(np.random.uniform(0.0001, 0.0005, 10))
    
    weights = expert.optimize(mu, cov)
    
    # Assert constraints (with tiny 1e-8 tolerance for floating point math)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= -1e-8)  
    
    # Risky assets (everything except the last one) must be <= 0.20
    assert np.all(weights[:-1] <= 0.20 + 1e-8)
    
    # Cash asset (the last one) must be <= 1.0
    assert weights[-1] <= 1.0 + 1e-8

def test_markowitz_symmetry():
    """
    Tests that if two assets have the EXACT same returns and risk, 
    the optimizer distributes weight perfectly equally between them.
    """
    # Remove the 0.20 cap just for this specific symmetry test
    expert = MarkowitzExpert(risk_aversion=1.0, max_weight=1.0) 
    
    mu = np.array([0.05, 0.05])
    cov = np.array([[0.01, 0.00], 
                    [0.00, 0.01]])
    
    weights = expert.optimize(mu, cov)
    
    # Both identical assets should get exactly 50% of the portfolio
    assert np.isclose(weights[0], 0.50)
    assert np.isclose(weights[1], 0.50)

def test_generate_labels_adds_cash():
    """
    Tests that the generate_labels function dynamically adds the Cash column 
    and outputs the correct matrix dimensions.
    """
    expert = MarkowitzExpert(rolling_window=5)
    
    # 10 days of synthetic data for 3 risky assets
    data = np.random.normal(0.001, 0.01, (10, 3))
    df = pd.DataFrame(data, columns=['A', 'B', 'C'])
    
    labels_df = expert.generate_labels(df)
    
    # Should have dropped the first 5 warmup days, leaving 5 days of trading
    assert len(labels_df) == 5
    
    # Should have 3 risky + 1 Cash = 4 columns
    assert len(labels_df.columns) == 4
    assert labels_df.columns[-1] == 'Cash'
    
    # Sum of weights on any day must equal exactly 1.0
    assert np.allclose(labels_df.sum(axis=1), 1.0)

def test_markowitz_cash_retreat():
    """
    Tests that the optimizer successfully retreats to the Cash asset (the last asset) 
    during a severe market crash, bypassing the max_weight cap for Cash.
    """
    expert = MarkowitzExpert(risk_aversion=1.0, max_weight=0.20, annual_risk_free_rate=0.0)
    
    # Simulate a severe crash: 5 risky assets all with terrible negative expected returns
    mu_risky = np.array([-0.10, -0.08, -0.15, -0.09, -0.12])
    cov_risky = np.eye(5) * 0.05
    
    # Append the Cash asset (return = expert.rf, variance = 0.0)
    mu_full = np.append(mu_risky, expert.rf)
    cov_full = np.zeros((6, 6))
    cov_full[:5, :5] = cov_risky
    
    weights = expert.optimize(mu_full, cov_full)
    
    # All risky assets should be dropped to 0.0, and Cash (index 5) should be exactly 1.0
    assert np.allclose(weights[:5], 0.0, atol=1e-6)
    assert np.isclose(weights[5], 1.0, atol=1e-6)
    assert np.isclose(np.sum(weights), 1.0)

def test_transaction_cost_turnover_reduction():
    """
    Verifies that incorporating transaction cost penalties in Markowitz optimization
    strictly reduces daily portfolio turnover relative to friction-agnostic Markowitz.
    """
    np.random.seed(42)
    n_days = 80
    n_assets = 5
    returns = np.random.normal(0.0005, 0.015, (n_days, n_assets))
    df = pd.DataFrame(returns, columns=[f"Asset_{i}" for i in range(n_assets)])

    expert_unpenalized = MarkowitzExpert(rolling_window=20, max_weight=0.20, transaction_cost=0.0)
    expert_penalized = MarkowitzExpert(rolling_window=20, max_weight=0.20, transaction_cost=0.002)

    weights_unpenalized = expert_unpenalized.generate_labels(df, cost_aware=False).to_numpy()
    weights_penalized = expert_penalized.generate_labels(df, cost_aware=True).to_numpy()

    # Calculate total rebalancing turnover across active periods
    turnover_unpenalized = np.sum(np.abs(np.diff(weights_unpenalized, axis=0)))
    turnover_penalized = np.sum(np.abs(np.diff(weights_penalized, axis=0)))

    assert turnover_penalized < turnover_unpenalized, (
        f"Cost-aware Markowitz turnover ({turnover_penalized:.4f}) must be lower than "
        f"friction-agnostic Markowitz turnover ({turnover_unpenalized:.4f})."
    )

def test_cost_aware_markowitz_constraints():
    """
    Verifies that cost-aware Markowitz allocations strictly obey budget,
    non-negativity, and single-stock bounds at every time step.
    """
    np.random.seed(101)
    n_days = 50
    n_assets = 4
    returns = np.random.normal(0.0003, 0.01, (n_days, n_assets))
    df = pd.DataFrame(returns, columns=[f"Stock_{i}" for i in range(n_assets)])

    expert = MarkowitzExpert(rolling_window=15, max_weight=0.20, transaction_cost=0.001)
    labels_df = expert.generate_labels(df, cost_aware=True)
    weights = labels_df.to_numpy()

    # Simplex constraint: sum = 1.0
    for w in weights:
        assert np.isclose(np.sum(w), 1.0, atol=1e-5)
        assert np.all(w >= -1e-6)
        # Risky asset cap (all columns except the last Cash column)
        assert np.all(w[:-1] <= 0.20 + 1e-5)
        # Cash cap
        assert w[-1] <= 1.0 + 1e-5

def test_cost_zero_equivalence():
    """
    Verifies that with zero transaction cost (cost_c = 0), cost-aware mode
    yields allocations consistent with classical Markowitz.
    """
    np.random.seed(7)
    n_days = 30
    n_assets = 3
    returns = np.random.normal(0.0004, 0.01, (n_days, n_assets))
    df = pd.DataFrame(returns, columns=['X', 'Y', 'Z'])

    expert = MarkowitzExpert(rolling_window=10, max_weight=0.20, transaction_cost=0.0)

    w_classical = expert.generate_labels(df, cost_aware=False).to_numpy()
    w_cost_mode_zero = expert.generate_labels(df, cost_aware=True).to_numpy()

    assert np.allclose(w_classical, w_cost_mode_zero, atol=1e-5), (
        "With cost_c = 0.0, cost_aware mode must match classical Markowitz."
    )