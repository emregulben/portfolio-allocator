import numpy as np
import pandas as pd
import pytest
from simulator.hmm import HybridJumpsHMM

def test_hmm_fit_basic():
    """
    Validates that the HMM fit method correctly populates all state variables
    with the expected shapes, and that matrices satisfy probability rules.
    """
    # 1. Create a dummy return series (1000 days of random returns)
    np.random.seed(42)
    dummy_returns = pd.Series(np.random.normal(loc=0.0002, scale=0.01, size=1000))
    
    # 2. Fit HMM with 10 states
    n_states = 10
    model = HybridJumpsHMM(n_states=n_states)
    model.fit(dummy_returns)
    
    # 3. Assert shapes
    assert model.boundaries.shape == (n_states + 1,)
    assert model.T.shape == (n_states, n_states)
    assert model.T_cumsum.shape == (n_states, n_states)
    assert model.pi_bar.shape == (n_states,)
    assert model.state_means.shape == (n_states,)
    assert model.state_stds.shape == (n_states,)
    
    # 4. Assert math rules
    # Boundaries must be strictly sorted
    assert np.all(np.diff(model.boundaries) > 0)
    
    # Rows of Transition Matrix must sum to 1.0 (within numerical tolerance)
    assert np.allclose(np.sum(model.T, axis=1), 1.0)
    
    # Stationary distribution must sum to 1.0
    assert pytest.approx(np.sum(model.pi_bar)) == 1.0
    

def test_hmm_fit_unvisited_states():
    """
    Validates the fallback mechanism when some states are never visited.
    """
    # Create returns with only a single constant value. 
    # Because there is no variation, they will all map to a single state.
    # The other 9 states will be completely unvisited.
    constant_returns = pd.Series(np.ones(100) * 0.005)
    
    n_states = 10
    model = HybridJumpsHMM(n_states=n_states)
    model.fit(constant_returns)
    
    # The global values should be used for unvisited states
    # Mean should be 0.005 and Std should be 0.0 (since all values are 0.005)
    assert pytest.approx(model.state_means[0]) == 0.005
    assert pytest.approx(model.state_stds[0]) == 0.0
    
    # Ensure transition matrix rows still sum to 1.0 even with unvisited states
    assert np.allclose(np.sum(model.T, axis=1), 1.0)
    

def test_hmm_fit_exact_values():
    """
    Validates the fit calculations using a simple, manually calculated
    set of inputs to ensure exact mathematical accuracy.
    """
    # 1. Define manually designed returns
    # Median is exactly 0.0, which acts as the boundary Q_1 for N=2
    returns = pd.Series([-0.02, 0.01, 0.0, 0.03, -0.01])
    
    # 2. Fit HMM with 2 states
    model = HybridJumpsHMM(n_states=2)
    model.fit(returns)

    # 3. Assert boundaries
    expected_boundaries = np.array([
        0.0 + 0.014 * np.log(0.002),  # Q_0
        0.0,                          # Q_1
        0.0 - 0.014 * np.log(0.002)   # Q_2
    ])
    assert np.allclose(model.boundaries, expected_boundaries)

    # 4. Assert transition matrix and cumulative transitions
    expected_T = np.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ])
    expected_T_cumsum = np.array([
        [0.0, 1.0],
        [1.0, 1.0]
    ])
    assert np.allclose(model.T, expected_T)
    assert np.allclose(model.T_cumsum, expected_T_cumsum)
    
    # 5. Assert state-conditional means and standard deviations
    expected_means = np.array([-0.01, 0.02])
    expected_stds = np.array([np.sqrt(0.0002 / 3), 0.01])
    assert np.allclose(model.state_means, expected_means)
    assert np.allclose(model.state_stds, expected_stds)
    
    # 6. Assert stationary distribution (spending 50% time in each state)
    expected_pi_bar = np.array([0.5, 0.5])
    assert np.allclose(model.pi_bar, expected_pi_bar)