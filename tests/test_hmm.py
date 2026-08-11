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
    
    
def test_hmm_simulate_unfitted_raises_error():
    """
    Validates that simulating states on an unfitted model raises a ValueError.
    """
    model = HybridJumpsHMM(n_states=10)
    with pytest.raises(ValueError, match="Model must be fitted before simulating states."):
        model.simulate_states(n_steps=100, epsilon=0.01, lambd=10)


def test_hmm_simulate_normal_transitions():
    """
    Validates normal HMM transitions without jumps (epsilon = 0.0)
    using the alternating exact values model.
    """
    # 1. Fit the exact model where state 0 -> 1 and 1 -> 0 always
    returns = pd.Series([-0.02, 0.01, 0.0, 0.03, -0.01])
    model = HybridJumpsHMM(n_states=2)
    model.fit(returns)
    
    # 2. Simulate 10 steps with jump probability epsilon = 0.0
    np.random.seed(42)
    states = model.simulate_states(n_steps=10, epsilon=0.0, lambd=0)
    
    # 3. Check shape and correct values
    assert states.shape == (10,)
    
    # Verify that the sequence strictly alternates: e.g. [0, 1, 0, 1, 0, 1...]
    # np.diff should be either +1 or -1 on every step
    assert np.all(np.abs(np.diff(states)) == 1)


def test_hmm_simulate_always_jump():
    """
    Validates that when epsilon = 1.0, the simulator only draws states
    from the designated tail states.
    """
    # 1. Fit HMM on dummy data
    dummy_returns = pd.Series(np.random.normal(0, 0.01, 100))
    model = HybridJumpsHMM(n_states=10)
    model.fit(dummy_returns)
    
    # 2. Simulate 200 steps with constant jump probability epsilon = 1.0
    # With n_tail = 2: s_bottom = [0, 1], s_top = [8, 9]
    n_tail = 2
    states = model.simulate_states(
        n_steps=200, 
        epsilon=1.0, 
        lambd=50, 
        n_tail=n_tail
    )
    
    # 3. Verify that every state from Day 2 onwards is in [0, 1, 8, 9]
    allowed_states = {0, 1, 8, 9}
    for state in states[1:]:
        assert state in allowed_states
    
    
def test_hmm_decode_unfitted_raises_error():
    """
    Validates that decoding states on an unfitted model raises a ValueError.
    """
    model = HybridJumpsHMM(n_states=10)
    dummy_states = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="Model must be fitted before decoding states."):
        model.decode_states(dummy_states)


def test_hmm_decode_exact_values():
    """
    Validates that decoded returns match the state-conditional mean,
    standard deviation, and Student-t random noise.
    """
    # 1. Fit the exact 2-state model
    returns = pd.Series([-0.02, 0.01, 0.0, 0.03, -0.01])
    model = HybridJumpsHMM(n_states=2)
    model.fit(returns)
    
    # 2. Define target states to decode (State 0 and State 1)
    states = np.array([0, 1])
    
    # 3. Generate expected noise Z using the same seed
    np.random.seed(42)
    expected_z = np.random.standard_t(df=5, size=2)
    
    # 4. Decode using the same seed
    np.random.seed(42)
    decoded_returns = model.decode_states(states)
    
    # 5. Calculate expected returns: Mean + Std * Z
    expected_returns = model.state_means[states] + model.state_stds[states] * expected_z
    
    assert np.allclose(decoded_returns, expected_returns)