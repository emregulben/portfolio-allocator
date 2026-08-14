import numpy as np
import pytest
from experts.markowitz import MarkowitzExpert

def test_markowitz_optimization_failure_raises_error():
    """
    Validates that providing mathematically impossible constraints 
    forces the optimizer to fail and raises our explicit RuntimeError.
    """
    # 10 assets with a 5% cap. It is impossible to sum to 100%.
    expert = MarkowitzExpert(risk_aversion=1.0, max_weight=0.05)
    
    mu = np.zeros(10)
    cov = np.eye(10)
    
    with pytest.raises(RuntimeError, match="Markowitz optimization failed"):
        expert.optimize(mu, cov)

def test_markowitz_constraints():
    """
    Tests that the optimizer perfectly obeys all constraints: 
    sum = 1.0, min >= 0.0, max <= max_weight.
    """
    expert = MarkowitzExpert(risk_aversion=1.0, max_weight=0.20)
    
    # Fake data for 10 assets
    mu = np.random.normal(0.0002, 0.01, 10)
    cov = np.diag(np.random.uniform(0.0001, 0.0005, 10))
    
    weights = expert.optimize(mu, cov)
    
    # Assert constraints (with tiny 1e-8 tolerance for floating point math)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= -1e-8)  
    assert np.all(weights <= 0.20 + 1e-8)

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