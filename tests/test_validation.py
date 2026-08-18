import numpy as np
import pytest
from simulator.validation import MetricsEvaluator

@pytest.fixture
def sample_data():
    """Provides identical and divergent distributions for testing."""
    np.random.seed(42)
    hist = np.random.normal(0, 1, 1000)
    sim_identical = hist.copy()
    sim_different = np.random.normal(2, 0.5, 1000) # Shifted mean, lower variance
    return hist, sim_identical, sim_different

def test_calculate_ks(sample_data):
    hist, sim_identical, sim_different = sample_data
    
    # Identical distributions should have KS statistic near 0
    ks_identical = MetricsEvaluator.calculate_ks(hist, sim_identical)
    assert ks_identical == 0.0
    
    # Different distributions should have a positive KS statistic
    ks_diff = MetricsEvaluator.calculate_ks(hist, sim_different)
    assert ks_diff > 0.5

def test_calculate_ad(sample_data):
    hist, sim_identical, sim_different = sample_data
    
    # AD statistic for identical data might be negative (due to SciPy normalization)
    ad_identical = MetricsEvaluator.calculate_ad(hist, sim_identical)
    assert isinstance(ad_identical, float)
    
    ad_diff = MetricsEvaluator.calculate_ad(hist, sim_different)
    assert ad_diff > ad_identical

def test_calculate_wasserstein(sample_data):
    hist, sim_identical, sim_different = sample_data
    
    wass_identical = MetricsEvaluator.calculate_wasserstein(hist, sim_identical)
    assert wass_identical == 0.0
    
    wass_diff = MetricsEvaluator.calculate_wasserstein(hist, sim_different)
    assert wass_diff > 0.0

def test_calculate_hellinger(sample_data):
    hist, sim_identical, sim_different = sample_data
    
    hell_identical = MetricsEvaluator.calculate_hellinger(hist, sim_identical)
    # Hellinger distance for identical data should be ~0 (allow tiny float precision gap)
    assert np.isclose(hell_identical, 0.0, atol=1e-5)
    
    hell_diff = MetricsEvaluator.calculate_hellinger(hist, sim_different)
    assert hell_diff > 0.0

def test_calculate_kurtosis_error(sample_data):
    hist, sim_identical, _ = sample_data
    
    kurt_err = MetricsEvaluator.calculate_kurtosis_error(hist, sim_identical)
    assert kurt_err == 0.0
    
    # Test against fat-tailed data (Student-t)
    fat_tails = np.random.standard_t(df=3, size=1000)
    kurt_err_diff = MetricsEvaluator.calculate_kurtosis_error(hist, fat_tails)
    assert kurt_err_diff > 0.0

def test_calculate_acf_mae(sample_data):
    hist, sim_identical, _ = sample_data
    
    acf_mae = MetricsEvaluator.calculate_acf_mae(hist, sim_identical, max_lag=10)
    assert acf_mae == 0.0
    
    # Create artificially autocorrelated data
    autocorr = np.zeros(1000)
    for i in range(1, 1000):
        autocorr[i] = 0.8 * autocorr[i-1] + np.random.normal()
        
    acf_mae_diff = MetricsEvaluator.calculate_acf_mae(hist, autocorr, max_lag=10)
    assert acf_mae_diff > 0.0

def test_evaluate_all(sample_data):
    hist, sim_identical, _ = sample_data
    
    metrics = MetricsEvaluator.evaluate_all(hist, sim_identical, max_lag=10)
    
    assert isinstance(metrics, dict)
    expected_keys = ["KS_Stat", "AD_Stat", "Wasserstein", "Hellinger", "Kurtosis_Error", "ACF_MAE"]
    for key in expected_keys:
        assert key in metrics