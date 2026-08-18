import numpy as np
import warnings
from scipy.stats import ks_2samp, anderson_ksamp, wasserstein_distance, gaussian_kde
from statsmodels.tsa.stattools import acf

class MetricsEvaluator:
    """
    Computes rigorous statistical distance metrics between historical and simulated distributions.
    """
    
    @staticmethod
    def calculate_ks(hist: np.ndarray, sim: np.ndarray) -> float:
        """
        Kolmogorov-Smirnov (KS) statistic.
        Intuition: Finds the widest vertical gap between the cumulative distributions. Smaller is better.
        """
        res = ks_2samp(hist, sim)
        return float(res.statistic) # type: ignore

    @staticmethod
    def calculate_ad(hist: np.ndarray, sim: np.ndarray) -> float:
        """
        Anderson-Darling (AD) statistic.
        Intuition: Like KS, but heavily penalizes differences in the extreme tails (market crashes).
        """
        # Suppress SciPy warnings if the p-value hits boundaries (common in large sample sizes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = anderson_ksamp([hist, sim])
        return float(res.statistic) # type: ignore

    @staticmethod
    def calculate_wasserstein(hist: np.ndarray, sim: np.ndarray) -> float:
        """
        Wasserstein-1 (Earth Mover's) distance.
        Intuition: The minimum mathematical "cost" to shovel the simulated distribution until it matches historical.
        """
        return wasserstein_distance(hist, sim)

    @staticmethod
    def calculate_hellinger(hist: np.ndarray, sim: np.ndarray) -> float:
        """
        Hellinger distance using Gaussian Kernel Density Estimation (KDE).
        Intuition: Strict measure of similarity between probability curves. 0 = identical, 1 = no overlap.
        """
        # 1. Create a shared evaluation grid spanning both datasets
        # We MUST add a buffer to capture the KDE Gaussian tails, otherwise the integral < 1.0
        buffer = 3.0 * max(np.std(hist), np.std(sim))
        min_val = min(np.min(hist), np.min(sim)) - buffer
        max_val = max(np.max(hist), np.max(sim)) + buffer
        grid = np.linspace(min_val, max_val, 2000)
        
        # 2. Fit smooth probability curves and evaluate them on the grid
        kde_hist = gaussian_kde(hist)(grid)
        kde_sim = gaussian_kde(sim)(grid)
        
        # 3. Calculate Bhattacharyya coefficient (approximate the integral)
        dx = grid[1] - grid[0]
        bc = np.sum(np.sqrt(kde_hist * kde_sim)) * dx
        
        # Guard against minor floating point errors exceeding 1.0
        return float(np.sqrt(1.0 - min(bc, 1.0)))

    @staticmethod
    def calculate_kurtosis_error(hist: np.ndarray, sim: np.ndarray) -> float:
        """
        Absolute error in excess kurtosis.
        Intuition: Kurtosis measures fat tails. This error tracks if the simulator captures crashes as often as history.
        """
        def get_kurt(data):
            var = np.var(data)
            return (np.mean((data - np.mean(data)) ** 4) / (var ** 2) - 3.0) if var > 0 else 0.0
            
        return abs(get_kurt(hist) - get_kurt(sim))

    @staticmethod
    def calculate_acf_mae(hist: np.ndarray, sim: np.ndarray, max_lag: int = 252) -> float:
        """
        Mean Absolute Error of the absolute Autocorrelation Function (ACF).
        Intuition: Measures Volatility Clustering. Compares the lag decay curves to see if volatile days follow volatile days.
        """
        acf_hist = np.asarray(acf(np.abs(hist), nlags=max_lag, fft=True))[1:]
        acf_sim = np.asarray(acf(np.abs(sim), nlags=max_lag, fft=True))[1:]
        return float(np.mean(np.abs(acf_hist - acf_sim)))
    
    @classmethod
    def evaluate_all(cls, hist: np.ndarray, sim: np.ndarray, max_lag: int = 252) -> dict:
        """Runs all metrics and returns a dictionary of results."""
        return {
            "KS_Stat": cls.calculate_ks(hist, sim),
            "AD_Stat": cls.calculate_ad(hist, sim),
            "Wasserstein": cls.calculate_wasserstein(hist, sim),
            "Hellinger": cls.calculate_hellinger(hist, sim),
            "Kurtosis_Error": cls.calculate_kurtosis_error(hist, sim),
            "ACF_MAE": cls.calculate_acf_mae(hist, sim, max_lag)
        }