from typing import Optional
import numpy as np
import pandas as pd


class SingleIndexModel:
    """
    Multi-asset market generator using Sharpe's Single-Index Model (SIM).
    Fits OLS regressions for N stocks against a market benchmark (SPY):
        Return_{i,t} = alpha_i + beta_i * Return_{Market,t} + residual_{i,t}
    """

    def __init__(self, tickers: list[str]) -> None:
        if not tickers:
            raise ValueError("Tickers list cannot be empty.")

        self.tickers: list[str] = tickers
        self.alphas: np.ndarray = np.empty(0, dtype=np.float64)
        self.betas: np.ndarray = np.empty(0, dtype=np.float64)
        self.residuals: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.is_fitted: bool = False

    def fit(self, stock_returns: pd.DataFrame, market_returns: pd.Series) -> None:
        """
        Fits OLS regressions for all stocks against market_returns.
        
        Args:
            stock_returns: Daily returns of N stocks (shape: T x N).
            market_returns: Daily returns of market benchmark (shape: T,).
        """
        if stock_returns.empty or market_returns.empty:
            raise ValueError("Input data cannot be empty.")

        if len(stock_returns) != len(market_returns):
            raise ValueError("stock_returns and market_returns must have identical length.")

        if list(stock_returns.columns) != self.tickers:
            raise ValueError("stock_returns columns must match model tickers exactly.")

        y = stock_returns.to_numpy()  # shape: (T, N)
        x = market_returns.to_numpy()  # shape: (T,)

        var_x = np.var(x, ddof=1)
        if var_x < 1e-12:
            raise ValueError("Market returns variance is zero or negligible.")

        mean_x = np.mean(x)
        mean_y = np.mean(y, axis=0)  # shape: (N,)

        # Sample covariance between each stock and market
        cov_xy = np.mean((y - mean_y) * (x[:, np.newaxis] - mean_x), axis=0) * (len(x) / (len(x) - 1))

        # OLS parameters
        self.betas = cov_xy / var_x
        self.alphas = mean_y - self.betas * mean_x

        # Empirical residuals: residual_{i,t} = y_{i,t} - (alpha_i + beta_i * x_t)
        fitted_values = self.alphas + np.outer(x, self.betas)
        self.residuals = y - fitted_values  # shape: (T, N)

        self.is_fitted = True

    def simulate(self, market_sim_returns: np.ndarray, random_seed: Optional[int] = None) -> pd.DataFrame:
        """
        Simulates multi-asset returns for all N stocks given a simulated market return path.
        
        Args:
            market_sim_returns: 1D array of simulated market returns (shape: n_steps,).
            random_seed: Optional random seed for reproducible residual sampling.
            
        Returns:
            pd.DataFrame: Simulated returns matrix of N stocks (shape: n_steps x N).
        """
        if not self.is_fitted:
            raise ValueError("SingleIndexModel must be fitted before simulating.")

        if random_seed is not None:
            np.random.seed(random_seed)

        n_steps = len(market_sim_returns)
        n_hist, n_assets = self.residuals.shape

        # Resample empirical residual rows with replacement
        resample_idx = np.random.choice(n_hist, size=n_steps, replace=True)
        sampled_residuals = self.residuals[resample_idx, :]  # shape: (n_steps, N)

        # Calculate stock returns: Return_{i,t} = alpha_i + beta_i * Market_t + residual_{i,t}
        sim_returns_matrix = self.alphas + np.outer(market_sim_returns, self.betas) + sampled_residuals

        return pd.DataFrame(sim_returns_matrix, columns=self.tickers)