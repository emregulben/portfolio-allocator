import numpy as np
import pandas as pd
from scipy.optimize import minimize

class MarkowitzExpert:
    def __init__(self, risk_aversion=1.0, rolling_window=60, max_weight=0.20, annual_risk_free_rate=0.0):
        self.gamma = risk_aversion
        self.window = rolling_window
        self.max_weight = max_weight
        # Convert annual rate to daily rate assuming 252 trading days
        self.rf = annual_risk_free_rate / 252.0

    def optimize(self, mu, cov):
        """
        Solves Mean-Variance optimization for a single time step.
        Math: Maximize (w^T * mu) - (gamma / 2) * (w^T * cov * w)
        """
        n_assets = len(mu)
        w0 = np.ones(n_assets) / n_assets
        
        def objective(w):
            port_return = w.T @ (mu - self.rf)
            port_risk = (self.gamma / 2) * (w.T @ cov @ w)
            return port_risk - port_return
            
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, self.max_weight) for _ in range(n_assets))
        
        res = minimize(
            objective, 
            w0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        
        if not res.success:
            raise RuntimeError(f"Markowitz optimization failed: {res.message}")
            
        return res.x
    
    def generate_labels(self, returns_df):
        returns_arr = returns_df.to_numpy()
        n_days, n_assets = returns_arr.shape
        weights = np.zeros((n_days, n_assets))
        
        # Start calculating weights only after the initial historical lookback window has passed
        for t in range(self.window, n_days):
            # Grab the historical data window strictly BEFORE day t to prevent data leakage
            window_data = returns_arr[t - self.window : t]
            
            mu = np.mean(window_data, axis=0)
            cov = np.cov(window_data, rowvar=False)
            
            weights[t] = self.optimize(mu, cov)
            
        # Return a DataFrame, dropping the initial warmup days where trading was impossible
        return pd.DataFrame(
            weights[self.window:], 
            index=returns_df.index[self.window:], 
            columns=returns_df.columns
        )