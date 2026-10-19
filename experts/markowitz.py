import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MarkowitzExpert:
    def __init__(
        self,
        risk_aversion=1.0,
        rolling_window=60,
        max_weight=0.20,
        annual_risk_free_rate=0.0,
        transaction_cost=0.0,
    ):
        self.gamma = risk_aversion
        self.window = rolling_window
        self.max_weight = max_weight
        # Convert annual rate to daily rate assuming 252 trading days
        self.rf = annual_risk_free_rate / 252.0
        self.cost_c = transaction_cost

    def optimize(self, mu, cov, prev_weights=None):
        """
        Solves Mean-Variance optimization for a single time step, optionally penalized
        by linear transaction costs relative to incumbent weights (Boyd et al., 2017).

        Math: Maximize (w^T * mu) - (gamma / 2) * (w^T * cov * w) - cost_c * ||w - prev_weights||_1
        """
        n_assets = len(mu)
        w0 = np.array(prev_weights, dtype=np.float64) if prev_weights is not None else np.ones(n_assets) / n_assets

        def objective(w):
            port_return = w.T @ (mu - self.rf)
            port_risk = (self.gamma / 2.0) * (w.T @ cov @ w)
            cost = 0.0
            if self.cost_c > 0.0 and prev_weights is not None:
                cost = self.cost_c * np.sum(np.abs(w - prev_weights))
            return port_risk - port_return + cost

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        # The last asset in mu/cov is assumed to be Cash. 
        # Risky assets are capped at self.max_weight, but Cash is allowed to go up to 100% (1.0)
        bounds = tuple((0.0, self.max_weight) if i < n_assets - 1 else (0.0, 1.0) for i in range(n_assets))

        res = minimize(
            objective, 
            w0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 250, 'ftol': 1e-6}, 
        )

        if not res.success:
            # If optimizer stops near convergence on an L1 kink, verify feasibility
            if np.isclose(np.sum(res.x), 1.0, atol=1e-3) and np.all(res.x >= -1e-5):
                clipped = np.clip(res.x, 0.0, None)
                return clipped / np.sum(clipped)
            raise RuntimeError(f"Markowitz optimization failed: {res.message}")

        return res.x

    def generate_labels(self, returns_df, cost_aware=False):
        """
        Generates optimal portfolio weight targets over the time series.

        Args:
            returns_df (pd.DataFrame): Asset return matrix of shape (n_days, n_risky_assets).
            cost_aware (bool): If True, incorporates intra-period asset return drift and
                penalizes rebalancing turnover against incumbent holdings using self.cost_c.

        Returns:
            pd.DataFrame: Portfolio allocations of shape (n_days - window, n_risky_assets + 1).
        """
        returns_arr = returns_df.to_numpy()
        n_days, n_risky_assets = returns_arr.shape

        # Weights matrix includes an extra column for Cash
        weights = np.zeros((n_days, n_risky_assets + 1))

        # Track post-drift holdings for cost-aware rebalancing
        prev_w_drifted = None

        # Start calculating weights only after the initial historical lookback window has passed
        for t in range(self.window, n_days):
            # Grab the historical data window strictly BEFORE day t to prevent data leakage
            window_data = returns_arr[t - self.window : t]

            mu_risky = np.mean(window_data, axis=0)
            cov_risky = np.cov(window_data, rowvar=False)

            # Dynamically construct full mu and cov including Cash
            # Cash return is self.rf, and its covariance with everything (including itself) is 0
            mu_full = np.append(mu_risky, self.rf)
            cov_full = np.zeros((n_risky_assets + 1, n_risky_assets + 1))
            cov_full[:n_risky_assets, :n_risky_assets] = cov_risky

            if cost_aware and self.cost_c > 0.0:
                w_opt = self.optimize(mu_full, cov_full, prev_weights=prev_w_drifted)
                weights[t] = w_opt

                # Intra-period price drift to calculate end-of-period holdings prior to next rebalance
                r_t = np.append(returns_arr[t], self.rf)
                gross_ret = np.sum(w_opt * r_t)
                denom = max(1.0 + gross_ret, 1e-8)
                prev_w_drifted = (w_opt * (1.0 + r_t)) / denom
            else:
                weights[t] = self.optimize(mu_full, cov_full)

        columns = list(returns_df.columns) + ['Cash']

        # Return a DataFrame, dropping the initial warmup days where trading was impossible
        return pd.DataFrame(
            weights[self.window:],
            index=returns_df.index[self.window:],
            columns=columns,
        )