import numpy as np
import pandas as pd
from scipy.stats import laplace
from statsmodels.tsa.stattools import acf
from typing import Optional

class HybridJumpsHMM:
    """
    Hybrid Hidden Markov Model with Jump-Diffusion (HMM-WJ) for returns simulation.
    """
    def __init__(self, n_states: int = 100):
        self.n_states = n_states
        
        self.boundaries = np.empty(0)
        self.T = np.empty((0, 0))
        self.T_cumsum = np.empty((0, 0))
        self.pi_bar = np.empty(0)
        self.state_means = np.empty(0)
        self.state_stds = np.empty(0)

    def fit(self, returns: pd.Series):
        """
        Fits the HMM parameters to historical daily returns using Laplace partitioning.
        """
        ret_data = returns.dropna().to_numpy()
        n_samples = len(ret_data)
        
        global_mean = np.mean(ret_data)
        global_std = np.std(ret_data)
        
        # 1. Fit a Laplace distribution
        loc_lap, scale_lap = laplace.fit(ret_data)
        
        # Numerical guard: prevent scale from being exactly 0 to avoid SciPy warnings
        if scale_lap < 1e-8:
            scale_lap = 1e-8
        
        # 2. Define the N+1 quantile boundaries (Algorithm 2, Lines 3-4)
        probs = np.linspace(0.001, 0.999, self.n_states + 1)
        self.boundaries = laplace.ppf(probs, loc=loc_lap, scale=scale_lap)
        
        # 3. Assign each return to a state from 1 to N (Algorithm 2, Line 5)
        inner_bins = self.boundaries[1:-1]
        assigned_states = np.digitize(ret_data, inner_bins, right=True) + 1
        
        # 4. Estimate the Transition Matrix (Algorithm 2, Lines 6-13)
        counts = np.zeros((self.n_states, self.n_states))
        for t in range(n_samples - 1):
            curr_state = assigned_states[t] - 1
            next_state = assigned_states[t + 1] - 1
            counts[curr_state, next_state] += 1
            
        self.T = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            row_sum = np.sum(counts[i, :])
            if row_sum > 0:
                self.T[i, :] = counts[i, :] / row_sum
            else:
                self.T[i, i] = 1.0
                
        self.T_cumsum = np.cumsum(self.T, axis=1)
        
        # 5. Compute Stationary Distribution (pi_bar) by powering T
        T_pow = np.linalg.matrix_power(self.T, 50)
        self.pi_bar = np.mean(T_pow, axis=0)
        self.pi_bar = self.pi_bar / np.sum(self.pi_bar)
        
        # 6. Estimate state-conditional means (mu_k) and standard deviations (sigma_k)
        self.state_means = np.zeros(self.n_states)
        self.state_stds = np.zeros(self.n_states)
        
        for k in range(1, self.n_states + 1):
            state_mask = (assigned_states == k)
            state_returns = ret_data[state_mask]
            
            if len(state_returns) > 0:
                self.state_means[k - 1] = np.mean(state_returns)
                self.state_stds[k - 1] = np.std(state_returns)
            else:
                self.state_means[k - 1] = global_mean
                self.state_stds[k - 1] = global_std
        
    def simulate_states(
        self,
        n_steps: int,
        epsilon: float,
        lambd: float,
        n_tail: int = 5,
        negative_jump_prob: float = 0.52
    ) -> np.ndarray:
        """
        Simulates a hidden state sequence of length n_steps using the Poisson jump mechanism.
        """
        if len(self.T_cumsum) == 0 or len(self.pi_bar) == 0:
            raise ValueError("Model must be fitted before simulating states.")
            
        states = np.zeros(n_steps, dtype=int)
        
        # Define 0-indexed tail states
        s_bottom = np.arange(n_tail)
        s_top = np.arange(self.n_states - n_tail, self.n_states)
        
        # Sample initial state from stationary distribution
        states[0] = np.searchsorted(np.cumsum(self.pi_bar), np.random.rand())
        
        counter = 1
        while counter < n_steps:
            if np.random.rand() < epsilon:
                # Sample jump duration from Poisson distribution
                k_jump = np.random.poisson(lambd)
                k_jump = min(k_jump, n_steps - counter)
                
                if k_jump > 0:
                    # Decide if each jump step is bottom or top tail
                    w = np.random.rand(k_jump)
                    is_bottom = w < negative_jump_prob
                    
                    # Uniformly sample from the selected tail states
                    states[counter : counter + k_jump] = np.where(
                        is_bottom,
                        np.random.choice(s_bottom, size=k_jump),
                        np.random.choice(s_top, size=k_jump)
                    )
                    counter += k_jump
            else:
                # Normal transition using cumulative probabilities
                prev_state = states[counter - 1]
                states[counter] = np.searchsorted(self.T_cumsum[prev_state], np.random.rand())
                counter += 1
                
        return states
    
    def decode_states(self, states: np.ndarray) -> np.ndarray:
        """
        Decodes a sequence of states into continuous daily returns using Student-t emissions.
        """
        if len(self.state_means) == 0 or len(self.state_stds) == 0:
            raise ValueError("Model must be fitted before decoding states.")
            
        # Draw standard Student-t random variables (df = 5) for each day
        z = np.random.standard_t(df=5, size=len(states))
        
        # Vectorized scaling and shifting using state-conditional parameters
        simulated_returns = self.state_means[states] + self.state_stds[states] * z
        return simulated_returns
    
    def grid_search(
        self,
        returns: pd.Series,
        epsilon_grid: Optional[list[float]] = None,
        lambda_grid: Optional[list[float]] = None,
        max_lag: int = 252,
        n_paths: int = 200,
        w_K: float = 0.20
    ) -> tuple[float, float]:
        """
        Performs multi-objective grid search over jump parameters (epsilon, lambda)
        to minimize discrepancy in absolute ACF and kurtosis (Algorithm 4).
        """
        if len(self.boundaries) == 0:
            raise ValueError("Model must be fitted before running grid search.")
            
        ret_data = returns.dropna().to_numpy()
        n_steps = len(ret_data)
        
        # Default search grids from the paper
        if epsilon_grid is None:
            epsilon_grid = [1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2, 2.5e-2]
        if lambda_grid is None:
            lambda_grid = [10, 25, 40, 55, 70, 85, 100, 130, 160]
            
        # 1. Compute observed statistics (ACF of absolute returns & Kurtosis)
        acf_obs = acf(np.abs(ret_data), nlags=max_lag, fft=True)[1:]
        var_obs = np.var(ret_data)
        k_obs = (np.mean((ret_data - np.mean(ret_data)) ** 4) / (var_obs ** 2) - 3.0) if var_obs > 0 else 0.0
        
        min_error = float("inf")
        best_eps = epsilon_grid[0]
        best_lambd = lambda_grid[0]
        
        # 2. Sweep over grid combinations
        for eps in epsilon_grid:
            for lambd in lambda_grid:
                acf_sum = np.zeros(max_lag)
                k_sum = 0.0
                
                # Run N simulated paths per grid point
                for _ in range(n_paths):
                    sim_states = self.simulate_states(n_steps=n_steps, epsilon=eps, lambd=lambd)
                    sim_returns = self.decode_states(sim_states)
                    
                    # Accumulate ACF of simulated absolute returns
                    acf_sim_path = acf(np.abs(sim_returns), nlags=max_lag, fft=True)[1:]
                    acf_sum += acf_sim_path
                    
                    # Accumulate simulated kurtosis
                    var_sim = np.var(sim_returns)
                    k_sim = (np.mean((sim_returns - np.mean(sim_returns)) ** 4) / (var_sim ** 2) - 3.0) if var_sim > 0 else 0.0
                    k_sum += k_sim
                    
                # Ensemble averages
                acf_sim_avg = acf_sum / n_paths
                k_sim_avg = k_sum / n_paths
                
                # 3. Calculate multi-objective error J (Equation 3)
                error = np.sum((acf_obs - acf_sim_avg) ** 2) + w_K * ((k_obs - k_sim_avg) ** 2)
                
                if error < min_error:
                    min_error = error
                    best_eps = eps
                    best_lambd = lambd
                    
        # Save best parameters to the instance
        self.epsilon = best_eps
        self.lambd = best_lambd
        
        return best_eps, best_lambd
    
    def simulate_states_inertial(self, n_steps: int, inertia: float = 0.0) -> np.ndarray:
        """
        Simulates a hidden state sequence using the Inertial Markov Chain algorithm.
        T_inertial = (1 - inertia) * T + inertia * I
        """
        if len(self.T_cumsum) == 0 or len(self.pi_bar) == 0:
            raise ValueError("Model must be fitted before simulating states.")
            
        states = np.zeros(n_steps, dtype=int)
        
        # Blend empirical transitions with Identity matrix
        I = np.eye(self.n_states)
        T_inertial = (1.0 - inertia) * self.T + inertia * I
        
        # Sample initial state from stationary distribution
        states[0] = np.searchsorted(np.cumsum(self.pi_bar), np.random.rand())
        
        for t in range(1, n_steps):
            # Transition using the inertial matrix
            states[t] = np.random.choice(self.n_states, p=T_inertial[states[t-1]])
            
        return states
    
    def grid_search_inertial(
        self,
        returns: pd.Series,
        inertia_grid: Optional[list[float]] = None,
        max_lag: int = 252,
        n_paths: int = 50,
        w_K: float = 0.20
    ) -> float:
        """
        Grid search to find the optimal inertia parameter that minimizes discrepancy 
        in absolute ACF and kurtosis.
        """
        if len(self.boundaries) == 0:
            raise ValueError("Model must be fitted before running grid search.")
            
        ret_data = returns.dropna().to_numpy()
        n_steps = len(ret_data)
        
        # Default search grid for Inertia (0.0 means pure empirical T, 0.95 means highly sticky)
        if inertia_grid is None:
            inertia_grid = list(np.linspace(0.0, 0.95, 20))
            
        # Compute observed statistics
        acf_obs = acf(np.abs(ret_data), nlags=max_lag, fft=True)[1:]
        var_obs = np.var(ret_data)
        k_obs = (np.mean((ret_data - np.mean(ret_data)) ** 4) / (var_obs ** 2) - 3.0) if var_obs > 0 else 0.0
        
        min_error = float("inf")
        best_inertia = inertia_grid[0]
        
        # Sweep over the 1D inertia grid
        for rho in inertia_grid:
            acf_sum = np.zeros(max_lag)
            k_sum = 0.0
            
            # Run N simulated paths per grid point
            for _ in range(n_paths):
                sim_states = self.simulate_states_inertial(n_steps=n_steps, inertia=rho)
                sim_returns = self.decode_states(sim_states)
                
                # Accumulate simulated ACF
                acf_sim_path = acf(np.abs(sim_returns), nlags=max_lag, fft=True)[1:]
                acf_sum += acf_sim_path
                
                # Accumulate simulated kurtosis
                var_sim = np.var(sim_returns)
                k_sim = (np.mean((sim_returns - np.mean(sim_returns)) ** 4) / (var_sim ** 2) - 3.0) if var_sim > 0 else 0.0
                k_sum += k_sim
                
            # Ensemble averages
            acf_sim_avg = acf_sum / n_paths
            k_sim_avg = k_sum / n_paths
            
            # Calculate multi-objective error J
            error = np.sum((acf_obs - acf_sim_avg) ** 2) + w_K * ((k_obs - k_sim_avg) ** 2)
            
            if error < min_error:
                min_error = error
                best_inertia = rho
                
        # Save best parameter to the instance
        self.inertia = best_inertia
        
        return float(best_inertia)