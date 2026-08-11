import numpy as np
import pandas as pd
from scipy.stats import laplace

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