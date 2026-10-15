import torch
from torch.utils.data import Dataset
import numpy as np

class PortfolioDataset(Dataset):
    def __init__(self, returns_df, weights_df, window_size=60, flatten=False):
        """
        Creates a dataset mapping historical return windows to expert portfolio weights.
        
        Args:
            returns_df (pd.DataFrame): DataFrame of daily stock returns.
            weights_df (pd.DataFrame): DataFrame of expert portfolio weights (the labels).
            window_size (int): Number of days to look back for the input features.
            flatten (bool): If True, flattens the 2D window into a 1D vector (for MLPs).
                            If False, preserves the (window_size, n_assets) geometry (for Transformers).
        """
        self.window_size = window_size
        self.flatten = flatten
        
        # Convert to numpy arrays upfront so slicing is instantaneous
        returns_arr = returns_df.to_numpy()
        weights_arr = weights_df.to_numpy()
        
        self.X = []
        self.Y = []
        
        for i in range(len(weights_arr)):
            # Extract the historical window of returns BEFORE this date
            window_returns = returns_arr[i : i + self.window_size]
            
            # Conditionally flatten the data based on the architecture we are feeding
            if self.flatten:
                feature_data = window_returns.flatten()
            else:
                feature_data = window_returns
                
            # The label is the expert's weight allocation for this date
            label_vector = weights_arr[i]
            
            self.X.append(feature_data)
            self.Y.append(label_vector)
            
        # Convert everything to PyTorch tensors
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class ConditionedPortfolioDataset(Dataset):
    def __init__(self, returns_df, weights_df, window_size=60, flatten=False):
        """
        Creates a dataset mapping historical return windows and incumbent allocations
        to expert portfolio weights for conditioned policies.
        
        Args:
            returns_df (pd.DataFrame): DataFrame of daily stock returns.
            weights_df (pd.DataFrame): DataFrame of expert portfolio weights (the labels).
            window_size (int): Number of days to look back for the input features.
            flatten (bool): If True, flattens the 2D window into a 1D vector (for MLPs).
                            If False, preserves the (window_size, n_assets) geometry (for Transformers).
        """
        self.window_size = window_size
        self.flatten = flatten
        
        returns_arr = returns_df.to_numpy()
        weights_arr = weights_df.to_numpy()
        action_dim = weights_arr.shape[1]
        
        self.X = []
        self.Y = []
        self.prev_weights = []
        
        for i in range(len(weights_arr)):
            window_returns = returns_arr[i : i + self.window_size]
            
            if self.flatten:
                feature_data = window_returns.flatten()
            else:
                feature_data = window_returns
                
            label_vector = weights_arr[i]
            
            if i == 0:
                prev_w = np.zeros(action_dim, dtype=np.float32)
                prev_w[-1] = 1.0  # 100% Cash for initial step
            else:
                prev_w = weights_arr[i - 1]
                
            self.X.append(feature_data)
            self.Y.append(label_vector)
            self.prev_weights.append(prev_w)
            
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.prev_weights = torch.tensor(np.array(self.prev_weights), dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.prev_weights[idx], self.Y[idx]