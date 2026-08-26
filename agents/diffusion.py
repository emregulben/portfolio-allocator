import torch
import torch.nn as nn
import math

class MLPDenoiser(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, t_dim=16):
        """
        Args:
            state_dim (int): The size of the flattened market returns.
            action_dim (int): The total number of assets in the portfolio (risky assets + cash).
            hidden_dim (int): Size of the hidden layers in the MLP.
            t_dim (int): Size of the sinusoidal time embedding.
        """
        super().__init__()
        self.t_dim = t_dim
        
        # The input to our MLP is the combination of the state, the noisy action, and the time embedding
        in_dim = state_dim + action_dim + t_dim
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Mish(), # Mish is a modern activation function that works great for MLPs
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim) # The output is the predicted noise (same shape as action_dim)
        )
        
    def time_embedding(self, t):
        """
        Converts a single integer timestep (e.g., t=45) into a robust vector of size t_dim.
        This helps the neural network understand the concept of "time" smoothly.
        """
        half_dim = self.t_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t * embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    def forward(self, state, action_noisy, t):
        """
        Predicts the noise added to the action.
        """
        # Embed the integer time into a vector
        t_embed = self.time_embedding(t)
        
        # Concatenate the market state, the noisy weights, and the time vector together
        x = torch.cat([state, action_noisy, t_embed], dim=-1)
        
        # Pass it through the MLP to predict the noise!
        return self.net(x)