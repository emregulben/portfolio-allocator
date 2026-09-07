import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLPDenoiser(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, t_dim=16, use_layernorm=False):
        """
        Args:
            state_dim (int): The size of the flattened market returns.
            action_dim (int): The total number of assets in the portfolio (risky assets + cash).
            hidden_dim (int): Size of the hidden layers in the MLP.
            t_dim (int): Size of the sinusoidal time embedding.
            use_layernorm (bool): Stabilizes gradients for large-dimensional inputs.
        """
        super().__init__()
        self.t_dim = t_dim
        
        # The input to our MLP is the combination of the state, the noisy action, and the time embedding
        in_dim = state_dim + action_dim + t_dim
        
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Mish())
        
        layers.extend([nn.Linear(hidden_dim, hidden_dim)])
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Mish())
        
        layers.extend([nn.Linear(hidden_dim, hidden_dim)])
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Mish())
        
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.net = nn.Sequential(*layers)
        
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

class TransformerDenoiser(nn.Module):
    def __init__(self, n_assets, action_dim, window_size=60, hidden_dim=256, t_dim=16, n_heads=4, n_layers=2):
        """
        An Attention-based Denoiser that processes the raw 2D market state.
        
        Args:
            n_assets (int): Number of stock features per day.
            action_dim (int): Total number of assets in the portfolio (stocks + cash).
            window_size (int): Number of days in the sequence.
            hidden_dim (int): Size of the Transformer embedding dimension (d_model).
            t_dim (int): Size of the sinusoidal time embedding.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of Transformer encoder layers.
        """
        super().__init__()
        self.t_dim = t_dim
        
        # 1. Project the raw stock returns (n_assets) up to the hidden_dim
        self.state_proj = nn.Linear(n_assets, hidden_dim)
        
        # 2. Positional Encodings so the Transformer understands chronological time
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, hidden_dim) * 0.02)
        
        # 3. The Transformer Backbone (processes the 2D matrix)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Final MLP to process the combined [State + Noisy Action + Time]
        in_dim = hidden_dim + action_dim + t_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.net = self.final_mlp
        
    def time_embedding(self, t):
        """Converts integer timestep into a continuous sinusoidal embedding."""
        half_dim = self.t_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t * embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    def forward(self, state, action_noisy, t):
        # 1. Process the 2D State Matrix (batch, window_size, n_assets)
        # Project up to hidden_dim and add positional encodings
        x = self.state_proj(state) + self.pos_embedding
        
        # Pass through the Self-Attention layers
        x = self.transformer(x)
        
        # Pool the sequence by taking the mean across the window_size dimension
        # This gives us a single 1D vector summarizing the entire 60-day market state
        state_repr = x.mean(dim=1) 
        
        # 2. Prepare Time Embeddings
        t_embed = self.time_embedding(t)
        
        # 3. Combine everything and predict the noise!
        combined = torch.cat([state_repr, action_noisy, t_embed], dim=-1)
        return self.final_mlp(combined)

class PortDiff(nn.Module):
    # Type hints to keep the IDE's static checker happy
    betas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    
    def __init__(self, denoiser, num_timesteps=100, beta_schedule="linear"):
        """
        The core Diffusion Model that wraps the MLPDenoiser.
        
        Args:
            denoiser (nn.Module): The neural network that predicts the noise.
            num_timesteps (int): The total number of steps to add/remove noise (T).
            beta_schedule (str): The noise scheduling strategy ("linear" or "cosine").
        """
        super().__init__()
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        
        # 1. Beta Schedule: Controls how much noise to add at each step
        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif beta_schedule == "cosine":
            # The Cosine Schedule adds noise gently at first to protect the structure
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            s = 0.008
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
        # 2. Alphas: Math shortcuts needed for the diffusion formulas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # We register these as "buffers" so PyTorch saves them but doesn't try to train them.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
    def forward(self, state, action_clean):
        """
        TRAINING ONLY: Calculates the loss by adding noise and asking the denoiser to guess it.
        """
        batch_size = state.shape[0]
        
        # 1. Pick a random timestep 't' for every sample in the batch
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=state.device).long()
        
        # 2. Generate pure random Gaussian noise
        noise = torch.randn_like(action_clean)
        
        # 3. Add the noise to the clean expert weights based on the schedule at timestep 't'
        sqrt_alphas_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        action_noisy = sqrt_alphas_t * action_clean + sqrt_one_minus_alphas_t * noise
        
        # 4. Ask the MLPDenoiser to guess the noise we just added
        predicted_noise = self.denoiser(state, action_noisy, t.unsqueeze(-1).float())
        
        # 5. Return the Mean Squared Error (MSE) between the true noise and the guessed noise
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    @torch.no_grad()
    def sample(self, state):
        """
        INFERENCE ONLY (Real World): Starts with pure noise and denoises it step-by-step to get portfolio weights.
        """
        batch_size = state.shape[0]
        # We find out how many assets there are by looking at the last layer of the denoiser
        action_dim = self.denoiser.net[-1].out_features
        
        # 1. Start with completely random pure Gaussian noise
        action_t = torch.randn((batch_size, action_dim), device=state.device)
        
        # 2. Loop backwards from T-1 down to 0
        for t_step in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size, 1), t_step, device=state.device, dtype=torch.float32)
            
            # Predict the noise using our trained denoiser
            predicted_noise = self.denoiser(state, action_t, t)
            
            # Math to subtract a fraction of the noise based on the timestep schedule
            alpha_t = (1.0 - self.betas[t_step])
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_step]
            
            # Remove the predicted noise
            action_t = (action_t - (1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_t)
            
            # If we aren't at the very last step, add a tiny bit of random noise back in 
            # (this prevents the model from collapsing to a single point, a key diffusion trick)
            if t_step > 0:
                noise = torch.randn_like(action_t)
                sigma_t = torch.sqrt(self.betas[t_step])
                action_t = action_t + sigma_t * noise
                
        # 3. Apply Softmax to ensure the final weights are positive and sum exactly to 1.0 (100%)
        final_weights = F.softmax(action_t, dim=-1)
        return final_weights