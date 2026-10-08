import torch
from torch.utils.data import DataLoader
import torch.optim as optim

class PortDiffAWRTrainer:
    def __init__(self, model, replay_buffer, learning_rate=1e-4, batch_size=256, beta=0.05, weight_clip=20.0):
        """
        Handles the PyTorch training loop for Reinforcement Learning (AWR).
        
        Args:
            model (nn.Module): The PortDiff model.
            replay_buffer (AWRReplayBuffer): The dataset containing states, actions, and advantages.
            learning_rate (float): Adam optimizer learning rate (lower for fine-tuning).
            batch_size (int): Number of samples per batch.
            beta (float): The temperature parameter. Controls how strictly we filter advantages.
                          Lower beta = strictly only copies the absolute best actions.
            weight_clip (float): Maximum allowed weight to prevent gradient explosion.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.beta = beta
        self.weight_clip = weight_clip
        
    def train_epoch(self):
        """Runs one full pass over the replay buffer, weighting losses by Advantage."""
        self.model.train()
        total_loss = 0.0
        
        # We create the DataLoader here so it always uses the most up-to-date compiled buffer
        loader = DataLoader(self.replay_buffer, batch_size=self.batch_size, shuffle=True)
        
        for state, action_noisy, advantage in loader:
            state = state.to(self.device)
            action_noisy = action_noisy.to(self.device)
            advantage = advantage.to(self.device)
            
            # 1. Calculate the AWR Weights mathematically: w = exp(Advantage / Beta)
            weights = torch.exp(advantage / self.beta)
            
            # 2. Clip the weights so a crazy high Sharpe Ratio doesn't explode the network
            weights = torch.clamp(weights, max=self.weight_clip)
            
            # 3. Clear old gradients
            self.optimizer.zero_grad()
            
            # 4. Forward pass: The model calculates MSE and multiplies it by our AWR weights!
            loss = self.model(state, action_noisy, weights=weights)
            
            # 5. Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)
        
    def train(self, epochs=5):
        """Trains the model for a few epochs on the filtered buffer."""
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            print(f"RL Finetuning Epoch {epoch:03d} | Weighted Loss: {train_loss:.5f}")