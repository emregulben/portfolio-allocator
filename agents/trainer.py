import copy
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

class PortDiffTrainer:
    def __init__(self, model, train_dataset, val_dataset, learning_rate=1e-3, batch_size=64):
        """
        Handles the PyTorch training loop for the PortDiff model.
        
        Args:
            model (nn.Module): The PortDiff model.
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset): The validation dataset.
            learning_rate (float): Adam optimizer learning rate.
            batch_size (int): Number of samples per batch.
        """
        # Hardware acceleration: Automatically uses Apple Silicon (MPS), NVIDIA (CUDA), or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # DataLoaders automatically shuffle and chunk our dataset into batches
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Track the best model state to prevent overfitting
        self.best_model_state = None
        
    def train_epoch(self):
        """Runs one full pass over the training data."""
        self.model.train()
        total_loss = 0.0
        
        for state, action_clean in self.train_loader:
            # Move data to GPU / Apple Silicon
            state = state.to(self.device)
            action_clean = action_clean.to(self.device)
            
            # 1. Clear old gradients
            self.optimizer.zero_grad()
            
            # 2. Forward pass (PortDiff automatically adds noise and calculates MSE loss)
            loss = self.model(state, action_clean)
            
            # 3. Backpropagation (calculate errors)
            loss.backward()
            
            # 4. Gradient Clipping: Enforce a mathematical speed limit to prevent explosion!
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 5. Optimizer Step (tweak weights)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
        
    @torch.no_grad() # Disable gradient tracking to save memory
    def validate_epoch(self):
        """Runs one full pass over the validation data. Calculates Diffusion MSE and Action-Space MAE."""
        self.model.eval()
        total_mse_loss = 0.0
        total_mae_loss = 0.0
        
        for state, action_clean in self.val_loader:
            state = state.to(self.device)
            action_clean = action_clean.to(self.device)
            
            # 1. Standard Diffusion Noise Loss
            loss = self.model(state, action_clean)
            total_mse_loss += loss.item()
            
            # 2. Direct Action-Space MAE (Imitation Quality)
            # Sample actual weights from pure noise to see how well we clone the expert
            sampled_actions = self.model.sample(state)
            mae = torch.nn.functional.l1_loss(sampled_actions, action_clean)
            total_mae_loss += mae.item()
            
        return total_mse_loss / len(self.val_loader), total_mae_loss / len(self.val_loader)
    
    def train(self, epochs=50):
        """
        Runs the training loop for the specified number of epochs.
        """
        print(f"Starting training on device: {self.device}")
        best_val_mae = float('inf')
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_mse, val_mae = self.validate_epoch()
            
            # Print progress every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.5f} | Val MSE: {val_mse:.5f} | Val MAE: {val_mae:.5f}")
                
            # Track the best model based on ACTUAL weight imitation (MAE)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                
        print(f"Training complete! Restoring best checkpoint (Val MAE: {best_val_mae:.5f}).")
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        return self.model