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
            
            # 3. Backpropagation (calculate errors) and Optimizer Step (tweak weights)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
        
    @torch.no_grad() # Disable gradient tracking to save memory
    def validate_epoch(self):
        """Runs one full pass over the validation data to check for overfitting."""
        self.model.eval()
        total_loss = 0.0
        
        for state, action_clean in self.val_loader:
            state = state.to(self.device)
            action_clean = action_clean.to(self.device)
            
            loss = self.model(state, action_clean)
            total_loss += loss.item()
            
        return total_loss / len(self.val_loader)
        
    def train(self, epochs=50):
        """
        Runs the training loop for the specified number of epochs.
        """
        print(f"Starting training on device: {self.device}")
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            # Print progress every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Train Loss (MSE): {train_loss:.6f} | Val Loss (MSE): {val_loss:.6f}")
                
            # Track the best model to prevent overfitting
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
        print("Training complete!")
        return self.model