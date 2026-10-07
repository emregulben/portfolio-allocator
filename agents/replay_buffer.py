import torch
from torch.utils.data import Dataset

class AWRReplayBuffer(Dataset):
    """
    A specialized PyTorch Dataset for Advantage-Weighted Regression (AWR).
    It stores the states observed by the agent, the specific noisy actions 
    the agent executed, and the calculated financial Advantage of those actions.
    """
    def __init__(self):
        # We start with empty lists to hold our exploration data
        self.states = []
        self.actions = []
        self.advantages = []

    def add_trajectory(self, states, actions, advantage):
        """
        Adds a full 5-year trajectory to the buffer.
        
        Args:
            states (torch.Tensor): The 3D tensor of historical market windows.
            actions (torch.Tensor): The 2D tensor of portfolios the agent executed.
            advantage (float): The single scalar score for how this entire 
                               trajectory performed compared to the batch average.
        """
        # The Sharpe Ratio judges the entire 5-year performance as a whole.
        # So we attach the exact same scalar advantage to every single trading day in this trajectory.
        advantage_tensor = torch.full((len(states), 1), advantage, dtype=torch.float32)
        
        self.states.append(states.cpu())
        self.actions.append(actions.cpu())
        self.advantages.append(advantage_tensor)

    def compile_buffer(self):
        """
        Converts the lists of trajectories into massive, contiguous PyTorch tensors 
        so they can be fed into a DataLoader for training.
        """
        if len(self.states) == 0:
            raise ValueError("Buffer is empty. Cannot compile.")
            
        self.compiled_states = torch.cat(self.states, dim=0)
        self.compiled_actions = torch.cat(self.actions, dim=0)
        self.compiled_advantages = torch.cat(self.advantages, dim=0)

    def __len__(self):
        # Returns the total number of individual trading days stored in the buffer
        return len(self.compiled_states)

    def __getitem__(self, idx):
        # Returns a single (State, Action, Advantage) tuple for the neural network
        return self.compiled_states[idx], self.compiled_actions[idx], self.compiled_advantages[idx]