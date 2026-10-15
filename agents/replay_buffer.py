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
        Appends a trajectory segment or chunk to the replay buffer, supporting
        both trajectory-level and step-level (daily) advantage signals.

        Args:
            states (torch.Tensor): Historical market feature windows of shape (num_steps, window_size, n_assets).
            actions (torch.Tensor): Portfolio allocations executed by the policy of shape (num_steps, action_dim).
            advantage (Union[float, torch.Tensor]): Advantage signal for credit assignment. Can be either:
                - A scalar float: broadcasted uniformly across all steps in the
                  chunk (used for trajectory-level metrics such as the Sharpe ratio).
                - A 1D or 2D torch.Tensor of shape (num_steps,) or (num_steps, 1):
                  provides localized per-step advantage values (used for daily 
                  utility-based credit assignment).
        """
        if isinstance(advantage, torch.Tensor) and advantage.numel() == len(states):
            advantage_tensor = advantage.view(-1, 1).float().cpu()
        else:
            advantage_tensor = torch.full((len(states), 1), float(advantage), dtype=torch.float32)
        
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

class ConditionedAWRReplayBuffer(Dataset):
    """
    A specialized PyTorch Dataset for Advantage-Weighted Regression (AWR)
    that stores incumbent portfolio weights w_{t-1} alongside states, actions, and advantages.
    """
    def __init__(self):
        self.states = []
        self.prev_weights = []
        self.actions = []
        self.advantages = []

    def add_trajectory(self, states, prev_weights, actions, advantage):
        """
        Appends a trajectory segment to the replay buffer with incumbent portfolio weights.
        """
        if isinstance(advantage, torch.Tensor) and advantage.numel() == len(states):
            advantage_tensor = advantage.view(-1, 1).float().cpu()
        else:
            advantage_tensor = torch.full((len(states), 1), float(advantage), dtype=torch.float32)

        self.states.append(states.cpu())
        self.prev_weights.append(prev_weights.cpu())
        self.actions.append(actions.cpu())
        self.advantages.append(advantage_tensor)

    def compile_buffer(self):
        """
        Converts trajectory lists into contiguous PyTorch tensors for DataLoader consumption.
        """
        if len(self.states) == 0:
            raise ValueError("Buffer is empty. Cannot compile.")

        self.compiled_states = torch.cat(self.states, dim=0)
        self.compiled_prev_weights = torch.cat(self.prev_weights, dim=0)
        self.compiled_actions = torch.cat(self.actions, dim=0)
        self.compiled_advantages = torch.cat(self.advantages, dim=0)

    def __len__(self):
        return len(self.compiled_states)

    def __getitem__(self, idx):
        return (
            self.compiled_states[idx],
            self.compiled_prev_weights[idx],
            self.compiled_actions[idx],
            self.compiled_advantages[idx],
        )