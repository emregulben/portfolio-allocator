import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.replay_buffer import AWRReplayBuffer

def test_awr_replay_buffer():
    buffer = AWRReplayBuffer()
    
    # 1. Create a fake 5-year trajectory (e.g., 1000 days, 60-day window, 10 assets)
    fake_states_1 = torch.rand(1000, 60, 10)
    fake_actions_1 = torch.rand(1000, 11)  # 10 assets + 1 cash
    fake_advantage_1 = 2.5  
    
    buffer.add_trajectory(fake_states_1, fake_actions_1, fake_advantage_1)
    
    # 2. Create a second fake trajectory that performed terribly
    fake_states_2 = torch.rand(1000, 60, 10)
    fake_actions_2 = torch.rand(1000, 11)
    fake_advantage_2 = -1.2 
    
    buffer.add_trajectory(fake_states_2, fake_actions_2, fake_advantage_2)
    
    # 3. Compile the buffer
    buffer.compile_buffer()
    
    # 4. Assertions
    assert len(buffer) == 2000, "Buffer should contain 2000 total trading days."
    
    # Check the first day
    state, action, advantage = buffer[0]
    assert abs(advantage.item() - 2.5) < 1e-5, "First trajectory advantage mismatch."
    assert state.shape == (60, 10), "State shape is corrupted."
    
    # Check the last day (using absolute difference to bypass floating point error)
    state, action, advantage = buffer[1999]
    assert abs(advantage.item() - (-1.2)) < 1e-5, "Second trajectory advantage mismatch."