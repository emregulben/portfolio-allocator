import torch
from torch.utils.data import TensorDataset

from agents.awr_trainer import PortDiffAWRTrainer
from agents.replay_buffer import AWRReplayBuffer
from agents.trainer import PortDiffTrainer


def test_il_trainer_uses_the_requested_device():
    # Without this the trainer silently moves the model to MPS, even when the
    # script asked for CPU (MPS is unreliable after the macOS 27.2 update).
    model = torch.nn.Linear(3, 2)
    data = TensorDataset(torch.zeros(4, 3), torch.zeros(4, 2))
    trainer = PortDiffTrainer(model, data, data, device="cpu")
    assert trainer.device == torch.device("cpu")
    assert next(model.parameters()).device == torch.device("cpu")


def test_awr_trainer_uses_the_requested_device():
    model = torch.nn.Linear(3, 2)
    trainer = PortDiffAWRTrainer(model, AWRReplayBuffer(), device="cpu")
    assert trainer.device == torch.device("cpu")
    assert next(model.parameters()).device == torch.device("cpu")


def _train_once(seed):
    from agents.diffusion import build_il_model
    il_cfg = {"hidden_dim": 32, "t_dim": 16, "use_layernorm": False,
              "num_timesteps": 10, "beta_schedule": "linear"}
    torch.manual_seed(seed)
    model = build_il_model(il_cfg, n_assets=3)
    data_gen = torch.Generator().manual_seed(1)
    states = torch.randn(16, 60, 3, generator=data_gen) * 0.01
    actions = torch.softmax(torch.randn(16, 4, generator=data_gen), dim=1)
    data = TensorDataset(states, actions)
    trainer = PortDiffTrainer(model, data, data, batch_size=4, device="cpu")
    trainer.train(epochs=2)
    return trainer.model.state_dict()


def test_same_seed_gives_the_same_trained_model():
    # The training scripts call torch.manual_seed(config seed) before building the model.
    # That must make the whole training (start weights, batch order, diffusion noise)
    # repeatable on CPU, so a retrain gives the exact same checkpoint.
    first, second = _train_once(0), _train_once(0)
    assert all(torch.equal(first[k], second[k]) for k in first)
    other = _train_once(1)
    assert not all(torch.equal(first[k], other[k]) for k in first)
