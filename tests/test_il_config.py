import os

import pytest
import torch
import yaml

from agents.diffusion import build_il_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def config():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("checkpoint, conditioned", [
    ("il_winner.pth", False),
    ("il_conditioned.pth", True),
])
def test_config_builds_the_saved_il_architecture(config, checkpoint, conditioned):
    # The model built from config.yaml must have exactly the layers of the saved winner
    # (hidden size, layernorm on/off) and the same noise schedule (betas), so a retrain
    # from the config reproduces the architecture the grid search picked.
    n_assets = len(config["data"]["stock_tickers"])
    model = build_il_model(config["il"], n_assets, conditioned=conditioned)
    saved = torch.load(os.path.join(ROOT, checkpoint), map_location="cpu")
    built = model.state_dict()

    assert set(built) == set(saved)
    for name, tensor in saved.items():
        assert built[name].shape == tensor.shape, name
    # Noise-schedule buffers (betas, alphas, ...) are fixed numbers set by the config.
    # Denoiser buffers such as BatchNorm running stats are learned, so they are skipped.
    schedule = [name for name, _ in model.named_buffers() if not name.startswith("denoiser.")]
    assert "betas" in schedule
    for name in schedule:
        assert torch.allclose(built[name], saved[name]), name


def test_config_training_settings_match_the_winner_run(config):
    # The winner was trained with the PortDiffTrainer defaults (lr 1e-3, batch 64) for
    # 15 epochs, the settings fixed in run_il_grid_search.py.
    il = config["il"]
    assert il["learning_rate"] == pytest.approx(1e-3)
    assert il["batch_size"] == 64
    assert il["epochs"] == 15
