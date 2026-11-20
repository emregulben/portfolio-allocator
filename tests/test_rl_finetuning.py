import torch

from scripts.run_rl_finetuning import ROLLOUT_BATCH, sample_clean_actions_unconditioned
from agents.diffusion import MeanCovMLPDenoiser, PortDiff


def make_model():
    torch.manual_seed(0)
    denoiser = MeanCovMLPDenoiser(n_assets=4, action_dim=5, hidden_dim=32)
    model = PortDiff(denoiser=denoiser, num_timesteps=10, beta_schedule="linear")
    return model.eval()


def test_constant_market_gives_same_clean_action_every_day():
    # Same market window on all days: with a fixed latent per path, the policy must
    # pick the same portfolio every day (no fake turnover from changing noise).
    model = make_model()
    window = torch.randn(1, 60, 4) * 0.01
    states = window.expand(6, -1, -1).contiguous()
    with torch.no_grad():
        actions = sample_clean_actions_unconditioned(model, states, traj=7)
    assert torch.allclose(actions, actions[0].expand_as(actions), atol=1e-6)


def test_path_uses_same_latent_as_conditioned_rollout_row():
    # Conditioned RL training samples ROLLOUT_BATCH paths per batch; path `traj` is
    # row traj % ROLLOUT_BATCH on every day. The unconditioned path must get that latent.
    model = make_model()
    states = torch.randn(4, 60, 4) * 0.01
    traj = ROLLOUT_BATCH + 3
    with torch.no_grad():
        path = sample_clean_actions_unconditioned(model, states, traj=traj)
        for t in range(states.shape[0]):
            batch = states[t].unsqueeze(0).expand(ROLLOUT_BATCH, -1, -1).contiguous()
            row = model.sample(batch, sampler="ddim")[traj % ROLLOUT_BATCH]
            assert torch.allclose(path[t], row, atol=1e-5)
