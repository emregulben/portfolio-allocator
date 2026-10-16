import pytest
import torch

from agents.diffusion import MeanCovMLPDenoiser, PortDiff


@pytest.fixture
def test_environment():
    """Initializes a lightweight diffusion model and synthetic state for testing."""
    torch.manual_seed(123)
    n_assets = 4
    action_dim = n_assets + 1  # 4 risky assets + 1 risk-free cash asset
    window_size = 60
    batch_size = 2

    denoiser = MeanCovMLPDenoiser(
        n_assets=n_assets,
        action_dim=action_dim,
        hidden_dim=32,
    )
    model = PortDiff(
        denoiser=denoiser,
        num_timesteps=10,
        beta_schedule="linear",
    )
    model.eval()

    state = torch.randn(batch_size, window_size, n_assets) * 0.02
    return model, state


def test_deterministic_reproducibility(test_environment):
    """Verifies that deterministic samplers (DDIM and DDPM-Det) yield identical outputs for identical states."""
    model, state = test_environment

    for sampler_name in ["ddim", "ddpm_det"]:
        weights_first = model.sample(state, sampler=sampler_name)
        weights_second = model.sample(state, sampler=sampler_name)

        assert torch.equal(weights_first, weights_second), (
            f"Deterministic sampler '{sampler_name}' must return identical weights on repeated evaluations."
        )


def test_stochastic_variance(test_environment):
    """Verifies that stochastic DDPM sampling exhibits dispersion across repeated evaluations."""
    model, state = test_environment

    weights_first = model.sample(state, sampler="ddpm_stoch")
    weights_second = model.sample(state, sampler="ddpm_stoch")

    assert not torch.allclose(weights_first, weights_second), (
        "Stochastic sampling must produce non-identical weights due to reverse step noise."
    )


def test_feasibility_constraints(test_environment):
    """Verifies that all sampling modes satisfy budget and single-asset allocation limits."""
    model, state = test_environment

    for sampler_name in ["ddim", "ddpm_det", "ddpm_stoch"]:
        weights = model.sample(state, sampler=sampler_name)

        # Budget constraint: weights must sum to 1.0
        row_sums = weights.sum(dim=-1).tolist()
        for s in row_sums:
            assert s == pytest.approx(1.0, abs=1e-5)

        # Non-negativity constraint
        assert torch.all(weights >= 0.0), "Portfolio weights must be non-negative."

        # Maximum single-stock constraint (excluding cash column at index -1)
        risky_weights = weights[:, :-1]
        assert torch.all(risky_weights <= 0.20 + 1e-5), (
            "Risky asset weights must not exceed the 20% cap."
        )


def test_ddim_ddpm_det_numerical_closeness(test_environment):
    """
    Verifies that Song et al. DDIM and zero-noise DDPM samplers yield consistent allocations.
    Both discretize the same probability flow trajectory in the zero-noise limit.
    """
    model, state = test_environment

    weights_ddim = model.sample(state, sampler="ddim")
    weights_ddpm_det = model.sample(state, sampler="ddpm_det")

    max_diff = torch.max(torch.abs(weights_ddim - weights_ddpm_det)).item()
    assert max_diff < 0.05, (
        f"DDIM and DDPM-Det allocations deviate by {max_diff:.4f}, expected close trajectory alignment."
    )


def test_invalid_sampler_raises(test_environment):
    """Verifies that unsupported sampler strings raise a ValueError."""
    model, state = test_environment

    with pytest.raises(ValueError, match="Unknown sampler"):
        model.sample(state, sampler="unsupported_sampler")