import numpy as np
import pandas as pd
import torch

from simulator.hmm import HybridJumpsHMM
from simulator.single_index import SingleIndexModel
from scripts.run_rl_finetuning import compute_daily_utility
from scripts.rl_diagnostics import (
    simulate_states_with_jump_flags,
    expected_asset_returns,
    advantage_parts,
    awr_advantage,
)


def fitted_hmm(n_states=20):
    np.random.seed(1)
    hmm = HybridJumpsHMM(n_states=n_states)
    hmm.fit(pd.Series(np.random.normal(0.0003, 0.01, size=2000)))
    return hmm


def fitted_sim(n_assets=3):
    rng = np.random.default_rng(2)
    market = pd.Series(rng.normal(0.0003, 0.01, size=500))
    stocks = pd.DataFrame(
        {f"S{i}": 0.0001 * i + (0.5 + 0.3 * i) * market + rng.normal(0, 0.01, 500) for i in range(n_assets)}
    )
    sim = SingleIndexModel(tickers=list(stocks.columns))
    sim.fit(stocks, market)
    return sim


def test_same_states_and_same_random_stream_as_the_simulator():
    # The diagnostic copy must draw exactly the same market path as the real simulator,
    # and leave the random generator in the same place (so later draws match too).
    hmm = fitted_hmm()
    for seed in [0, 7, 123]:
        np.random.seed(seed)
        expected = hmm.simulate_states(1500, epsilon=0.01, lambd=20.0)
        after_expected = np.random.rand()
        np.random.seed(seed)
        states, _ = simulate_states_with_jump_flags(hmm, 1500, epsilon=0.01, lambd=20.0)
        after_states = np.random.rand()
        assert np.array_equal(states, expected)
        assert after_states == after_expected


def test_jump_flags_mark_tail_states_only_during_jumps():
    hmm = fitted_hmm()
    np.random.seed(3)
    states, flags = simulate_states_with_jump_flags(hmm, 3000, epsilon=0.01, lambd=20.0, n_tail=2)
    assert flags.any() and not flags.all()
    tail = (states < 2) | (states >= hmm.n_states - 2)
    assert tail[flags].all()
    np.random.seed(3)
    _, no_flags = simulate_states_with_jump_flags(hmm, 3000, epsilon=0.0, lambd=20.0)
    assert not no_flags.any()


def test_expected_returns_use_transition_row_or_jump_mean():
    hmm = fitted_hmm(n_states=10)
    sim = fitted_sim()
    window = 3
    states = np.array([4, 5, 6, 0, 9, 3])
    flags = np.array([False, False, False, True, True, False])
    mu = expected_asset_returns(hmm, sim, states, flags, window, n_tail=2, negative_jump_prob=0.5)

    # one row per decision day t = window..end, columns = stocks + cash
    assert mu.shape == (len(states) - window, 3 + 1)
    assert np.all(mu[:, -1] == 0.0)
    # day 3 is a jump day: market mean = average of bottom-2 and top-2 state means
    jump_m = 0.5 * hmm.state_means[:2].mean() + 0.5 * hmm.state_means[-2:].mean()
    assert np.allclose(mu[0, :-1], sim.alphas + sim.betas * jump_m)
    # day 5 is a normal day: market mean = T[state of day 4] . state means
    normal_m = hmm.T[9] @ hmm.state_means
    assert np.allclose(mu[2, :-1], sim.alphas + sim.betas * normal_m)


def test_advantage_parts_add_up_to_utility_difference():
    torch.manual_seed(0)
    T, A = 30, 4
    windows = torch.randn(T, 20, A - 1) * 0.01
    returns = torch.randn(T, A) * 0.01
    prev = torch.softmax(torch.randn(T, A), dim=1)
    clean = torch.softmax(torch.randn(T, A), dim=1)
    noisy = torch.softmax(torch.randn(T, A), dim=1)
    parts = advantage_parts(clean, noisy, prev, returns, windows, gamma=1.0, cost_c=0.0005)
    total = parts["return"] + parts["risk"] + parts["cost"]
    expected = (compute_daily_utility(noisy, returns, windows, 1.0, 0.0005, prev)
                - compute_daily_utility(clean, returns, windows, 1.0, 0.0005, prev))
    assert torch.allclose(total, expected, atol=1e-7)
    turnover_gap = (noisy - prev).abs().sum(1) - (clean - prev).abs().sum(1)
    assert torch.allclose(parts["cost"], -0.0005 * turnover_gap, atol=1e-8)


def test_advantage_parts_are_zero_without_exploration():
    torch.manual_seed(1)
    T, A = 10, 4
    clean = torch.softmax(torch.randn(T, A), dim=1)
    parts = advantage_parts(clean, clean.clone(), clean.clone(), torch.randn(T, A) * 0.01,
                            torch.randn(T, 20, A - 1) * 0.01, gamma=1.0, cost_c=0.0005)
    for v in parts.values():
        assert torch.all(v == 0)


def test_uniform_mode_gives_zero_advantage_and_others_keep_the_difference():
    n = torch.tensor([0.3, -0.1, 0.2])
    c = torch.tensor([0.1, 0.1, 0.1])
    assert torch.equal(awr_advantage(n, c, "uniform"), torch.zeros(3))
    assert torch.allclose(awr_advantage(n, c, "realized"), n - c)
    assert torch.allclose(awr_advantage(n, c, "expected"), n - c)
