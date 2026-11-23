"""
Diagnostics for the AWR fine-tuning of the conditioned policy.

Three commands (outputs go to runs/rl_diagnostics/ unless --out is given):

  analyze   One training-style rollout of IL Cond (no training). Splits the one-day advantage
            into its return, risk and cost parts and measures what the AWR weights prefer.
  train     AWR fine-tuning of IL Cond with one change at a time:
              --mode realized : the current reward (realized one-day utility)
              --mode uniform  : all AWR weights equal (advantage set to 0)
              --mode expected : realized return replaced by E[r_t | simulator state]
              --noise SIGMA   : exploration noise (default 0.05)
  evaluate  Evaluates checkpoints on the final-evaluation markets (seeds 9000+, same latents)
            and pairs them with IL Cond, RL Cond and cost-aware Markowitz from
            final_evaluation_results.json.

The repository models are only read, never overwritten.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simulator.loader import MarketDataLoader  # noqa: E402
from simulator.hmm import HybridJumpsHMM  # noqa: E402
from simulator.single_index import SingleIndexModel  # noqa: E402
from agents.diffusion import MeanCovMLPDenoiser, PortDiff, map_to_feasible_portfolio  # noqa: E402
from agents.replay_buffer import ConditionedAWRReplayBuffer  # noqa: E402
from agents.awr_trainer import PortDiffAWRTrainer  # noqa: E402
from scripts.run_rl_finetuning import compute_daily_utility, ROLLOUT_BATCH  # noqa: E402

CHUNK = 252  # advantage chunks of one trading year, as in run_rl_finetuning.py


# ---------------------------------------------------------------- helpers (tested)

def simulate_states_with_jump_flags(hmm, n_steps, epsilon, lambd, n_tail=5, negative_jump_prob=0.52):
    """
    Same random draws as HybridJumpsHMM.simulate_states, so the market path is identical,
    but also returns which days belong to a jump episode.
    """
    states = np.zeros(n_steps, dtype=int)
    in_jump = np.zeros(n_steps, dtype=bool)
    s_bottom = np.arange(n_tail)
    s_top = np.arange(hmm.n_states - n_tail, hmm.n_states)
    states[0] = np.searchsorted(np.cumsum(hmm.pi_bar), np.random.rand())
    counter = 1
    while counter < n_steps:
        if np.random.rand() < epsilon:
            k_jump = min(np.random.poisson(lambd), n_steps - counter)
            if k_jump > 0:
                is_bottom = np.random.rand(k_jump) < negative_jump_prob
                states[counter:counter + k_jump] = np.where(
                    is_bottom, np.random.choice(s_bottom, size=k_jump), np.random.choice(s_top, size=k_jump))
                in_jump[counter:counter + k_jump] = True
                counter += k_jump
        else:
            states[counter] = np.searchsorted(hmm.T_cumsum[states[counter - 1]], np.random.rand())
            counter += 1
    return states, in_jump


def expected_asset_returns(hmm, sim_model, states, jump_flags, window, n_tail=5, negative_jump_prob=0.52):
    """
    Expected return of every asset on each decision day t = window, ..., len(states) - 1.

    Market: on a jump day the mean of the jump tail states, otherwise T[s_{t-1}] . state means.
    Stocks: alpha + beta * expected market return (single-index model). Cash: 0.
    The policy chooses w_t before r_t is drawn, so replacing r_t by this expectation keeps the
    expected reward and removes most of its day-to-day noise.
    """
    jump_mean = (negative_jump_prob * hmm.state_means[:n_tail].mean()
                 + (1 - negative_jump_prob) * hmm.state_means[-n_tail:].mean())
    normal_mean = hmm.T @ hmm.state_means
    m = np.where(jump_flags[window:], jump_mean, normal_mean[states[window - 1:-1]])
    stocks = sim_model.alphas + np.outer(m, sim_model.betas)
    return np.concatenate([stocks, np.zeros((len(m), 1))], axis=1)


def advantage_parts(clean, noisy, prev_weights, returns, windows, gamma, cost_c):
    """One-day advantage (noisy minus clean utility) split into its return, risk and cost parts."""
    cov_windows = windows - windows.mean(dim=1, keepdim=True)
    cov = torch.bmm(cov_windows.transpose(1, 2), cov_windows) / (windows.shape[1] - 1)

    def parts(w):
        ret = (w * returns).sum(1)
        wr = w[:, :-1].unsqueeze(1)
        var = torch.bmm(torch.bmm(wr, cov), wr.transpose(1, 2)).view(-1)
        to = (w - prev_weights).abs().sum(1)
        return ret, -(gamma / 2.0) * var, -cost_c * to

    rn, kn, cn = parts(noisy)
    rc, kc, cc = parts(clean)
    return {"return": rn - rc, "risk": kn - kc, "cost": cn - cc}


def awr_advantage(noisy_utility, clean_utility, mode):
    """Paired advantage; in 'uniform' mode it is zero, so every sample gets the same AWR weight."""
    if mode == "uniform":
        return torch.zeros_like(noisy_utility)
    return noisy_utility - clean_utility


# ---------------------------------------------------------------- setup

def load_setup():
    config = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    tick_m, ticks = config["data"]["market_ticker"], config["data"]["stock_tickers"]
    data = MarketDataLoader(tickers=[tick_m] + ticks, start_date=config["data"]["start_date"],
                            end_date=config["data"]["end_date"],
                            cache_path=os.path.join(ROOT, config["data"]["snapshot_path"])).fetch_data()
    hmm = HybridJumpsHMM(n_states=config["hmm"]["n_states"], emission_dist="gaussian")
    hmm.fit(data[tick_m])
    hmm.epsilon, hmm.lambd = 0.0025, 85.0  # Simulator B, as in RL training and final evaluation
    sim = SingleIndexModel(tickers=ticks, emission_dist="gaussian")
    sim.fit(pd.DataFrame(data[ticks]), data[tick_m])
    return config, hmm, sim


def load_conditioned(path, n_assets):
    den = MeanCovMLPDenoiser(n_assets=n_assets, action_dim=n_assets + 1, hidden_dim=1024,
                             condition_on_prev_weights=True)
    model = PortDiff(denoiser=den, num_timesteps=100, beta_schedule="linear")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def simulate_markets(config, hmm, sim, seeds):
    """Windows, realized returns and expected returns for each seed (same paths as training/evaluation)."""
    window = config["expert"]["rolling_window"]
    n_steps = config["hmm"]["simulation"]["n_steps"]
    rf = config["expert"]["annual_risk_free_rate"] / 252.0
    S, R, MU = [], [], []
    for seed in seeds:
        np.random.seed(seed)
        s, flags = simulate_states_with_jump_flags(hmm, n_steps, hmm.epsilon, hmm.lambd)
        stocks = sim.simulate(hmm.decode_states(s), random_seed=seed).to_numpy()
        S.append(np.array([stocks[t - window:t] for t in range(window, len(stocks))]))
        R.append(np.concatenate([stocks[window:], np.full((len(stocks) - window, 1), rf)], axis=1))
        MU.append(expected_asset_returns(hmm, sim, s, flags, window))
    as_t = lambda x: torch.tensor(np.array(x), dtype=torch.float32)  # noqa: E731
    return as_t(S), as_t(R), as_t(MU)


def drift(w, r):
    g = (w * r).sum(1, keepdim=True)
    return w * (1 + r) / torch.clamp(1 + g, min=1e-8)


def explore_rollout(model, S, R, noise):
    """Training-style rollout: clean DDIM action, noisy action executed (as in run_rl_conditioned)."""
    B, T, A = S.shape[0], S.shape[1], R.shape[2]
    pw = torch.zeros(B, A)
    pw[:, -1] = 1.0
    C, N, P = [], [], []
    with torch.no_grad():
        for t in range(T):
            P.append(pw.clone())
            c = model.sample(S[:, t], prev_weights=pw, sampler="ddim")
            n = map_to_feasible_portfolio(c + torch.randn_like(c) * noise, max_weight=0.20)
            C.append(c)
            N.append(n)
            pw = drift(n, R[:, t])
    return torch.stack(C, 1), torch.stack(N, 1), torch.stack(P, 1)


# ---------------------------------------------------------------- commands

def cmd_analyze(args):
    config, hmm, sim = load_setup()
    gamma, cost_c = config["expert"]["risk_aversion"], config["expert"]["transaction_cost"]
    torch.manual_seed(0)
    S, R, MU = simulate_markets(config, hmm, sim, range(args.start_seed, args.start_seed + args.paths))
    model = load_conditioned(os.path.join(ROOT, "il_conditioned.pth"), S.shape[-1])
    C, N, P = explore_rollout(model, S, R, args.noise)
    flat = lambda x: x.reshape(-1, x.shape[-1])  # noqa: E731
    Sf = S.reshape(-1, S.shape[2], S.shape[3])
    real = advantage_parts(flat(C), flat(N), flat(P), flat(R), Sf, gamma, cost_c)
    exp = advantage_parts(flat(C), flat(N), flat(P), flat(MU), Sf, gamma, cost_c)
    adv_real = real["return"] + real["risk"] + real["cost"]
    adv_exp = exp["return"] + exp["risk"] + exp["cost"]
    extra_turnover = -real["cost"] / cost_c
    corr = lambda a, b: float(np.corrcoef(a.numpy(), b.numpy())[0, 1])  # noqa: E731
    out = {
        "std": {k: float(v.std()) for k, v in real.items()},
        "std_expected_return_part": float(exp["return"].std()),
        "corr_adv_return_part": corr(adv_real, real["return"]),
        "corr_adv_extra_turnover": corr(adv_real, extra_turnover),
        "corr_expected_adv_extra_turnover": corr(adv_exp, extra_turnover),
        "corr_adv_expected_adv": corr(adv_real, adv_exp),
        "cash_bias_noisy_minus_clean": float((N[..., -1] - C[..., -1]).mean()),
    }
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(args.out, "analyze.json"), "w"), indent=2)


def cmd_train(args):
    config, hmm, sim = load_setup()
    rl = config["rl"]
    gamma, cost_c = config["expert"]["risk_aversion"], config["expert"]["transaction_cost"]
    torch.manual_seed(rl["seed"])
    iterations = args.iterations or rl["iterations"]
    total = args.trajs or rl["trajectories_per_iteration"]
    model = load_conditioned(os.path.join(ROOT, "il_conditioned.pth"), len(config["data"]["stock_tickers"]))
    buffer = ConditionedAWRReplayBuffer()
    trainer = PortDiffAWRTrainer(model=model, replay_buffer=buffer, learning_rate=rl["learning_rate"],
                                 batch_size=rl["batch_size"], beta=rl["beta"],
                                 weight_clip=rl["weight_clip"], device="cpu")
    log = []
    for it in range(1, iterations + 1):
        for b0 in range(0, total, ROLLOUT_BATCH):
            seeds = [it * 1000 + traj for traj in range(b0, min(b0 + ROLLOUT_BATCH, total))]
            S, R, MU = simulate_markets(config, hmm, sim, seeds)
            model.eval()
            C, N, P = explore_rollout(model, S, R, args.noise)
            Rw = MU if args.mode == "expected" else R
            for i in range(S.shape[0]):
                for k in range(S.shape[1] // CHUNK):
                    sl = slice(k * CHUNK, (k + 1) * CHUNK)
                    cu = compute_daily_utility(C[i, sl], Rw[i, sl], S[i, sl], gamma, cost_c, P[i, sl])
                    nu = compute_daily_utility(N[i, sl], Rw[i, sl], S[i, sl], gamma, cost_c, P[i, sl])
                    buffer.add_trajectory(S[i, sl], P[i, sl], N[i, sl], awr_advantage(nu, cu, args.mode))
            print(f"iteration {it}: {min(b0 + ROLLOUT_BATCH, total)}/{total} paths", flush=True)
        if args.mode != "uniform":
            adv = torch.cat(buffer.advantages, 0)
            buffer.advantages = [(a - adv.mean()) / (adv.std() + 1e-8) for a in buffer.advantages]
        buffer.compile_buffer()
        trainer.train(epochs=rl["epochs_per_iteration"])
        buffer = ConditionedAWRReplayBuffer()
        trainer.replay_buffer = buffer
        path = os.path.join(args.out, f"rl_cond_{args.tag}_it{it}.pth")
        torch.save(model.state_dict(), path)
        log.append({"iteration": it, "checkpoint": path})
    json.dump({"args": vars(args), "log": log}, open(os.path.join(args.out, f"train_{args.tag}.json"), "w"), indent=2)
    print("TRAINING DONE", flush=True)


def cmd_evaluate(args):
    from scipy import stats
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from final_evaluation import compute_portfolio_metrics, rollout_conditioned_batch

    config, hmm, sim = load_setup()
    ref = json.load(open(os.path.join(ROOT, "final_evaluation_results.json")))["per_seed"]
    S, R, _ = simulate_markets(config, hmm, sim, range(9000, 9000 + args.paths))
    cost_c = config["expert"]["transaction_cost"]
    summary = {}
    for ck in args.checkpoints:
        W = rollout_conditioned_batch(load_conditioned(ck, S.shape[-1]), S, R, torch.device("cpu"), sampler="ddim")
        mets = [compute_portfolio_metrics(W[i], R[i].numpy(), cost_c) for i in range(args.paths)]
        row = {k: float(np.mean([d[k] for d in mets])) for k in mets[0]}
        row["cash"] = float(W[..., -1].mean())
        a = np.array([d["net_sharpe"] for d in mets])
        for key in ["il_conditioned_det", "rl_conditioned_det", "markowitz_tc"]:
            b = np.array([d["net_sharpe"] for d in ref[key][:args.paths]])
            row[f"net_sharpe_vs_{key}"] = {"diff": float((a - b).mean()),
                                           "p": float(stats.ttest_rel(a, b).pvalue),
                                           "wins": int((a > b).sum())}
        summary[os.path.basename(ck)] = row
        print(os.path.basename(ck), json.dumps(row), flush=True)
    json.dump(summary, open(os.path.join(args.out, f"eval_{args.name}.json"), "w"), indent=2)
    print("EVAL DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "rl_diagnostics"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--paths", type=int, default=20)
    a.add_argument("--start-seed", type=int, default=9000)
    a.add_argument("--noise", type=float, default=0.05)
    t = sub.add_parser("train")
    t.add_argument("--mode", choices=["realized", "uniform", "expected"], required=True)
    t.add_argument("--noise", type=float, default=0.05)
    t.add_argument("--iterations", type=int, default=None)
    t.add_argument("--trajs", type=int, default=None)
    t.add_argument("--tag", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("checkpoints", nargs="+")
    e.add_argument("--paths", type=int, default=100)
    e.add_argument("--name", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"analyze": cmd_analyze, "train": cmd_train, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
