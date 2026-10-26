"""REINFORCE (vanilla policy gradient) on CartPole-v1 with an optional baseline.

Train + evaluate:
    python -m policy_gradient.cartpole_reinforce --out results/cartpole/none --baseline none
    python -m policy_gradient.cartpole_reinforce --out results/cartpole/value --baseline value

Objective: discounted return over the episode, G_t = sum_k gamma^(k-t) r_k (gamma = 0.95).
Loss for one episode: -sum_t log pi(a_t|s_t) * (G_t - b_t), one Adam step per episode.
b_t = 0 (Part 1), a running mean of returns (constant baseline) or a learned state
value V(s_t) fitted to G_t by least squares (Part 2).
"""
import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CsvLogger, discounted_returns, set_seed

ENV_ID = "CartPole-v1"


class CartPolePolicy(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))  # logits


class CartPoleValue(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x))).squeeze(-1)


def run_episode(env, policy, greedy=False, max_steps=None):
    """Sample one episode. Returns (states, actions, rewards)."""
    obs, _ = env.reset()
    states, actions, rewards = [], [], []
    done = False
    while not done:
        s = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits = policy(s)
            a = int(logits.argmax()) if greedy else int(torch.distributions.Categorical(logits=logits).sample())
        obs, r, term, trunc, _ = env.step(a)
        states.append(s.numpy())
        actions.append(a)
        rewards.append(float(r))
        done = term or trunc
        if max_steps is not None and len(rewards) >= max_steps:
            break
    return np.array(states, dtype=np.float32), np.array(actions), np.array(rewards)


def reinforce_loss(policy, states, actions, advantages, normalize_by_length=False):
    """-sum_t log pi(a_t|s_t) A_t (the REINFORCE episode gradient). With
    normalize_by_length=True the sum is divided by the episode length, so the
    gradient scale no longer grows with the episode length."""
    logits = policy(torch.as_tensor(states))
    logp = F.log_softmax(logits, dim=1).gather(1, torch.as_tensor(actions)[:, None]).squeeze(1)
    loss = -(logp * torch.as_tensor(advantages, dtype=torch.float32)).sum()
    return loss / len(actions) if normalize_by_length else loss


def train(args):
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    env = gym.make(ENV_ID)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    policy = CartPolePolicy(args.hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    value = CartPoleValue(args.hidden) if args.baseline == "value" else None
    vopt = torch.optim.Adam(value.parameters(), lr=args.value_lr) if value is not None else None
    ema = None
    log = CsvLogger(os.path.join(args.out, "episodes.csv"),
                    ["episode", "reward", "length", "loss", "G_mean", "G_std", "adv_mean", "adv_std",
                     "baseline_mean", "value_loss", "entropy", "wall_time_s"])
    t0 = time.time()
    rewards_hist = []
    for ep in range(1, args.episodes + 1):
        states, actions, rewards = run_episode(env, policy)
        G = discounted_returns(rewards, args.gamma)
        value_loss = ""
        baseline_mean = 0.0
        if args.baseline == "none":
            adv = G
        elif args.baseline == "mean":
            if ema is None:
                ema = float(G.mean())
            baseline_mean = ema
            adv = G - ema
            ema = (1 - args.ema_rate) * ema + args.ema_rate * float(G.mean())
        elif args.baseline == "value":
            with torch.no_grad():
                V = value(torch.as_tensor(states)).numpy()
            baseline_mean = float(V.mean())
            adv = G - V
        else:
            raise ValueError(args.baseline)
        opt.zero_grad()
        loss = reinforce_loss(policy, states, actions, adv, normalize_by_length=(args.loss_norm == "mean"))
        loss.backward()
        opt.step()
        if value is not None:
            G_t = torch.as_tensor(G, dtype=torch.float32)
            for _ in range(args.value_epochs):
                vopt.zero_grad()
                vl = F.mse_loss(value(torch.as_tensor(states)), G_t)
                vl.backward()
                vopt.step()
            value_loss = float(vl.item())
        with torch.no_grad():
            lp = F.log_softmax(policy(torch.as_tensor(states)), dim=1)
            entropy = float(-(lp.exp() * lp).sum(1).mean())
        ep_r = float(rewards.sum())
        rewards_hist.append(ep_r)
        log.write(episode=ep, reward=ep_r, length=len(rewards), loss=float(loss.item()), G_mean=float(G.mean()),
                  G_std=float(G.std()), adv_mean=float(adv.mean()), adv_std=float(adv.std()),
                  baseline_mean=baseline_mean, value_loss=value_loss, entropy=entropy,
                  wall_time_s=round(time.time() - t0, 2))
        if ep % args.print_every == 0:
            print(f"ep {ep:5d} | reward {ep_r:6.1f} | MA100 {np.mean(rewards_hist[-100:]):6.1f} | "
                  f"entropy {entropy:.3f} | {time.time() - t0:6.1f}s", flush=True)
    log.close()
    torch.save({"policy": policy.state_dict(), "value": value.state_dict() if value is not None else None,
                "config": vars(args)}, os.path.join(args.out, "checkpoint.pt"))
    env.close()
    return policy


def evaluate(policy, episodes, seed, greedy=False, out=None):
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    rewards = []
    for _ in range(episodes):
        _, _, r = run_episode(env, policy, greedy=greedy)
        rewards.append(float(r.sum()))
    env.close()
    rewards = np.array(rewards)
    summary = {"episodes": int(len(rewards)), "mean": float(rewards.mean()), "std": float(rewards.std()),
               "min": float(rewards.min()), "max": float(rewards.max()), "mode": "greedy" if greedy else "sampled"}
    if out is not None:
        tag = summary["mode"]
        np.save(os.path.join(out, f"eval_rewards_{tag}.npy"), rewards)
        with open(os.path.join(out, f"eval_summary_{tag}.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return rewards, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/cartpole/none")
    p.add_argument("--baseline", choices=["none", "mean", "value"], default="none")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--value-lr", type=float, default=0.01)
    p.add_argument("--value-epochs", type=int, default=5)
    p.add_argument("--ema-rate", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--loss-norm", choices=["sum", "mean"], default="sum",
                   help="sum: plain REINFORCE episode gradient; mean: divide by the episode length")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=500)
    p.add_argument("--print-every", type=int, default=100)
    args = p.parse_args()
    torch.set_num_threads(1)
    policy = train(args)
    _, summary = evaluate(policy, args.eval_episodes, seed=args.seed + 1000, greedy=False, out=args.out)
    print("eval (sampled actions):", json.dumps(summary))
    _, summary_g = evaluate(policy, args.eval_episodes, seed=args.seed + 2000, greedy=True, out=args.out)
    print("eval (greedy actions):", json.dumps(summary_g))


if __name__ == "__main__":
    main()
