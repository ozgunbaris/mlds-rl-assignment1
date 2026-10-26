"""Figures and summary tables for the CartPole runs.

    python -m policy_gradient.make_plots_cartpole --results results/cartpole --out results/cartpole/figures
"""
import argparse
import glob
import json
import os
import re
from collections import OrderedDict, defaultdict

import numpy as np

import viz
from policy_gradient.common import moving_average, read_csv_column


def load_runs(results):
    runs = []
    for cfg_path in sorted(glob.glob(os.path.join(results, "*", "config.json"))):
        d = os.path.dirname(cfg_path)
        if not os.path.exists(os.path.join(d, "episodes.csv")) or not os.path.exists(os.path.join(d, "eval_summary_sampled.json")):
            continue
        cfg = json.load(open(cfg_path))
        rewards = read_csv_column(os.path.join(d, "episodes.csv"), "reward")
        ev = json.load(open(os.path.join(d, "eval_summary_sampled.json")))
        evg = json.load(open(os.path.join(d, "eval_summary_greedy.json"))) if os.path.exists(os.path.join(d, "eval_summary_greedy.json")) else None
        ev_rewards = np.load(os.path.join(d, "eval_rewards_sampled.npy"))
        runs.append({"dir": d, "name": os.path.basename(d), "baseline": cfg["baseline"], "lr": cfg["lr"], "seed": cfg["seed"],
                     "episodes": cfg["episodes"], "rewards": rewards, "eval": ev, "eval_greedy": evg, "eval_rewards": ev_rewards})
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/cartpole")
    p.add_argument("--out", default="results/cartpole/figures")
    p.add_argument("--window", type=int, default=100)
    args = p.parse_args()
    os.makedirs(os.path.join(args.out, "per_run"), exist_ok=True)
    runs = load_runs(args.results)
    print(f"{len(runs)} runs")

    # per-run required figures
    for r in runs:
        tag = f"{r['baseline']}_lr{r['lr']}_seed{r['seed']}"
        viz.training_curve(r["rewards"], os.path.join(args.out, "per_run", f"{tag}_training.png"),
                           f"CartPole-v1 REINFORCE, baseline = {r['baseline']}, lr = {r['lr']}, seed {r['seed']}", window=args.window)
        viz.reward_histogram(r["eval_rewards"], os.path.join(args.out, "per_run", f"{tag}_hist.png"),
                             f"CartPole-v1: 500 roll-out episodes, baseline = {r['baseline']}, lr = {r['lr']}, seed {r['seed']}",
                             bins=np.arange(0, 526, 25))

    # grouped summaries
    groups = defaultdict(list)
    for r in runs:
        groups[(r["baseline"], r["lr"])].append(r)
    summary = {}
    for (b, lr), rs in sorted(groups.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
        rs = sorted(rs, key=lambda r: r["seed"])
        final = [float(moving_average(r["rewards"], args.window)[-1]) for r in rs]
        peak = [float(moving_average(r["rewards"], args.window).max()) for r in rs]
        ev_means = [r["eval"]["mean"] for r in rs]
        ev_stds = [r["eval"]["std"] for r in rs]
        evg_means = [r["eval_greedy"]["mean"] for r in rs if r["eval_greedy"]]
        first500 = [int(np.argmax(moving_average(r["rewards"], args.window) >= 475) + 1) if (moving_average(r["rewards"], args.window) >= 475).any() else None for r in rs]
        summary[f"{b}_lr{lr}"] = {
            "baseline": b, "lr": lr, "seeds": [r["seed"] for r in rs], "train_episodes": rs[0]["episodes"],
            "final_ma100_per_seed": final, "peak_ma100_per_seed": peak,
            "first_episode_ma100_ge_475_per_seed": first500,
            "eval_mean_per_seed": ev_means, "eval_std_per_seed": ev_stds, "eval_greedy_mean_per_seed": evg_means,
            "eval_mean_avg": float(np.mean(ev_means)), "eval_mean_min": float(np.min(ev_means)), "eval_mean_max": float(np.max(ev_means)),
            "seeds_collapsed_eval_lt_50": int(sum(m < 50 for m in ev_means)),
            "best_seed": rs[int(np.argmax(ev_means))]["seed"],
        }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    md = ["| baseline | lr | seeds | final MA100 per seed | peak MA100 per seed | roll-out mean (500 ep.) per seed | roll-out std per seed | collapsed seeds |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in summary.items():
        md.append(f"| {v['baseline']} | {v['lr']} | {len(v['seeds'])} | " + ", ".join(f"{x:.0f}" for x in v["final_ma100_per_seed"]) + " | " +
                  ", ".join(f"{x:.0f}" for x in v["peak_ma100_per_seed"]) + " | " + ", ".join(f"{x:.1f}" for x in v["eval_mean_per_seed"]) + " | " +
                  ", ".join(f"{x:.1f}" for x in v["eval_std_per_seed"]) + f" | {v['seeds_collapsed_eval_lt_50']} |")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    # comparison figures: per lr, baselines side by side (seed bands)
    lrs = sorted({lr for _, lr in groups}, reverse=True)
    for lr in lrs:
        curves = OrderedDict()
        for b in ["none", "mean", "value"]:
            if (b, lr) in groups:
                curves[f"baseline = {b}"] = [r["rewards"] for r in groups[(b, lr)]]
        viz.compare_curves(curves, os.path.join(args.out, f"compare_baselines_lr{lr}.png"),
                           f"CartPole-v1: moving-average reward, lr = {lr}, mean +- 1 std over seeds", window=args.window)
    for b in ["none", "mean", "value"]:
        curves = OrderedDict()
        for lr in lrs:
            if (b, lr) in groups:
                curves[f"lr = {lr}"] = [r["rewards"] for r in groups[(b, lr)]]
        if curves:
            viz.compare_curves(curves, os.path.join(args.out, f"compare_lr_{b}.png"),
                               f"CartPole-v1: moving-average reward, baseline = {b}, mean +- 1 std over seeds", window=args.window)
    # all seeds of each configuration as individual MA lines
    for (b, lr), rs in groups.items():
        curves = OrderedDict((f"seed {r['seed']}", r["rewards"]) for r in sorted(rs, key=lambda r: r["seed"]))
        viz.compare_curves(curves, os.path.join(args.out, f"seeds_{b}_lr{lr}.png"),
                           f"CartPole-v1: moving-average reward per seed, baseline = {b}, lr = {lr}", window=args.window)


if __name__ == "__main__":
    main()
