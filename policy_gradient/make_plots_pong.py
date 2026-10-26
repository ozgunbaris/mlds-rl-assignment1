"""Figures and summary for the Pong runs.

    python -m policy_gradient.make_plots_pong --runs results/pong/none_lr1e-3 results/pong/value_lr1e-3 --out results/pong/figures
"""
import argparse
import csv
import glob
import json
import os
from collections import OrderedDict

import numpy as np

import viz
from policy_gradient.common import moving_average


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--out", default="results/pong/figures")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--diag-runs", type=int, default=None, help="only the first N runs appear in the entropy/advantage plots")
    p.add_argument("--group", action="append", default=[], metavar="LABEL=GLOB",
                   help="seed group, e.g. 'no baseline=results/pong/cluster_none_seed*'; drawn as mean +- std over seeds")
    p.add_argument("--group-name", default="seed_groups", help="stem of the group figure/json (compare_<name>.png, <name>.json)")
    p.add_argument("--group-title", default="mean +- 1 std over seeds")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in args.runs]
    summary = {}
    curves_ep, curves_steps, curves_time = OrderedDict(), OrderedDict(), OrderedDict()
    import matplotlib.pyplot as plt
    viz.setup()
    fig_adv, ax_adv = plt.subplots(figsize=(9, 4.2))
    fig_ent, ax_ent = plt.subplots(figsize=(9, 4.2))
    for k, (run, label) in enumerate(zip(args.runs, labels)):
        name = os.path.basename(run.rstrip("/"))
        cfg = json.load(open(os.path.join(run, "config.json")))
        eps = read_csv(os.path.join(run, "episodes.csv"))
        rewards = np.array([float(r["reward"]) for r in eps])
        steps = np.array([int(r["env_steps"]) for r in eps])
        wall = np.array([float(r["wall_time_s"]) for r in eps]) / 3600
        ma = moving_average(rewards, args.window)
        viz.training_curve(rewards, os.path.join(args.out, f"{name}_training.png"),
                           f"Pong-v5 REINFORCE, baseline = {cfg['baseline']}, lr = {cfg['lr']}: episode reward during training", window=args.window)
        s = {"baseline": cfg["baseline"], "lr": cfg["lr"], "episodes": int(len(rewards)), "env_steps": int(steps[-1]),
             "hours": float(wall[-1]), "final_ma100": float(ma[-1]), "best_ma100": float(ma.max()), "best_ma100_episode": int(ma.argmax() + 1),
             "first_ma100": float(ma[min(99, len(ma) - 1)])}
        for tag in ["sampled", "greedy"]:
            f = os.path.join(run, f"eval_summary_{tag}.json")
            if os.path.exists(f):
                ev = json.load(open(f))
                s[f"eval_{tag}"] = {kk: ev[kk] for kk in ["episodes", "mean", "std", "min", "max"]}
                er = np.load(os.path.join(run, f"eval_rewards_{tag}.npy"))
                viz.reward_histogram(er, os.path.join(args.out, f"{name}_hist_{tag}.png"),
                                     f"Pong-v5: {len(er)} roll-out episodes ({tag} actions), baseline = {cfg['baseline']}, lr = {cfg['lr']}",
                                     bins=np.arange(-21.5, 22.5, 1.0))
        summary[name] = s
        curves_ep[label] = rewards
        curves_steps[label] = (steps, ma)
        curves_time[label] = (wall, ma)
        ups = read_csv(os.path.join(run, "updates.csv"))
        if ups and (args.diag_runs is None or k < args.diag_runs):
            u = np.array([int(r["update"]) for r in ups])
            ax_adv.plot(u, [float(r["G_std"]) for r in ups], color=viz.SERIES[k], lw=1.2, alpha=0.5, label=f"{label}: std of G_t")
            ax_adv.plot(u, [float(r["adv_std"]) for r in ups], color=viz.SERIES[k], lw=2, label=f"{label}: std of advantage")
            ax_ent.plot(u, [float(r["entropy"]) for r in ups], color=viz.SERIES[k], lw=1.5, label=label)
    ax_adv.set_xlabel("update (batch of 10 episodes)")
    ax_adv.set_ylabel("standard deviation over the batch")
    ax_adv.set_title("Pong-v5: spread of the return G_t vs the advantage used in the gradient")
    ax_adv.legend(fontsize=8)
    fig_adv.savefig(os.path.join(args.out, "compare_advantage_std.png"))
    ax_ent.set_xlabel("update (batch of 10 episodes)")
    ax_ent.set_ylabel("mean policy entropy (nats, max ln 2 = 0.693)")
    ax_ent.set_title("Pong-v5: policy entropy during training")
    ax_ent.legend(fontsize=8)
    fig_ent.savefig(os.path.join(args.out, "compare_entropy.png"))
    plt.close("all")
    viz.compare_curves(curves_ep, os.path.join(args.out, "compare_episodes.png"),
                       f"Pong-v5: {args.window}-episode moving-average reward vs training episode", window=args.window)
    for fname, data, xl in [("compare_env_steps.png", curves_steps, "environment steps (frames / 4)"),
                            ("compare_wall_time.png", curves_time, "training wall-clock hours")]:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for k, (label, (x, ma)) in enumerate(data.items()):
            ax.plot(x, ma, color=viz.SERIES[k], lw=2, label=label)
        ax.set_xlabel(xl)
        ax.set_ylabel(f"{args.window}-episode moving-average reward")
        ax.set_title(f"Pong-v5: moving-average reward vs {xl}")
        ax.legend()
        fig.savefig(os.path.join(args.out, fname))
        plt.close(fig)
    # seed groups: mean +- std over seeds of the moving average, plus a summary per group
    if args.group:
        gcurves = OrderedDict()
        gsummary = {}
        for spec in args.group:
            label, pattern = spec.split("=", 1)
            dirs = sorted(glob.glob(pattern))
            rew = [np.array([float(r["reward"]) for r in read_csv(os.path.join(d, "episodes.csv"))]) for d in dirs]
            rew = [r for r in rew if len(r) >= args.window]
            if not rew:
                continue
            gcurves[label] = rew
            finals = [float(moving_average(r, args.window)[-1]) for r in rew]
            bests = [float(moving_average(r, args.window).max()) for r in rew]
            ev = {}
            for tag in ["sampled", "greedy"]:
                ms = [json.load(open(os.path.join(d, f"eval_summary_{tag}.json")))["mean"] for d in dirs if os.path.exists(os.path.join(d, f"eval_summary_{tag}.json"))]
                sds = [json.load(open(os.path.join(d, f"eval_summary_{tag}.json")))["std"] for d in dirs if os.path.exists(os.path.join(d, f"eval_summary_{tag}.json"))]
                if ms:
                    ev[tag] = {"mean_per_seed": ms, "std_per_seed": sds, "mean_over_seeds": float(np.mean(ms)), "std_over_seeds": float(np.std(ms))}
            gsummary[label] = {"runs": dirs, "episodes_per_seed": [int(len(r)) for r in rew], "final_ma100_per_seed": finals,
                               "final_ma100_mean": float(np.mean(finals)), "final_ma100_std": float(np.std(finals)),
                               "best_ma100_per_seed": bests, "eval": ev}
        if gcurves:
            viz.compare_curves(gcurves, os.path.join(args.out, f"compare_{args.group_name}.png"),
                               f"Pong-v5: {args.window}-episode moving-average reward, {args.group_title}", window=args.window)
            with open(os.path.join(args.out, f"{args.group_name}.json"), "w") as f:
                json.dump(gsummary, f, indent=2)
            summary[args.group_name] = gsummary
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
