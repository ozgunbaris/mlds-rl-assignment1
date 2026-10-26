"""Validation experiments for REINFORCE on Pong-v5.

    python -m policy_gradient.validate_pong --policy results/pong/none_lr1e-3/checkpoint.pt \
        --value-ckpt results/pong/value_lr1e-3/checkpoint.pt --out results/pong/validation

P1  preprocessing: figure of raw frame, preprocessed frame and difference frame; pixel counts
P2  action semantics: RIGHT (2) moves the agent's paddle up, LEFT (3) moves it down (measured)
P3  random RIGHT/LEFT policy level and reward structure of complete games
P4  discounted return along one episode (gamma = 0.99): sawtooth at every point
P5  behaviour probe: P(up) when the ball is above vs below the paddle, random vs trained policy
P6  variance of the per-episode gradient with no baseline, constant baseline and value baseline
P7  first 10 recorded steps of an episode (inputs, action probabilities, rewards)
"""
import argparse
import json
import os

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import viz
from policy_gradient.common import discounted_returns, set_seed
from policy_gradient.pong_reinforce import ACTIONS, ENV_ID, INPUT_DIM, PongPolicy, PongValue, preprocess

gym.register_envs(ale_py)

AGENT_COLS = (70, 72)   # agent paddle columns in the 80x80 image
BALL_COLS = (10, 70)    # columns where only the ball can appear


def paddle_and_ball(x80):
    """Row centres of the agent paddle and of the ball in a preprocessed 80x80 frame (None if absent)."""
    pad = np.argwhere(x80[:, AGENT_COLS[0]:AGENT_COLS[1]] > 0)
    ball = np.argwhere(x80[:, BALL_COLS[0]:BALL_COLS[1]] > 0)
    py = float(pad[:, 0].mean()) if len(pad) else None
    by = float(ball[:, 0].mean()) if len(ball) else None
    return py, by


def run_episode(env, policy, seed=None, greedy=False, record=False):
    obs, _ = env.reset(seed=seed)
    prev = np.zeros((80, 80), dtype=np.float32)
    cur = preprocess(obs)
    xs, acts, rews, probs, frames = [], [], [], [], []
    done = False
    while not done:
        x = (cur - prev).reshape(INPUT_DIM)
        with torch.no_grad():
            logits = policy(torch.from_numpy(x)[None])
            pr = F.softmax(logits, 1)[0].numpy()
        a = int(logits.argmax()) if greedy else int(np.random.choice(2, p=pr))
        obs, r, term, trunc, _ = env.step(int(ACTIONS[a]))
        if record:
            xs.append(x.astype(np.int8))
            frames.append(cur.copy())
        acts.append(a)
        rews.append(float(r))
        probs.append(float(pr[0]))
        prev, cur = cur, preprocess(obs)
        done = term or trunc
    return np.array(xs), np.array(acts), np.array(rews), np.array(probs), frames


def episode_gradient(policy, xs, acts, adv, chunk=4096):
    policy.zero_grad()
    T = len(acts)
    for lo in range(0, T, chunk):
        hi = min(T, lo + chunk)
        logp = F.log_softmax(policy(torch.from_numpy(xs[lo:hi]).float()), 1)
        lp = logp.gather(1, torch.as_tensor(acts[lo:hi])[:, None]).squeeze(1)
        (-(lp * torch.as_tensor(adv[lo:hi], dtype=torch.float32)).sum() / T).backward()
    return torch.cat([p.grad.reshape(-1) for p in policy.parameters()]).numpy().copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="checkpoint of the trained policy (Part 1)")
    p.add_argument("--value-ckpt", default=None, help="checkpoint with a value network (Part 2)")
    p.add_argument("--out", default="results/pong/validation")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--random-episodes", type=int, default=20)
    p.add_argument("--probe-episodes", type=int, default=5)
    p.add_argument("--grad-episodes", type=int, default=12)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(2)
    set_seed(0)
    R, lines = {}, []

    def note(title, text):
        lines.append(f"**{title}.** {text}")
        print(f"[{title}] {text}", flush=True)

    env = gym.make(ENV_ID)
    rand_policy = PongPolicy()  # untrained network == (almost) uniform policy
    ck = torch.load(args.policy, map_location="cpu", weights_only=False)
    trained = PongPolicy()
    trained.load_state_dict(ck["policy"])
    trained.eval()
    R["trained_checkpoint"] = {"path": args.policy, "episodes": ck["state"]["episodes"], "updates": ck["state"]["updates"]}

    # P1 preprocessing figure
    obs, _ = env.reset(seed=0)
    for _ in range(60):
        obs, *_ = env.step(0)
    prev = preprocess(obs)
    obs, *_ = env.step(0)
    cur = preprocess(obs)
    import matplotlib.pyplot as plt
    viz.setup()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    axes[0].imshow(obs)
    axes[0].set_title("raw frame 210x160x3")
    axes[1].imshow(cur, cmap="gray_r")
    axes[1].set_title(f"preprocess(): 80x80, {int(cur.sum())} active pixels")
    axes[2].imshow(cur - prev, cmap="bwr", vmin=-1, vmax=1)
    axes[2].set_title("network input: current - previous")
    for ax in axes:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(os.path.join(args.out, "fig_val_pong_preprocess.png"))
    plt.close(fig)
    R["preprocess"] = {"active_pixels": int(cur.sum()), "diff_nonzero": int((cur != prev).sum()), "unique_values": sorted(set(np.unique(cur).tolist()))}
    note("P1 preprocessing", f"a mid-rally frame keeps {int(cur.sum())} of 6400 pixels (two paddles and the ball), values are {R['preprocess']['unique_values']}, "
         f"and the difference frame has {R['preprocess']['diff_nonzero']} nonzero entries (moving objects only).")

    # P2 action semantics
    def paddle_row_after(action, n=15):
        env.reset(seed=1)
        for _ in range(40):
            obs, *_ = env.step(0)
        y0, _ = paddle_and_ball(preprocess(obs))
        for _ in range(n):
            obs, *_ = env.step(action)
        y1, _ = paddle_and_ball(preprocess(obs))
        return y0, y1
    r0, r1 = paddle_row_after(2)
    l0, l1 = paddle_row_after(3)
    R["action_semantics"] = {"RIGHT_rows": [r0, r1], "LEFT_rows": [l0, l1]}
    note("P2 action semantics", f"holding RIGHT (2) for 15 steps moves the agent paddle from row {r0:.1f} to {r1:.1f}; holding LEFT (3) moves it from row {l0:.1f} to {l1:.1f} "
         f"(row 0 is the top): RIGHT = up, LEFT = down.")

    # P3 random policy level + reward structure
    rr, lens, ok = [], [], True
    for i in range(args.random_episodes):
        _, _, rew, _, _ = run_episode(env, rand_policy, seed=100 + i)
        wins, losses = int((rew > 0).sum()), int((rew < 0).sum())
        ok &= (max(wins, losses) == 21) and rew[-1] != 0 and set(np.unique(rew)).issubset({-1.0, 0.0, 1.0})
        rr.append(rew.sum())
        lens.append(len(rew))
    rr = np.array(rr)
    R["random_policy"] = {"episodes": len(rr), "mean": float(rr.mean()), "std": float(rr.std()), "max": float(rr.max()),
                          "mean_length": float(np.mean(lens)), "structure_ok": bool(ok)}
    note("P3 random policy", f"{len(rr)} games with uniformly random RIGHT/LEFT: mean score {rr.mean():.2f}, std {rr.std():.2f}, best {rr.max():.0f}, "
         f"mean length {np.mean(lens):.0f} steps. Every game ended when one side reached 21 and rewards were in {{-1, 0, +1}}: {ok}.")

    # P4 discounted return along an episode
    xs, acts, rew, probs, _ = run_episode(env, trained, seed=5, record=True)
    G = discounted_returns(rew, args.gamma)
    t = np.arange(len(rew))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(t, G, color=viz.BLUE, lw=1.2, label="G_t (gamma = 0.99)")
    pts = np.flatnonzero(rew != 0)
    ax.scatter(pts, rew[pts], color=np.where(rew[pts] > 0, viz.AQUA, viz.RED), s=14, zorder=3, label="points (+1 won, -1 lost)")
    ax.set_xlabel("step within the episode")
    ax.set_ylabel("discounted return")
    ax.set_title(f"Pong-v5: discounted return along one trained-policy episode (score {rew.sum():+.0f})")
    ax.legend()
    fig.savefig(os.path.join(args.out, "fig_val_pong_returns.png"))
    plt.close(fig)
    first_loss = int(np.flatnonzero(rew < 0)[0]) if (rew < 0).any() else None
    R["returns"] = {"episode_score": float(rew.sum()), "length": int(len(rew)), "G_min": float(G.min()), "G_max": float(G.max()),
                    "G_at_first_loss": float(G[first_loss]) if first_loss is not None else None,
                    "G_10_steps_before_first_loss": float(G[first_loss - 10]) if first_loss and first_loss >= 10 else None,
                    "expected_ratio_0.99^10": 0.99 ** 10}
    note("P4 discounted return", f"over an episode of {len(rew)} steps G_t ranges in [{G.min():.2f}, {G.max():.2f}]; at the step of the first lost point G = {R['returns']['G_at_first_loss']}, "
         f"ten steps earlier G = {R['returns']['G_10_steps_before_first_loss']} (ratio should be close to 0.99^10 = {0.99**10:.3f} when no other point is close).")

    # P5 behaviour probe
    def probe(policy, n_eps, seed0):
        up_above, up_below = [], []
        for i in range(n_eps):
            _, acts_i, _, probs_i, frames = run_episode(env, policy, seed=seed0 + i, record=True)
            for pr_up, fr in zip(probs_i, frames):
                py, by = paddle_and_ball(fr)
                if py is None or by is None:
                    continue
                if by < py - 2:
                    up_above.append(pr_up)
                elif by > py + 2:
                    up_below.append(pr_up)
        return {"P_up_given_ball_above": float(np.mean(up_above)), "P_up_given_ball_below": float(np.mean(up_below)),
                "n_above": len(up_above), "n_below": len(up_below)}
    R["probe"] = {"trained": probe(trained, args.probe_episodes, 200), "random": probe(rand_policy, 2, 300)}
    for k, v in R["probe"].items():
        note(f"P5 behaviour probe ({k} policy)", f"P(up | ball above paddle) = {v['P_up_given_ball_above']:.3f} over {v['n_above']} frames, "
             f"P(up | ball below paddle) = {v['P_up_given_ball_below']:.3f} over {v['n_below']} frames; difference {v['P_up_given_ball_above'] - v['P_up_given_ball_below']:+.3f}.")

    # P6 gradient variance
    value = None
    if args.value_ckpt and os.path.exists(args.value_ckpt):
        vck = torch.load(args.value_ckpt, map_location="cpu", weights_only=False)
        if vck.get("value") is not None:
            value = PongValue()
            value.load_state_dict(vck["value"])
            value.eval()
            vpolicy = PongPolicy()
            vpolicy.load_state_dict(vck["policy"])
            vpolicy.eval()
    if value is not None:
        eps = [run_episode(env, vpolicy, seed=400 + i, record=True) for i in range(args.grad_episodes)]
        Gs = [discounted_returns(e[2], args.gamma) for e in eps]
        const = float(np.mean(np.concatenate(Gs)))
        grads = {"none": [], "mean": [], "value": []}
        for (xs_i, acts_i, _, _, _), G_i in zip(eps, Gs):
            grads["none"].append(episode_gradient(vpolicy, xs_i, acts_i, G_i))
            grads["mean"].append(episode_gradient(vpolicy, xs_i, acts_i, G_i - const))
            with torch.no_grad():
                V = np.concatenate([value(torch.from_numpy(xs_i[lo:lo + 4096]).float()).numpy() for lo in range(0, len(xs_i), 4096)])
            grads["value"].append(episode_gradient(vpolicy, xs_i, acts_i, G_i - V))
        gnone = np.mean(grads["none"], 0)
        gv = {}
        for k, gl in grads.items():
            gl = np.stack(gl)
            m = gl.mean(0)
            tr = float(((gl - m) ** 2).sum(1).mean())
            gv[k] = {"trace_cov": tr, "mean_norm": float(np.linalg.norm(m)), "snr_of_M_episode_average": float(np.linalg.norm(m) / np.sqrt(tr / len(gl))),
                     "cos_with_none": float(np.dot(m, gnone) / (np.linalg.norm(m) * np.linalg.norm(gnone) + 1e-12))}
        gv["episodes"] = args.grad_episodes
        gv["mean_score"] = float(np.mean([e[2].sum() for e in eps]))
        gv["value_checkpoint"] = args.value_ckpt
        R["gradient_variance"] = gv
        note("P6 gradient variance", f"{args.grad_episodes} episodes of the Part-2 policy (mean score {gv['mean_score']:.1f}): trace of the per-episode gradient covariance "
             f"= {gv['none']['trace_cov']:.4g} with no baseline, {gv['mean']['trace_cov']:.4g} with the constant baseline ({gv['none']['trace_cov']/gv['mean']['trace_cov']:.2f}x smaller), "
             f"{gv['value']['trace_cov']:.4g} with the learned value baseline ({gv['none']['trace_cov']/gv['value']['trace_cov']:.2f}x smaller); "
             f"cosine of the mean gradients with the no-baseline mean: {gv['mean']['cos_with_none']:.3f} and {gv['value']['cos_with_none']:.3f}; "
             f"signal-to-noise ratio of the {args.grad_episodes}-episode average gradient: none {gv['none']['snr_of_M_episode_average']:.2f}, "
             f"constant {gv['mean']['snr_of_M_episode_average']:.2f}, value {gv['value']['snr_of_M_episode_average']:.2f}.")

    # P7 first ten steps
    md = ["| step | nonzero input pixels | P(RIGHT = up) | action | reward |", "|---|---|---|---|---|"]
    for i in range(10):
        md.append(f"| {i} | {int((xs[i] != 0).sum())} | {probs[i]:.3f} | {'RIGHT (up)' if acts[i] == 0 else 'LEFT (down)'} | {rew[i]:.0f} |")
    lines.append("\n".join(md))
    R["trace"] = {"nonzero": [int((xs[i] != 0).sum()) for i in range(10)], "p_up": [float(probs[i]) for i in range(10)], "actions": [int(a) for a in acts[:10]]}
    note("P7 trace", "the first input of an episode is the full first frame (both paddles, no previous frame), later inputs only contain the moving pixels; see the table.")

    env.close()
    with open(os.path.join(args.out, "validation.json"), "w") as f:
        json.dump(R, f, indent=2, default=float)
    with open(os.path.join(args.out, "validation.md"), "w") as f:
        f.write("\n\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
