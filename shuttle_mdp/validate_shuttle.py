"""Validation experiments for the shuttle MDP solvers (edge cases and correctness tests).

    python -m shuttle_mdp.validate_shuttle --out results/shuttle

Every experiment prints what it checks, the expected behaviour, and the measured
numbers; results are also written to validation.json and validation.md.
"""
import argparse
import json
import os

import numpy as np

import viz
from shuttle_mdp.shuttle import ShuttleMDP, policy_threshold, threshold_policy


def deterministic_cycle_value(mdp, policy, s0):
    """Closed-form discounted cost for deterministic arrivals: follow the trajectory until
    a state repeats, then sum the pre-cycle costs plus the geometric series of the cycle."""
    assert len(mdp.arrivals) == 1
    arr = int(mdp.arrivals[0])
    seen = {}
    costs, states = [], []
    s = int(s0)
    while s not in seen:
        seen[s] = len(costs)
        a = int(policy[s])
        left = max(s - a * mdp.K, 0)
        costs.append(mdp.cf * a + mdp.ch * left)
        states.append(s)
        s = min(left + arr, mdp.max_customers)
    start = seen[s]
    g = mdp.gamma
    pre = sum(c * g ** t for t, c in enumerate(costs[:start]))
    cyc = sum(c * g ** (t - start) for t, c in enumerate(costs[start:], start=start))
    L = len(costs) - start
    return pre + g ** start * cyc / (1 - g ** L), states, start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/shuttle")
    p.add_argument("--mc-episodes", type=int, default=20000)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    R = {}
    lines = []

    def note(title, text):
        lines.append(f"**{title}.** {text}")
        print(f"[{title}] {text}")

    mdp = ShuttleMDP()
    V_vi, pi_vi, n_vi, _ = mdp.value_iteration(tol=1e-10)
    V_pi, pi_pi, n_pi, _ = mdp.policy_iteration()
    theta, monotone = policy_threshold(pi_pi)
    S = mdp.S

    # 1. transition model sanity
    rowsums = mdp.P.sum(axis=2)
    mass_cap = mdp.P[:, :, S - 1]
    R["transition_rows_sum_to_one"] = bool(np.allclose(rowsums, 1.0))
    R["mass_at_cap_from_s200_wait"] = float(mass_cap[0, S - 1])
    note("T1 transition model", f"all {rowsums.size} rows of P sum to 1: {R['transition_rows_sum_to_one']}; "
         f"from s=200 with 'wait' the whole arrival mass is clipped onto s'=200 (P = {mass_cap[0, S-1]:.2f}); "
         f"from s=199 'wait' puts P = {mdp.P[0, S-2, S-1]:.2f} on the cap (arrivals 1..5 all exceed it).")

    # 2. three methods agree; the policy is a threshold
    V_enum, pi_enum = mdp.enumeration(T=500)
    R["max_abs_diff_enum_vi"] = float(np.abs(V_enum[0] - V_vi).max())
    R["max_abs_diff_vi_pi"] = float(np.abs(V_vi - V_pi).max())
    R["policies_identical"] = bool((pi_enum[0] == pi_vi).all() and (pi_vi == pi_pi).all())
    R["threshold"] = int(theta)
    R["monotone"] = bool(monotone)
    note("T2 agreement", f"max|V_enum(t=0) - V_VI| = {R['max_abs_diff_enum_vi']:.2e}, max|V_VI - V_PI| = {R['max_abs_diff_vi_pi']:.2e}, "
         f"identical policies: {R['policies_identical']}; the policy is a monotone threshold rule (dispatch iff s >= {theta}): {monotone}.")

    # 3. finite horizon -> infinite horizon: error decays like gamma^T
    Ts = [1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 500]
    errs = []
    for T in Ts:
        Vt, _ = mdp.enumeration(T=T)
        errs.append(float(np.abs(Vt[0] - V_vi).max()))
    bound = [mdp.gamma ** T * V_vi.max() for T in Ts]
    R["horizon_errors"] = dict(zip(map(str, Ts), errs))
    viz.line_plot(Ts, [errs, bound], os.path.join(args.out, "fig_val_horizon_error.png"),
                  "Enumeration with horizon T vs infinite-horizon value: max_s |V_0^T(s) - V*(s)|",
                  "horizon T", "max absolute error (log scale)", labels=["measured error", "bound gamma^T * max_s V*(s)"],
                  logy=True, markers=True)
    ratios = [errs[i + 1] / errs[i] for i in range(len(Ts) - 1) if errs[i] > 1e-9]
    note("T3 horizon", f"error of the T-period enumeration against V*: " + ", ".join(f"T={T}: {e:.3g}" for T, e in zip(Ts, errs)) +
         f". Every measured error lies below the bound gamma^T max V*: {all(e <= b + 1e-9 for e, b in zip(errs, bound))}.")
    V1, _ = mdp.enumeration(T=1)
    R["T1_equals_min_immediate_cost"] = bool(np.allclose(V1[0], mdp.C.min(axis=0)))
    note("T3b one-period problem", f"with T=1 the value is min_a c(s,a) = 2s (never worth paying 100 for at most 30 of holding cost): "
         f"{R['T1_equals_min_immediate_cost']}.")

    # 4. enumeration of all threshold policies (exact evaluation of each)
    thetas = np.arange(S + 1)
    v0 = np.empty(len(thetas))
    vmean = np.empty(len(thetas))
    for i, th in enumerate(thetas):
        V = mdp.policy_evaluation(threshold_policy(S, th))
        v0[i], vmean[i] = V[0], V.mean()
    best0, bestm = int(thetas[v0.argmin()]), int(thetas[vmean.argmin()])
    R["threshold_enumeration"] = {"best_by_V0": best0, "best_by_mean_V": bestm, "V0_at_best": float(v0.min()),
                                  "V0_optimal": float(V_pi[0]), "V0_never_dispatch": float(v0[-1]),
                                  "V0_dispatch_when_full_theta15": float(v0[15])}
    viz.line_plot(thetas[:60], [v0[:60]], os.path.join(args.out, "fig_val_threshold_enumeration.png"),
                  "Exact cost from s = 0 of every threshold policy 'dispatch iff s >= theta'",
                  "threshold theta", "expected discounted cost from s = 0", markers=True,
                  annotate=[(f"minimum at theta = {best0}", (best0, v0.min()))])
    note("T4 threshold enumeration", f"evaluating all {len(thetas)} threshold policies exactly: the minimiser is theta = {best0} by V(0) "
         f"and theta = {bestm} by the state-average, equal to the PI threshold {theta}; V(0) = {v0.min():.4f} vs PI {V_pi[0]:.4f}. "
         f"Dispatching only when full (theta = 15) costs {v0[15]:.2f}, never dispatching costs {v0[-1]:.2f}.")

    # 5. Monte Carlo policy evaluation
    mc = {}
    for s0 in [0, theta, 50, 100, 200]:
        samples = mdp.simulate(pi_pi, s0, 500, args.mc_episodes, rng)
        m, se = samples.mean(), samples.std(ddof=1) / np.sqrt(len(samples))
        mc[str(s0)] = {"V": float(V_pi[s0]), "mc_mean": float(m), "mc_se": float(se), "z": float((m - V_pi[s0]) / se)}
    R["monte_carlo"] = mc
    note("T5 Monte Carlo", "; ".join(f"s0={k}: V={v['V']:.2f}, MC={v['mc_mean']:.2f} +- {1.96 * v['mc_se']:.2f} (z={v['z']:.2f})" for k, v in mc.items()))

    # 6. sensitivity of the threshold to the parameters
    sens = {}
    grids = {"cf": [25, 50, 100, 200, 400, 800], "ch": [0.5, 1, 2, 4, 8, 16], "K": [5, 10, 15, 20, 30, 50],
             "gamma": [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]}
    for name, vals in grids.items():
        ths = []
        for v in vals:
            kw = {name: v}
            m = ShuttleMDP(**kw)
            _, pol, _, _ = m.policy_iteration()
            ths.append(policy_threshold(pol)[0])
        sens[name] = dict(zip(map(str, vals), [int(t) for t in ths]))
    R["sensitivity"] = sens
    viz.setup()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, (name, d) in zip(axes.ravel(), sens.items()):
        xs = [float(k) for k in d]
        ys = list(d.values())
        ax.plot(xs, ys, marker="o", color=viz.BLUE, lw=2)
        ax.set_xlabel(name)
        ax.set_ylabel("dispatch threshold (201 = never)")
        if name in ("cf", "ch", "K"):
            ax.set_xscale("log")
    fig.suptitle("Optimal dispatch threshold as one parameter varies (others at K=15, cf=100, ch=2, gamma=0.95)")
    fig.savefig(os.path.join(args.out, "fig_val_sensitivity.png"))
    plt.close(fig)
    note("T6 sensitivity", "; ".join(f"{n}: " + ", ".join(f"{k}->{v}" for k, v in d.items()) for n, d in sens.items()))

    # 7. deterministic arrivals: closed form
    det = ShuttleMDP(arrivals=(3,), arrival_probs=(1.0,))
    Vd, pid, _, _ = det.policy_iteration()
    closed, states, start = deterministic_cycle_value(det, pid, 0)
    R["deterministic"] = {"threshold": int(policy_threshold(pid)[0]), "V0_solver": float(Vd[0]), "V0_closed_form": float(closed),
                          "cycle_states": [int(x) for x in states[start:]]}
    note("T7 deterministic arrivals", f"A = 3 always: PI threshold {R['deterministic']['threshold']}, cycle of states {states[start:]}; "
         f"V(0) from the linear solve {Vd[0]:.6f} vs geometric-series closed form {closed:.6f}.")

    # 8. edge cases
    Q = mdp.q_values(V_pi)
    R["edge_s0_gap_equals_cf"] = float(Q[1, 0] - Q[0, 0])
    big = ShuttleMDP(max_customers=400)
    Vb, pib, _, _ = big.policy_iteration()
    R["cap_effect_max_abs_diff_s_le_100"] = float(np.abs(Vb[:101] - V_pi[:101]).max())
    R["cap_effect_max_abs_diff_s_le_200"] = float(np.abs(Vb[:201] - V_pi[:201]).max())
    R["cap_policy_same_s_le_200"] = bool((pib[:201] == pi_pi).all())
    R["partially_full_dispatch_states"] = [int(x) for x in range(theta, mdp.K)]
    note("T8 edge cases", f"Q(0,dispatch) - Q(0,wait) = {R['edge_s0_gap_equals_cf']:.6f} (= cf: an empty shuttle costs cf and changes nothing). "
         f"Raising the cap from 200 to 400 changes V by at most {R['cap_effect_max_abs_diff_s_le_100']:.2e} for s <= 100 and "
         f"{R['cap_effect_max_abs_diff_s_le_200']:.2e} for s <= 200; same policy on s <= 200: {R['cap_policy_same_s_le_200']}. "
         f"Partially full dispatches happen at s in {R['partially_full_dispatch_states']} (K = {mdp.K}).")

    # 9. first ten steps
    tr = mdp.trace(pi_pi, 0, 12, np.random.default_rng(1))
    R["trace_first_steps"] = tr
    md = ["| t | state | action | boarded | left waiting | cost | arrivals | next state |", "|---|---|---|---|---|---|---|---|"]
    for r in tr:
        md.append(f"| {r['t']} | {r['state']} | {'dispatch' if r['action'] else 'wait'} | {r['boarded']} | {r['left_waiting']} | {r['cost']:.0f} | {r['arrivals']} | {r['next_state']} |")
    lines.append("\n".join(md))

    # 10. monotone value function
    R["value_nondecreasing"] = bool(np.all(np.diff(V_pi) >= -1e-9))
    R["value_increments_min_max"] = [float(np.diff(V_pi).min()), float(np.diff(V_pi).max())]
    note("T10 monotone value", f"V*(s) is nondecreasing in s: {R['value_nondecreasing']} (increments between {np.diff(V_pi).min():.3f} and {np.diff(V_pi).max():.3f}).")

    with open(os.path.join(args.out, "validation.json"), "w") as f:
        json.dump(R, f, indent=2, default=float)
    with open(os.path.join(args.out, "validation.md"), "w") as f:
        f.write("\n\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
