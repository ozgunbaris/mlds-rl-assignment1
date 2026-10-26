"""Solve the shuttle dispatch MDP three ways and produce the report figures.

    python -m shuttle_mdp.run_shuttle --out results/shuttle
"""
import argparse
import csv
import json
import os

import numpy as np

import viz
from shuttle_mdp.shuttle import ShuttleMDP, policy_threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/shuttle")
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--max-customers", type=int, default=200)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    mdp = ShuttleMDP(K=15, cf=100, ch=2, arrivals=(1, 2, 3, 4, 5), max_customers=args.max_customers, gamma=args.gamma)
    s = np.arange(mdp.S)

    # a) enumeration over the finite horizon
    V_enum, pi_enum = mdp.enumeration(T=args.T)
    # b) value iteration
    V_vi, pi_vi, n_vi, residuals = mdp.value_iteration(tol=1e-10)
    # c) policy iteration
    V_pi, pi_pi, n_pi, pi_history = mdp.policy_iteration()

    theta, monotone = policy_threshold(pi_pi)
    Q = mdp.q_values(V_pi)

    # ---------------------------------------------------------------- figures
    viz.line_plot(s, [V_enum[0]], os.path.join(args.out, "fig_a_enumeration_value_T500.png"),
                  f"(a) Enumeration, T = {args.T}: optimal value function at time 0",
                  "customers waiting at time 0", "expected discounted cost from time 0")
    stages = [0, args.T - 100, args.T - 20, args.T - 5, args.T - 1]
    viz.line_plot(s, [V_enum[t] for t in stages], os.path.join(args.out, "fig_a_enumeration_stages.png"),
                  "(a) Enumeration: value function at several stages t of the 500-period problem",
                  "customers waiting", "expected discounted cost-to-go", labels=[f"t = {t}" for t in stages])
    viz.line_plot(s, [V_vi], os.path.join(args.out, "fig_b_value_iteration_value.png"),
                  f"(b) Value iteration (infinite horizon, {n_vi} sweeps): optimal value function",
                  "customers waiting", "expected discounted cost")
    viz.line_plot(np.arange(1, len(residuals) + 1), [residuals], os.path.join(args.out, "fig_b_vi_convergence.png"),
                  "(b) Value iteration: sup-norm change per sweep (log scale)", "sweep", "max_s |V_{n+1}(s) - V_n(s)|",
                  logy=True)
    viz.line_plot(s, [pi_pi], os.path.join(args.out, "fig_c_policy_iteration_policy.png"),
                  f"(c) Policy iteration ({n_pi} iterations): optimal policy, dispatch iff s >= {theta}",
                  "customers waiting", "action (1 = dispatch, 0 = wait)", step=True,
                  annotate=[(f"threshold = {theta}", (theta, 0.5))])
    viz.line_plot(s, [Q[1] - Q[0]], os.path.join(args.out, "fig_c_q_gap.png"),
                  "(c) Q(s, dispatch) - Q(s, wait) under the optimal value function",
                  "customers waiting", "cost difference (negative = dispatch is better)",
                  annotate=[(f"sign change at s = {theta}", (theta, 0.0))])
    viz.line_plot(s, [V_enum[0], V_vi, V_pi], os.path.join(args.out, "fig_value_comparison.png"),
                  "Value functions from the three methods (curves coincide)", "customers waiting",
                  "expected discounted cost", labels=["enumeration, t = 0", "value iteration", "policy iteration"])
    thresholds = [policy_threshold(pol)[0] for pol in pi_history]
    viz.line_plot(np.arange(len(thresholds)), [thresholds], os.path.join(args.out, "fig_c_pi_thresholds.png"),
                  "(c) Policy iteration: dispatch threshold of the policy after each iteration",
                  "policy iteration step (0 = initial never-dispatch policy)", "smallest state that dispatches (S+1 = never)",
                  markers=True)

    # ---------------------------------------------------------------- tables
    with open(os.path.join(args.out, "value_functions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s", "V_enum_t0", "V_value_iteration", "V_policy_iteration", "policy_enum_t0", "policy_vi", "policy_pi",
                    "Q_wait", "Q_dispatch"])
        for i in s:
            w.writerow([int(i), V_enum[0, i], V_vi[i], V_pi[i], int(pi_enum[0, i]), int(pi_vi[i]), int(pi_pi[i]), Q[0, i], Q[1, i]])
    summary = {
        "parameters": {"K": 15, "cf": 100, "ch": 2, "arrivals": "Uniform{1,...,5}", "max_customers": args.max_customers,
                       "gamma": args.gamma, "T_enumeration": args.T},
        "threshold_policy": {"dispatch_iff_s_ge": theta, "is_monotone_threshold": monotone},
        "value_iteration": {"sweeps": int(n_vi), "final_residual": float(residuals[-1]),
                            "error_bound_gamma_over_1_minus_gamma_times_residual": float(residuals[-1] * args.gamma / (1 - args.gamma))},
        "policy_iteration": {"iterations": int(n_pi), "thresholds_per_iteration": [int(t) for t in thresholds]},
        "agreement": {"max_abs_V_enum0_minus_V_vi": float(np.abs(V_enum[0] - V_vi).max()),
                      "max_abs_V_vi_minus_V_pi": float(np.abs(V_vi - V_pi).max()),
                      "policies_identical": bool((pi_enum[0] == pi_vi).all() and (pi_vi == pi_pi).all())},
        "values": {str(int(i)): float(V_pi[i]) for i in [0, 5, 10, 12, 13, 14, 15, 20, 30, 50, 100, 150, 200]},
        "enumeration_stage_policies_equal_to_stationary": {str(t): bool((pi_enum[t] == pi_pi).all()) for t in [0, 100, 250, 400, 450, 480, 495, 499]},
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
