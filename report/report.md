---
title: "MLDS Reinforcement Learning, Assignment 1"
author: "GitHub username: ozgunbaris"
date: "October 26, 2026"
geometry: margin=0.9in
fontsize: 11pt
colorlinks: true
---

# Repository and how to run the code

GitHub username: **ozgunbaris**. Repository: **mlds-rl-assignment1**, <https://github.com/ozgunbaris/mlds-rl-assignment1>.

All code is in the repository. `README.md` lists every command. Short version (Python 3.12, run from the repository root):

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m shuttle_mdp.run_shuttle --out results/shuttle            # Part 2 figures
python -m shuttle_mdp.validate_shuttle --out results/shuttle       # Part 2 validation experiments
python -m policy_gradient.cartpole_reinforce --out results/cartpole/none_seed0  --baseline none  --lr 0.01
python -m policy_gradient.cartpole_reinforce --out results/cartpole/value_seed0 --baseline value --lr 0.01
python -m policy_gradient.pong_reinforce --out results/pong/none_lr1e-3  --baseline none  --lr 1e-3 --time-limit-hours 5
python -m policy_gradient.pong_reinforce --out results/pong/value_lr1e-3 --baseline value --lr 1e-3 --time-limit-hours 5
python -m policy_gradient.pong_reinforce --eval results/pong/none_lr1e-3/checkpoint.pt --eval-episodes 500 --out results/pong/none_lr1e-3
python -m pytest tests -q                                           # 25 unit and property tests
```

`scripts/run_cartpole_sweeps.sh` and `scripts/run_pong.sh` contain the exact loops used for every figure. Each `results/*` folder holds the logs (`episodes.csv`, `updates.csv`), checkpoints, figures and `validation.md` with the numbers quoted below.

# Part 2: Value and policy iteration (shuttle dispatch)

## Model

The state $s \in \{0, 1, \dots, 200\}$ is the number of customers waiting at the start of a period. The action $a \in \{0, 1\}$ is 1 when a shuttle of capacity $K = 15$ is dispatched. The one-period cost is

$$c(s, a) = c_f\, a + c_h \max(s - aK, 0), \qquad c_f = 100,\; c_h = 2,$$

so the holding cost is charged on the customers left behind after the dispatch decision. Arrivals $A_t \sim \mathrm{Unif}\{1, \dots, 5\}$ join after the dispatch and the next state is $s' = \min(\max(s - aK, 0) + A_t, 200)$. The objective is the expected discounted cost with $\gamma = 0.95$; all three solvers minimise cost. The transition matrices $P_a$ (2 x 201 x 201) and cost vectors are built once in `shuttle_mdp/shuttle.py` and shared by the three methods.

Assumptions beyond the statement: (1) the cap of 200 is enforced by clipping, so arrivals that would exceed 200 are lost; (2) the finite-horizon problem has zero terminal cost, $V_T \equiv 0$; (3) ties between actions are broken toward "wait".

## (a) Enumeration with T = 500

Enumeration means exhaustive backward induction: for $t = 499, \dots, 0$ and every state the two action values $c(s,a) + \gamma \sum_{s'} P_a(s,s') V_{t+1}(s')$ are enumerated and the minimum is stored. The value function at time 0 is in Figure 1. $V_0$ is convex-looking and increasing, from $V_0(0) = 576.67$ to $V_0(200) = 3821.59$, and is flat between $s = 13$ and $s = 15$ ($V_0 = 676.67$ there) because every one of those states dispatches, empties the station and pays the same 100.

![(a) Optimal value function at time 0 from enumeration over 500 periods.](../results/shuttle/fig_a_enumeration_value_T500.png){ width=78% }

Figure 2 shows the same backward induction at several stages. The curve at $t = 499$ is the one-period cost $2s$, the curve at $t = 495$ still bends only slightly, and the curves for $t \le 400$ are indistinguishable from $t = 0$: with $\gamma = 0.95$ the influence of the horizon decays as $0.95^{500 - t}$.

![(a) Value function at stages t = 0, 400, 480, 495 and 499 of the 500-period problem.](../results/shuttle/fig_a_enumeration_stages.png){ width=78% }

## (b) Value iteration

Value iteration applies $V \leftarrow \min_a [c_a + \gamma P_a V]$ from $V = 0$ until the sup-norm change is below $10^{-10}$. It stopped after 519 sweeps with a final change of $9.9 \times 10^{-11}$, which bounds the distance to $V^*$ by $\gamma/(1-\gamma)$ times that change, $1.9 \times 10^{-9}$. Figure 3 shows $V^*$ and Figure 4 the geometric convergence (slope $\log 0.95$ per sweep).

![(b) Optimal value function from value iteration.](../results/shuttle/fig_b_value_iteration_value.png){ width=78% }

![(b) Sup-norm change per value-iteration sweep (log scale).](../results/shuttle/fig_b_vi_convergence.png){ width=70% }

## (c) Policy iteration

Policy iteration starts from "never dispatch", evaluates each policy exactly by solving $(I - \gamma P_\pi) V = c_\pi$, and improves greedily, changing an action only when it is strictly better. It converged in 6 iterations; the dispatch threshold of the successive policies was 201 (never), 3, 26, 7, 14, 13, 13. The optimal policy (Figure 5) is a threshold rule: **dispatch if and only if $s \ge 13$**. Figure 6 plots $Q(s, \text{dispatch}) - Q(s, \text{wait})$, which crosses zero between $s = 12$ and $s = 13$.

![(c) Optimal policy from policy iteration.](../results/shuttle/fig_c_policy_iteration_policy.png){ width=78% }

![(c) Action-value gap under the optimal value function; negative means dispatching is cheaper.](../results/shuttle/fig_c_q_gap.png){ width=70% }

The threshold is below the capacity: with 13 or 14 customers the shuttle leaves with empty seats. Waiting one more period with 13 customers costs $2 \times 13 = 26$ in holding cost plus the extra cost of a longer queue later, while delaying the dispatch only saves $(1-\gamma) c_f = 5$ in discounting; the exact comparison is the curve in Figure 6.

## Agreement between the three methods

The three value functions coincide: $\max_s |V_0^{T=500}(s) - V_{VI}(s)| = 3.1 \times 10^{-9}$ and $\max_s |V_{VI}(s) - V_{PI}(s)| = 1.9 \times 10^{-9}$ (Figure 7), and the three policies are identical in every state. The stage policies of the enumeration equal the stationary policy for every $t \le 480$; only the last few stages differ (at $t = 495$ and $t = 499$ the remaining horizon is too short to recover the 100 dispatch cost, so those stages dispatch later or never). That end effect is expected and does not reach $t = 0$.

![Value functions of the three methods drawn on top of each other.](../results/shuttle/fig_value_comparison.png){ width=70% }

## Edge cases

- **Empty station ($s = 0$).** Dispatching an empty shuttle costs $c_f$ and changes nothing else, so $Q(0,1) - Q(0,0)$ must equal exactly 100. The computed gap is 100.000000 (test T8).
- **Fewer customers than seats ($s < K$).** $\max(s - K, 0)$ keeps the holding cost at zero and the next state at $A_t$. The solver finds that dispatching with 13 or 14 customers is optimal, so this branch is exercised by the optimal policy itself.
- **The cap at 200.** Rows of $P$ still sum to one (all 402 rows checked); from $s = 199$ and $s = 200$ with "wait" the whole arrival mass lands on 200. Raising the cap to 400 changes $V$ by at most $4.6 \times 10^{-13}$ on $s \le 200$ and leaves the policy unchanged, so the cap does not distort the answer: under the optimal policy the queue never climbs toward 200.
- **Ties in policy improvement.** The improvement step keeps the current action unless the alternative is strictly better by more than $10^{-9}$; this removes cycling between equal-valued actions.
- **Horizon effects.** The finite-horizon problem with $V_T = 0$ makes the last stages myopic (Figure 2). A tolerance of $10^{-10}$ in value iteration and $T = 500$ in enumeration were chosen so that these effects are below $10^{-8}$ at $t = 0$.
- **Monotone structure.** $V^*$ is nondecreasing in $s$ (every increment is between 0 and 24.85) and the policy is monotone (once dispatching is optimal it stays optimal for larger $s$); both are checked in the tests because a bug in the transition model typically breaks them first.

## Tests that validate correctness

`shuttle_mdp/validate_shuttle.py` runs the experiments below and writes `results/shuttle/validation.md`; `tests/test_shuttle.py` contains 14 pytest checks of the same facts with tolerances.

1. **Horizon test (T3).** The error $\max_s |V_0^T - V^*|$ of the $T$-period enumeration is 3420 for $T = 1$, 917 for $T = 10$, 52.2 for $T = 50$, 4.02 for $T = 100$, 0.024 for $T = 200$ and $3.1 \times 10^{-9}$ for $T = 500$. Every value is below the bound $\gamma^T \max_s V^*(s)$ (Figure 8). With $T = 1$ the value is exactly $2s$ and the policy never dispatches, as a hand calculation requires.
2. **Enumeration of all threshold policies (T4).** Every policy "dispatch iff $s \ge \theta$", $\theta = 0, \dots, 201$, was evaluated exactly. The minimiser is $\theta = 13$ both for $V(0)$ and for the state average, with $V(0) = 576.6702$, equal to the policy-iteration value. Dispatching only when the shuttle is full ($\theta = 15$) costs 597.36, never dispatching costs 2203.91 (Figure 9).
3. **Monte Carlo policy evaluation (T5).** 20,000 simulated 500-period trajectories of the optimal policy give mean discounted costs of $576.78 \pm 0.43$ from $s = 0$, $676.54 \pm 0.43$ from $s = 13$, $1001.63 \pm 0.43$ from $s = 50$, $1709.29 \pm 0.61$ from $s = 100$ and $3822.37 \pm 1.04$ from $s = 200$ (95% intervals). The Bellman solutions 576.67, 676.67, 1001.53, 1709.04 and 3821.59 all lie inside; the largest z-score is 1.47.
4. **Deterministic arrivals (T7).** With $A_t \equiv 3$ the optimal trajectory from 0 is the cycle 3, 6, 9, 12, 15 and $V(0)$ is a geometric series that can be summed by hand: $(6\gamma + 12\gamma^2 + 18\gamma^3 + 24\gamma^4 + 100\gamma^5)/(1 - \gamma^5) = 569.753019$. The linear-solve value is 569.753019.
5. **Parameter sensitivity (T6, Figure 10).** The threshold rises with the dispatch cost (8, 11, 13, 14, 15 for $c_f$ = 25, 50, 100, 200, 400; never dispatch at 800), falls with the holding cost (15, 14, 13, 11, 8, 5 for $c_h$ = 0.5, 1, 2, 4, 8, 16), rises with capacity (5, 9, 13, 16, 17, 17 for $K$ = 5, 10, 15, 20, 30, 50) and needs patience: with $\gamma \le 0.5$ the agent never dispatches, because 100 now can never be recovered from at most 30 of holding cost per period, while $\gamma \ge 0.9$ gives threshold 13. Each direction matches the economic reading of the costs.
6. **Agreement and structure (T1, T2, T8, T10)** as described above: stochastic $P$, three identical policies, values equal to $10^{-9}$, exact $c_f$ gap at $s = 0$, insensitivity to the cap, monotone $V^*$ and threshold policy.
7. **First steps of a trajectory.** The table below follows the optimal policy from an empty station (seed 1). The station fills by 1 to 5 customers per period, dispatches as soon as 13 or more wait, and the shuttle leaves 0 customers behind in each of these dispatches.

| t | state | action | boarded | left waiting | cost | arrivals | next state |
|---|---|---|---|---|---|---|---|
| 0 | 0 | wait | 0 | 0 | 0 | 3 | 3 |
| 1 | 3 | wait | 0 | 3 | 6 | 5 | 8 |
| 2 | 8 | wait | 0 | 8 | 16 | 1 | 9 |
| 3 | 9 | wait | 0 | 9 | 18 | 5 | 14 |
| 4 | 14 | dispatch | 14 | 0 | 100 | 2 | 2 |
| 5 | 2 | wait | 0 | 2 | 4 | 3 | 5 |
| 6 | 5 | wait | 0 | 5 | 10 | 5 | 10 |
| 7 | 10 | wait | 0 | 10 | 20 | 3 | 13 |
| 8 | 13 | dispatch | 13 | 0 | 100 | 3 | 3 |
| 9 | 3 | wait | 0 | 3 | 6 | 1 | 4 |
| 10 | 4 | wait | 0 | 4 | 8 | 4 | 8 |
| 11 | 8 | wait | 0 | 8 | 16 | 3 | 11 |

![Test T3: error of the finite-horizon value at t = 0 against the infinite-horizon value.](../results/shuttle/fig_val_horizon_error.png){ width=68% }

![Test T4: exact cost of every threshold policy from an empty station.](../results/shuttle/fig_val_threshold_enumeration.png){ width=68% }

![Test T6: optimal threshold as one parameter varies.](../results/shuttle/fig_val_sensitivity.png){ width=80% }

## With three more months

- Solve the problem with a batch arrival distribution estimated from data (for example a Poisson or an empirical histogram) and with time-varying arrival rates, and test whether the threshold structure survives.
- Verify the structural result analytically: prove that $V^*$ is convex and that the optimal policy is a threshold, then test the solver on random instances against the proof's conditions.
- Compare against average-cost formulations and against a two-shuttle version (state = customers waiting plus shuttles available), where the optimal policy is no longer a single threshold.
- Stress-test numerical settings: tolerances, $T$, and larger caps, with timing, to document how the sweep count of value iteration grows as $\gamma \to 1$ and how policy iteration stays at a handful of iterations.
- Test the robustness of the threshold policy when the true arrival distribution differs from the modelled one (simulate the policy under misspecified arrivals and measure the cost increase).

# Part 3, Policy Gradient Part 1: REINFORCE

## Algorithm

REINFORCE (Williams, 1992) is implemented in PyTorch in `policy_gradient/cartpole_reinforce.py` and `policy_gradient/pong_reinforce.py`. One iteration does the following.

1. Run the current stochastic policy $\pi_\theta$ until the environment terminates or truncates, sampling $a_t \sim \pi_\theta(\cdot \mid s_t)$ from the softmax over the network's logits.
2. Compute the discounted return of every step by the backward recursion $G_t = r_t + \gamma\, G_{t+1}$, $G_T = 0$, so $G_t = \sum_{k \ge t} \gamma^{k-t} r_k$. This is the objective the assignment asks for: the discounted reward over the episode.
3. Take one Adam step on the surrogate loss $L(\theta) = -\sum_t \log \pi_\theta(a_t \mid s_t)\,(G_t - b_t)$, whose gradient is the REINFORCE estimate $\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)(G_t - b_t)$. In Part 1 the baseline is $b_t = 0$; Part 2 uses the baselines described in that section.

**CartPole-v1.** The policy is a 4-128-2 network with ReLU. $\gamma = 0.95$, Adam with learning rate 0.01 as prescribed, one update per episode, 2000 training episodes; smaller learning rates (0.003 and 0.001, 3000 episodes) and a length-normalised loss ($L/T$) were also run because the prescribed setting is unstable (next section). Log-probabilities come from `log_softmax`, which avoids $\log 0$ when the policy becomes nearly deterministic.

**Pong-v5.** `ALE/Pong-v5` from gymnasium 1.4 with ale-py 0.12 and the version's defaults: frame skip 4, sticky actions with probability 0.25, full 6-action space exposed but only RIGHT (2) and LEFT (3) are ever sent (`ACTIONS = [2, 3]`; a counter in the training loop confirms this at the end of every run). Every frame goes through the assignment's `preprocess()` (crop, 2x downsample, background removal, binarise to 80 x 80); the only change is an explicit copy before the in-place operations, because the vector environment hands out views of a shared buffer. The network input is the difference between the current and the previous preprocessed frame (6400 values in $\{-1, 0, 1\}$; the previous frame is all zeros at the first step), which encodes the motion of the ball and paddles that a single frame does not show. The policy is a 6400-200-2 network with ReLU and a softmax over {RIGHT, LEFT}. $\gamma = 0.99$. Four environments run in parallel (`gymnasium.vector.AsyncVectorEnv`); each finished episode is appended to a batch and one Adam step is taken per 10 episodes on the loss averaged over all steps in the batch, with learning rate $10^{-3}$. Trajectories still in flight at update time keep running under the new parameters; their log-probabilities are recomputed at update time, so the gradient is always evaluated at the current parameters, and the resulting slight off-policy error over at most one update is accepted. The prescribed 0.01 was not used: with a 6400-input layer an Adam step of 0.01 per weight equals the initial weight scale (about 0.0125), and the comparison in the Pong section shows that $10^{-3}$ already learns faster than $3 \times 10^{-4}$ and $10^{-4}$. Training is checkpointed every 5 updates and can be resumed. Inputs are stored as int8 and the update processes chunks of 4096 steps, so a batch of 10 long episodes fits in a few hundred MB.

Each run logs every episode (`episodes.csv`) and every update (`updates.csv`: loss, entropy, gradient norm, standard deviation of $G_t$ and of the advantage), and the plots below are produced from these logs by `make_plots_cartpole.py` and `make_plots_pong.py`. The moving average in every training plot is a trailing 100-episode mean (fewer episodes at the start). After training, 500 episodes are rolled out with the trained network, both with sampled actions (the policy as trained) and with greedy actions; the histograms show the sampled roll-outs and the greedy numbers are given in the text.

## CartPole-v1

CARTPOLE_PART1_PLACEHOLDER

## Pong-v5

PONG_PART1_PLACEHOLDER

## Edge cases (Part 1)

PG1_EDGE_PLACEHOLDER

## Tests that validate correctness (Part 1)

PG1_TESTS_PLACEHOLDER

## With three more months (Part 1)

PG1_MONTHS_PLACEHOLDER

# Policy Gradient Part 2: REINFORCE with a baseline

PG2_PLACEHOLDER
