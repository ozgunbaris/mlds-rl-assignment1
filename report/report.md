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

**CartPole-v1.**

- *Truncation at 500 steps.* CartPole-v1 ends with `truncated = True` after 500 steps. REINFORCE treats that step as the end of the episode and does not bootstrap, so the returns of the last steps before a truncation are smaller than the infinite-horizon values. With $\gamma = 0.95$ a return 60 steps before the cut is within $0.95^{60} \cdot 20 = 0.9$ of its untruncated value, so only the tail of a perfect episode is affected, and all those returns stay positive. The roll-out statistics count a 500-step episode as reward 500.
- *All rewards are +1.* Without a baseline every $G_t$ is positive, so every sampled action is reinforced; learning only comes from differences between the returns of different steps and episodes. This is the main source of variance and is quantified in Part 2.
- *Short early episodes and the return scale.* The first episodes last 8 to 30 steps and $G_t \le 1/(1-\gamma) = 20$ for every $t$; a 20-step survival already gives $G_0 = 12.8$ and a 500-step one gives 20.0 (validation V5). The learning signal therefore only distinguishes the next 20 to 60 steps, which is enough for balancing but makes the objective nearly flat once episodes are long.
- *Growing gradient scale and collapse.* The episode gradient is a sum over $T$ steps with returns up to 20, so its scale grows by a factor of about 25 between a 20-step and a 500-step episode. Adam normalises by a running root-mean-square with a memory of about 1000 updates, so when episodes suddenly become long the steps become too large. The policy then becomes deterministic ($\text{entropy} \to 0$; the gradient $\nabla \log \pi(a|s) \to 0$ when $\pi(a|s) \to 1$), and a policy that collapsed onto a bad action cannot recover because its gradient vanishes. This is observed in 3 of 5 seeds at learning rate 0.01 (final roll-out mean 9.4, always pushing to one side) and motivates the smaller learning rate and the length-normalised loss reported below.
- *Numerical safety.* Log-probabilities are computed with `log_softmax` and actions are sampled from `Categorical(logits=...)`, so a near-deterministic policy never produces $\log 0$ or a NaN gradient; a zero advantage gives an exactly zero gradient (unit test).
- *Seed dependence.* Results differ a lot between seeds, so every configuration was run with 5 seeds and the comparison plots show the mean and standard deviation across seeds rather than one run.

**Pong-v5.**

- *Automatic reset inside the vector environment.* With gymnasium 1.4 the step after a terminal step returns the first observation of the next game, ignores the action and gives reward 0. That step must not be recorded as a transition and the difference frame must restart from an all-zero previous frame; the `FrameTracker` class handles it, a unit test exercises the sequence (terminal step, reset step, first real step), and the training loop asserts after every episode that its own episode reward and length equal the ones reported by gymnasium's `RecordEpisodeStatistics` wrapper (no mismatch occurred in about N_PONG_EPISODES episodes).
- *First frame of a game.* There is no previous frame, so the input is the first frame itself; this is the only input that shows the static paddles, and a test checks it.
- *In-place preprocessing.* The assignment's `preprocess()` edits its input in place; applied to the shared observation buffer of the vector environment it would corrupt later observations, so the function first copies the crop. A test checks that the frame is unchanged after preprocessing.
- *Two of six actions.* Only indices 0 and 1 of the policy exist, mapped to environment actions 2 (RIGHT, measured to move the paddle up) and 3 (LEFT, down). Pong serves automatically, so FIRE is not needed; the action counter at the end of each run confirms that no other action was sent, and every recorded game reached 21 points for one side.
- *Sticky actions.* `Pong-v5` repeats the previous action with probability 0.25 regardless of the agent's choice. The recorded action is the one the agent sampled, which is the correct quantity for the policy gradient; the repetition is part of the environment's transition noise and makes the game stochastic even for a fixed seed.
- *Sparse rewards and long episodes.* Rewards are $\pm 1$ at the 21 to 41 points of a game and 0 elsewhere; a game lasts from about 800 steps (losing 21-0 quickly) to several thousand steps once rallies get longer. $G_t$ is computed over the whole episode as required, but with $\gamma = 0.99$ a point's reward is discounted by 0.37 after 100 steps, so each return is dominated by the next point (validation P4 shows the sawtooth). Inputs are stored as int8 and the update works in chunks, so a batch of 10 long games fits in memory.
- *In-flight trajectories at update time.* With four parallel games, up to three games are unfinished when the batch of 10 completes; they continue under the updated policy and are used with log-probabilities recomputed at their own update, so the gradient is always taken at the current parameters and the data is at most one update old.
- *Frame limit.* The environment also truncates a game after 108,000 frames (27,000 steps); this never happened (the longest recorded game had N_PONG_MAXLEN steps) and would be treated like a terminal step.

## Tests that validate correctness (Part 1)

PG1_TESTS_PLACEHOLDER

## With three more months (Part 1)

- Run every Pong configuration with at least 5 seeds (only one seed per configuration fit in the time available) and report confidence bands on the learning curves and on the time to reach a given score.
- Validate the gradient estimator on a problem with a known exact gradient: a two-state bandit or a small gridworld where $\nabla_\theta J(\theta)$ can be computed from the MDP, and compare the REINFORCE estimate (same code path) with the exact gradient and with a finite-difference estimate of $J$.
- Compare against a reference implementation (for example CleanRL) on CartPole with identical seeds and hyperparameters; identical learning curves would rule out implementation errors that only show up as slower learning.
- Study the CartPole collapse in detail: log Adam's second-moment statistics and the parameter change per update, then test gradient clipping, learning-rate decay and an entropy bonus, and map the region of learning rates and loss scalings where the collapse disappears.
- Evaluate during training with a fixed set of evaluation seeds and greedy actions, so that the reported curve measures the policy rather than the exploration noise.
- Pong ablations: difference frame vs. a stack of 2 to 4 frames, a small convolutional policy, $\gamma \in \{0.95, 0.99, 0.999\}$, batch sizes 1 to 40 episodes, restarting the return at every point (Karpathy's trick) vs. the full-episode return, and sticky actions off, to measure how much of the roll-out spread is environment noise.
- Robustness tests of the trained Pong policy: horizontally mirrored frames, random no-op starts, a different frame skip, and an opponent with a different difficulty setting.
- Train Pong until the agent wins consistently (score above 0) and record the number of frames needed, then profile and parallelise the collection (more environments, GPU for the update) since compute was the binding constraint here.

# Policy Gradient Part 2: REINFORCE with a baseline

## Baselines used

Two baselines are implemented in the same scripts (`--baseline mean` and `--baseline value`); both leave the gradient unbiased because neither depends on the action taken at step $t$.

- **Constant baseline** (`mean`): $b$ is an exponential moving average of the mean discounted return. CartPole updates it once per episode with rate 0.05 (initialised with the first episode's mean $G_t$); Pong updates it once per batch of 10 episodes with rate 0.1.
- **Learned state-value baseline** (`value`, the main choice for Part 2): a second network $V_\phi(s)$ with the same architecture as the policy but one output (4-128-1 for CartPole, 6400-200-1 for Pong). The advantage is $A_t = G_t - V_\phi(s_t)$ with $V_\phi$ detached, computed with the parameters $\phi$ from *before* the current update so that the baseline is a function of the state only. After the policy step, $\phi$ is fitted to the Monte Carlo targets $G_t$ by least squares: 5 Adam steps (learning rate equal to the policy's) on the episode for CartPole, and 2 passes over the batch in minibatches of 2048 steps with learning rate $10^{-3}$ for Pong. The policy update is otherwise identical to Part 1, including the learning rates.

The learned baseline removes the part of $G_t$ that is predictable from the state (how long the pole is likely to survive from here; whether the ball is about to pass the paddle), so the remaining advantage reflects the effect of the action. The constant baseline only removes the common level of the returns. Both reductions are measured directly in the tests below.

## CartPole-v1 with a baseline

PG2_CARTPOLE_PLACEHOLDER

## Pong-v5 with a baseline

PG2_PONG_PLACEHOLDER

## Edge cases (Part 2)

PG2_EDGE_PLACEHOLDER

## Tests that validate correctness (Part 2)

PG2_TESTS_PLACEHOLDER

## With three more months (Part 2)

- Measure the bias of the value baseline when the value network is fitted on the same episode that is used in the update (the implementation avoids this by using the pre-update network, but the alternative is common and its bias could be quantified by comparing the mean gradients over many episodes).
- Compare the variance and sample efficiency of the two baselines with reward-to-go only, with per-time-step baselines $b_t$, with normalised advantages, with generalised advantage estimation at several $\lambda$, and with a bootstrapped critic (actor-critic), all on the same seeds and budgets.
- Vary the capacity and the number of fitting steps of the value network to map how under- and over-fitting of $V_\phi$ changes the variance reduction and the final score.
- Run 5 or more seeds for Pong with and without the baseline and test the difference in the number of episodes to reach -15, -10 and 0 with a paired test across seeds.
- Try a shared trunk for policy and value on Pong and measure whether the value loss gradients help or hurt the policy features.
