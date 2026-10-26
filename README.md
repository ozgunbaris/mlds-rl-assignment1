# MLDS Reinforcement Learning, Assignment 1

Solutions for the three parts of the assignment:

| Part | Task | Code | Results |
|---|---|---|---|
| 2 | Shuttle dispatch MDP: enumeration (T = 500), value iteration, policy iteration | `shuttle_mdp/` | `results/shuttle/` |
| 3 (Part 1) | REINFORCE on CartPole-v1 and Pong-v5 (PyTorch) | `policy_gradient/` | `results/cartpole/`, `results/pong/` |
| 3 (Part 2) | REINFORCE with a baseline (constant mean-return baseline and learned state-value baseline) | same scripts, `--baseline mean|value` | same folders |

The written report is `report/report.pdf` (source `report/report.md`).

## Setup

Python 3.12 was used. All dependencies are in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`gymnasium[atari]` + `ale-py` provide `ALE/Pong-v5` (the ROMs ship with `ale-py >= 0.9`). Every command below is run from the repository root.

## Part 2: shuttle dispatch MDP

```bash
python -m shuttle_mdp.run_shuttle --out results/shuttle        # solves (a), (b), (c), writes figures + summary.json
python -m shuttle_mdp.validate_shuttle --out results/shuttle   # validation experiments T1-T10, writes validation.md/json
python -m pytest tests/test_shuttle.py -q                      # 14 unit/property tests
```

Figures: `fig_a_enumeration_value_T500.png`, `fig_b_value_iteration_value.png`, `fig_c_policy_iteration_policy.png` are the three required plots; the other `fig_*` files are diagnostics described in the report.

## Part 3: policy gradient

### CartPole-v1

One run trains for a fixed number of episodes, then rolls out 500 episodes with sampled actions and 500 with greedy actions:

```bash
python -m policy_gradient.cartpole_reinforce --out results/cartpole/none_seed0  --baseline none  --lr 0.01 --seed 0   # Part 1
python -m policy_gradient.cartpole_reinforce --out results/cartpole/value_seed0 --baseline value --lr 0.01 --seed 0   # Part 2
```

Options: `--baseline {none,mean,value}`, `--lr`, `--gamma` (0.95), `--episodes` (2000), `--loss-norm {sum,mean}`, `--seed`. The experiments in the report (3 baselines x {lr 0.01, 0.003, 0.001} x 5 seeds, plus the length-normalised loss at lr 0.01) were produced with the loops in `scripts/run_cartpole_sweeps.sh`.

```bash
python -m policy_gradient.make_plots_cartpole --results results/cartpole --out results/cartpole/figures   # curves, histograms, summary tables
python -m policy_gradient.validate_cartpole --results results/cartpole --out results/cartpole/validation  # validation experiments V1-V6
```

### Pong-v5

Training runs several environments in parallel and checkpoints every 5 updates; it can be stopped (Ctrl-C) and resumed with `--resume`.

```bash
python -m policy_gradient.pong_reinforce --out results/pong/none_lr1e-3  --baseline none  --lr 1e-3 --num-envs 4 --batch-episodes 10 --time-limit-hours 5   # Part 1
python -m policy_gradient.pong_reinforce --out results/pong/value_lr1e-3 --baseline value --lr 1e-3 --num-envs 4 --batch-episodes 10 --time-limit-hours 5   # Part 2
```

Roll out 500 episodes with a trained checkpoint, make the figures, run the validation experiments:

```bash
python -m policy_gradient.pong_reinforce --eval results/pong/none_lr1e-3/checkpoint.pt --eval-episodes 500 --out results/pong/none_lr1e-3
python -m policy_gradient.pong_reinforce --eval results/pong/none_lr1e-3/checkpoint.pt --eval-episodes 500 --out results/pong/none_lr1e-3 --greedy
python -m policy_gradient.make_plots_pong --runs results/pong/none_lr1e-3 results/pong/value_lr1e-3 --out results/pong/figures
python -m policy_gradient.validate_pong --policy results/pong/none_lr1e-3/checkpoint.pt --value-ckpt results/pong/value_lr1e-3/checkpoint.pt --out results/pong/validation
```

### Unit tests for the policy-gradient code

```bash
python -m pytest tests -q
```

`tests/test_policy_gradient.py` checks the discounted-return recursion against a brute-force double loop, that one policy-gradient step raises the probability of a positively-rewarded action, that a zero advantage gives a zero gradient, the preprocessing function (shape, binary values, no in-place modification of the frame), that only actions 2 (RIGHT) and 3 (LEFT) are sent to Pong, the difference-frame bookkeeping across the vector environment's automatic reset, and the structure of a complete Pong game (one side reaches 21, rewards in {-1, 0, 1}).

## Repository layout

```
shuttle_mdp/shuttle.py            MDP model, enumeration, value iteration, policy iteration, simulation
shuttle_mdp/run_shuttle.py        required figures + summary
shuttle_mdp/validate_shuttle.py   validation experiments
policy_gradient/common.py         discounted returns, moving average, CSV logging
policy_gradient/cartpole_reinforce.py
policy_gradient/pong_reinforce.py
policy_gradient/make_plots_*.py   figures and summary tables
policy_gradient/validate_*.py     validation experiments
viz.py                            plotting style
tests/                            pytest suites
results/                          logs (episodes.csv, updates.csv), checkpoints, figures, summaries
report/                           report.md and report.pdf
```
