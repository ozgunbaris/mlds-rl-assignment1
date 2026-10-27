# MLDS Reinforcement Learning, Assignment 1

Solutions for the three parts of the assignment:

| Part | Task | Code | Results |
|---|---|---|---|
| 2 | Shuttle dispatch MDP: enumeration (T = 500), value iteration, policy iteration | `shuttle_mdp/` | `results/shuttle/` |
| 3 (Part 1) | REINFORCE on CartPole-v1 and Pong-v5 (PyTorch) | `policy_gradient/` | `results/cartpole/`, `results/pong/` |
| 3 (Part 2) | REINFORCE with a baseline (constant mean-return baseline and learned state-value baseline) | same scripts, `--baseline mean|value` | same folders |

The written report is `report/report.pdf` (source `report/report.md`; rebuild with `report/build.sh`, which needs pandoc and xelatex).

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

### Pong on the Deepdish cluster (Nomad)

The seeded Pong experiments (5 seeds x {no baseline, constant baseline, value baseline} at lr 1e-3, and 5 seeds each at lr 0.01 and 3e-4 without baseline) ran as Nomad batch jobs on the lab cluster, CPU only. `scripts/nomad/make_job.sh` generates a job file that embeds the code as a base64 tarball, installs the dependencies in a `python:3.12-slim` container, runs the unit tests, trains all runs in parallel for a fixed number of hours, rolls out 500 games per run (sampled and greedy actions) and packs the results:

```bash
HOURS=4.0 BASELINES="none mean value" CORES=36 MEMORY=40960 JOB_NAME=pong-reinforce NODE=mlds-deepdish5 LR=1e-3 PREFIX=cluster scripts/nomad/make_job.sh
HOURS=3.3 BASELINES="none" CORES=12 MEMORY=12288 JOB_NAME=pong-lr1e-2 NODE=mlds-deepdish5 LR=1e-2 PREFIX=cluster_lr1e-2 scripts/nomad/make_job.sh
HOURS=3.3 BASELINES="none" CORES=12 MEMORY=12288 JOB_NAME=pong-lr3e-4 NODE=mlds-deepdish2 LR=3e-4 PREFIX=cluster_lr3e-4 scripts/nomad/make_job.sh
# on the Nomad server: nomad job validate <file>; nomad job plan <file>; nomad job run -detach <file>
scripts/fetch_cluster_results.sh <alloc-id> <job-name>     # copies results/pong/<prefix>_* and the job logs back
scripts/check_runs.sh                                       # one-shot status of local and cluster runs
```

The generated job files used for the report are in `scripts/nomad/*.nomad`; the fetched runs are `results/pong/cluster_*` (same layout as the local runs: `episodes.csv`, `updates.csv`, `eval_summary_*.json`, `eval_rewards_*.npy`, `policy_weights.pt`, `train.log`). `scripts/finalize_pong_local.sh` and `scripts/finalize_pong_plots.sh` produce the local roll-outs, all Pong figures (`results/pong/figures`) and the validation experiments.

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
scripts/                          experiment loops, Nomad job generator, cluster fetch and status scripts
results/                          logs (episodes.csv, updates.csv), policy weights, figures, summaries, validation outputs
report/                           report.md, build.py (figure numbering + pandoc), report.pdf
```
