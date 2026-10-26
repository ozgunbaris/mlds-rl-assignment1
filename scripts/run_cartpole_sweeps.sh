#!/usr/bin/env bash
# CartPole experiments used in the report. Run from the repository root with the venv active.
set -e
# prescribed setting: Adam, lr 0.01, gamma 0.95, plain REINFORCE episode gradient (sum over steps)
for B in none mean value; do for SEED in 0 1 2 3 4; do
  python -m policy_gradient.cartpole_reinforce --out results/cartpole/${B}_seed$SEED --baseline $B --seed $SEED --episodes 2000 --eval-episodes 500
done; done
# smaller learning rates
for LR in 0.003 0.001; do for B in none mean value; do for SEED in 0 1 2 3 4; do
  python -m policy_gradient.cartpole_reinforce --out results/cartpole/${B}_lr${LR}_seed$SEED --baseline $B --lr $LR --value-lr $LR --seed $SEED --episodes 3000 --eval-episodes 500
done; done; done
# lr 0.01 with the episode gradient divided by the episode length
for B in none mean value; do for SEED in 0 1 2 3 4; do
  python -m policy_gradient.cartpole_reinforce --out results/cartpole/${B}_lr0.01_meanloss_seed$SEED --baseline $B --lr 0.01 --loss-norm mean --seed $SEED --episodes 2000 --eval-episodes 500
done; done
python -m policy_gradient.make_plots_cartpole --results results/cartpole --out results/cartpole/figures
python -m policy_gradient.validate_cartpole --results results/cartpole --out results/cartpole/validation
