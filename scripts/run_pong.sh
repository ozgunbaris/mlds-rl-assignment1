#!/usr/bin/env bash
# Pong experiments used in the report (each training run took about 5 hours on a 10-core laptop CPU).
set -e
python -m policy_gradient.pong_reinforce --out results/pong/none_lr1e-3  --baseline none  --lr 1e-3 --num-envs 4 --batch-episodes 10 --time-limit-hours 5.5 --seed 0
python -m policy_gradient.pong_reinforce --out results/pong/value_lr1e-3 --baseline value --lr 1e-3 --num-envs 4 --batch-episodes 10 --time-limit-hours 5.3 --seed 0
# learning-rate comparison runs (stopped early, see report)
python -m policy_gradient.pong_reinforce --out results/pong/none_lr3e-4  --baseline none  --lr 3e-4 --num-envs 4 --batch-episodes 10 --time-limit-hours 0.4 --seed 0
python -m policy_gradient.pong_reinforce --out results/pong/none_lr1e-4  --baseline none  --lr 1e-4 --num-envs 4 --batch-episodes 10 --time-limit-hours 0.15 --seed 0
python -m policy_gradient.pong_reinforce --out results/pong/value_lr3e-4 --baseline value --lr 3e-4 --num-envs 4 --batch-episodes 10 --time-limit-hours 0.25 --seed 0
for R in none_lr1e-3 value_lr1e-3; do
  python -m policy_gradient.pong_reinforce --eval results/pong/$R/checkpoint.pt --eval-episodes 500 --out results/pong/$R
  python -m policy_gradient.pong_reinforce --eval results/pong/$R/checkpoint.pt --eval-episodes 500 --out results/pong/$R --greedy
done
python -m policy_gradient.make_plots_pong --runs results/pong/none_lr1e-3 results/pong/value_lr1e-3 results/pong/none_lr3e-4 results/pong/value_lr3e-4 results/pong/none_lr1e-4 --out results/pong/figures
python -m policy_gradient.validate_pong --policy results/pong/none_lr1e-3/checkpoint.pt --value-ckpt results/pong/value_lr1e-3/checkpoint.pt --out results/pong/validation
