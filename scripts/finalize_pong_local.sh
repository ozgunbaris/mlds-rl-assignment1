#!/usr/bin/env bash
# After the two local Pong runs finish: 500-episode roll-outs (sampled and greedy), weight export.
set -e
cd "$(dirname "$0")/.."
for R in none_lr1e-3 value_lr1e-3; do
  OMP_NUM_THREADS=1 .venv/bin/python -m policy_gradient.pong_reinforce --eval results/pong/$R/checkpoint.pt --eval-episodes 500 --num-envs 4 --out results/pong/$R > results/pong/$R/eval_sampled.log 2>&1 &
  OMP_NUM_THREADS=1 .venv/bin/python -m policy_gradient.pong_reinforce --eval results/pong/$R/checkpoint.pt --eval-episodes 500 --num-envs 4 --out results/pong/$R --greedy > results/pong/$R/eval_greedy.log 2>&1 &
done
wait
for R in none_lr1e-3 value_lr1e-3 none_lr3e-4 value_lr3e-4 none_lr1e-4; do
  .venv/bin/python -m policy_gradient.export_weights results/pong/$R/checkpoint.pt results/pong/$R/policy_weights.pt
done
grep -h '"mean"\|"std"\|"mode"' results/pong/none_lr1e-3/eval_summary_*.json results/pong/value_lr1e-3/eval_summary_*.json
echo LOCAL_EVAL_DONE
