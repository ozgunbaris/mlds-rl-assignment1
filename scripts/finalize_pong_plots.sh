#!/usr/bin/env bash
# Figures, summaries and validation experiments for all Pong runs (local and cluster).
set -e
cd "$(dirname "$0")/.."
.venv/bin/python -m policy_gradient.make_plots_pong \
  --runs results/pong/none_lr1e-3 results/pong/value_lr1e-3 results/pong/none_lr3e-4 results/pong/value_lr3e-4 results/pong/none_lr1e-4 \
  --labels "no baseline, lr 1e-3 (local)" "value baseline, lr 1e-3 (local)" "no baseline, lr 3e-4 (stopped)" "value baseline, lr 3e-4 (stopped)" "no baseline, lr 1e-4 (stopped)" \
  --diag-runs 2 --out results/pong/figures \
  --group "no baseline=results/pong/cluster_none_seed*" \
  --group "constant baseline=results/pong/cluster_mean_seed*" \
  --group "value baseline=results/pong/cluster_value_seed*" \
  --group-name seed_groups --group-title "lr 1e-3, 5 seeds per variant" > results/pong/figures/make_plots.log 2>&1
tail -3 results/pong/figures/make_plots.log
# learning-rate study (no baseline): the 1e-3 group comes from the main cluster job
.venv/bin/python -m policy_gradient.make_plots_pong --runs results/pong/none_lr1e-3 --labels "no baseline, lr 1e-3 (local)" \
  --out results/pong/figures_lr \
  --group "lr 0.01=results/pong/cluster_lr1e-2_none_seed*" \
  --group "lr 1e-3=results/pong/cluster_none_seed*" \
  --group "lr 3e-4=results/pong/cluster_lr3e-4_none_seed*" \
  --group-name lr_groups --group-title "no baseline, 5 seeds per learning rate" > results/pong/figures_lr/make_plots.log 2>&1 || true
mkdir -p results/pong/figures_lr; cp -f results/pong/figures_lr/compare_lr_groups.png results/pong/figures_lr/lr_groups.json results/pong/figures/ 2>/dev/null || true
[ "${SKIP_VALIDATION:-0}" = 1 ] || OMP_NUM_THREADS=1 .venv/bin/python -m policy_gradient.validate_pong --policy results/pong/none_lr1e-3/checkpoint.pt \
  --value-ckpt results/pong/value_lr1e-3/checkpoint.pt --out results/pong/validation 2>&1 | grep -v "A.L.E\|Powered"
echo PLOTS_AND_VALIDATION_DONE
