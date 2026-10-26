#!/usr/bin/env bash
# One-shot status of the local runs and of the three Nomad jobs (one ssh command, no retries).
cd "$(dirname "$0")/.."
echo "=== LOCAL $(date '+%H:%M') | pong mains: $(pgrep -f 'pong_reinforce.*--out results/pong' | wc -l | tr -d ' ') | cartpole: $(pgrep -f cartpole_reinforce | wc -l | tr -d ' ')"
for D in none_lr1e-3 value_lr1e-3; do echo "$D: $(grep '^upd' results/pong/$D/train.log | tail -1 | cut -c1-118)"; done
ssh -o ConnectTimeout=10 -o BatchMode=yes obkir@kranj.iems.private '
for J in pong-reinforce pong-lr1e-2 pong-lr3e-4; do
  A=$(nomad job status $J 2>/dev/null | sed -n "/^Allocations/,/^$/p" | awk "NR==3{print \$1}")
  echo "=== $J alloc $A: $(nomad alloc status $A 2>&1 | grep -E "Task .run. is" | head -1)"
  nomad alloc logs $A run 2>&1 | grep -E "^results/pong|TRAINING_DONE|EVAL_|^END|PACKED" | tail -17 | cut -c1-118
  E=$(nomad alloc logs -stderr $A run 2>&1 | grep -v "A.L.E\|Powered by\|^$" | tail -2); if [ -n "$E" ]; then echo "stderr: $E"; fi
done'
