#!/bin/sh
# Runs inside the python:3.12-slim container on the cluster node. Plain POSIX sh.
# No dollar-brace or double-curly sequences (Nomad template rule).
set -u
BASE=/alloc/data
WORK=$BASE/work
LOGS=$BASE/logs
OUT=$BASE/pong_results.tar.gz
mkdir -p $WORK $LOGS

pack() {
  cd $BASE
  if [ -d $WORK/results ]; then
    tar -czf $OUT.tmp --exclude=checkpoint.pt --exclude=checkpoint.pt.tmp -C $WORK results -C $BASE logs && mv $OUT.tmp $OUT
  else
    tar -czf $OUT.tmp -C $BASE logs && mv $OUT.tmp $OUT
  fi
  echo "PACKED $(wc -c < $OUT) bytes $(date -u +%H:%M:%S)"
  cd $WORK
}
trap pack EXIT

CORES=$(python3 -c "import os; print(len(os.sched_getaffinity(0)))")
echo "BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) node=$NODE_NAME cores=$CORES hours=$HOURS memory.max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
base64 -d /local/code.b64 > $BASE/code.tar.gz
tar -xzf $BASE/code.tar.gz -C $WORK
cd $WORK
echo "pip torch (cpu) $(date -u +%H:%M:%S)"
pip install --no-cache-dir --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cpu > $LOGS/pip_torch.log 2>&1 || { tail -n 20 $LOGS/pip_torch.log; exit 1; }
echo "pip gymnasium/ale $(date -u +%H:%M:%S)"
pip install --no-cache-dir --disable-pip-version-check "numpy>=2,<3" "gymnasium[atari]==1.4.0" "ale-py==0.12.1" pytest > $LOGS/pip_other.log 2>&1 || { tail -n 20 $LOGS/pip_other.log; exit 1; }
python3 -c "import torch, gymnasium, ale_py; print('versions', torch.__version__, gymnasium.__version__, ale_py.__version__)"
python3 -m pytest -q tests/test_policy_gradient.py > $LOGS/pytest.log 2>&1
echo "pytest rc=$? $(tail -n 1 $LOGS/pytest.log)"

PIDS=""
for B in none value; do
  for S in 0 1 2 3 4; do
    D=results/pong/cluster_"$B"_seed"$S"
    mkdir -p $D
    python3 -m policy_gradient.pong_reinforce --out $D --baseline $B --lr 1e-3 --num-envs 4 --batch-episodes 10 --time-limit-hours $HOURS --seed $S --torch-threads 1 > $D/train.log 2>&1 &
    PIDS="$PIDS $!"
    echo "started $D pid $!"
  done
done

while true; do
  alive=0
  for p in $PIDS; do
    if kill -0 $p 2>/dev/null; then alive=1; fi
  done
  if [ $alive -eq 0 ]; then break; fi
  sleep 600
  for B in none value; do
    for S in 0 1 2 3 4; do
      D=results/pong/cluster_"$B"_seed"$S"
      echo "$D: $(grep '^upd' $D/train.log | tail -n 1)"
    done
  done
  pack
done
echo "TRAINING_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

for MODE in sampled greedy; do
  for B in none value; do
    for S in 0 1 2 3 4; do
      D=results/pong/cluster_"$B"_seed"$S"
      if [ $MODE = greedy ]; then G=--greedy; else G=""; fi
      python3 -m policy_gradient.pong_reinforce --eval $D/checkpoint.pt --eval-episodes 500 --num-envs 4 --out $D --seed $S $G > $D/eval_$MODE.log 2>&1 &
    done
  done
  wait
  echo "EVAL_$MODE done $(date -u +%H:%M:%S)"
done
for B in none value; do
  for S in 0 1 2 3 4; do
    D=results/pong/cluster_"$B"_seed"$S"
    python3 -m policy_gradient.export_weights $D/checkpoint.pt $D/policy_weights.pt
  done
done
pack
echo "END $(date -u +%Y-%m-%dT%H:%M:%SZ)"
