#!/usr/bin/env bash
# Fetch the packed results of the Nomad job from the allocation and unpack them into results/pong/.
# Usage: scripts/fetch_cluster_results.sh <alloc-id>
set -e
cd "$(dirname "$0")/.."
ALLOC=${1:?alloc id}
TMP=$(mktemp -d)
ssh -o ConnectTimeout=10 -o BatchMode=yes obkir@kranj.iems.private "nomad alloc fs $ALLOC alloc/data/pong_results.tar.gz" > $TMP/pong_results.tar.gz
ls -la $TMP/pong_results.tar.gz
tar -xzf $TMP/pong_results.tar.gz -C $TMP
mkdir -p results/pong/cluster_logs
cp -R $TMP/results/pong/cluster_* results/pong/
cp -R $TMP/logs/. results/pong/cluster_logs/
ls -d results/pong/cluster_* | wc -l
echo FETCH_DONE
