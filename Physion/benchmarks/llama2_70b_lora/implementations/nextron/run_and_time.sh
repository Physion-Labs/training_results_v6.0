#!/bin/bash
# One scored llama2_70b_lora run on one 8x H100 node with NexTron (closed source, in its container image).
# Drops the page cache, draws a fresh seed from /dev/urandom, trains until eval loss <= 0.925; the MLPerf log is
# written to ./output/mllog.txt (run_start / run_stop bound the scored time).
set -euo pipefail
cd "$(dirname "$0")"
sync && echo 3 > /proc/sys/vm/drop_caches                     # needs root on the host
SEED=$(( $(od -An -N4 -tu4 /dev/urandom) % 2147483647 ))
python3 -c "import json; c=json.load(open('config_1x8_h100.json')); c['training']['seed']=$SEED; json.dump(c, open('config.run.json','w'), indent=2)"
mkdir -p output
docker run --rm --gpus all --ipc=host --network=host --ulimit memlock=-1 --shm-size=64g \
  -v "$PWD":/workspace -w /workspace "${NEXTRON_IMAGE:?NexTron image}" \
  nextron sft --config /workspace/config.run.json
