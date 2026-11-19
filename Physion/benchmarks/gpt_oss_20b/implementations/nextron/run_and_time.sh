#!/bin/bash
# One gpt_oss_20b run on one 8x H100 node with NexTron (closed source, in its container image). Trains from the
# initialization in ./models (see README) until the eval loss on the first 1,024 validation samples is <= 3.34, then
# stops. Seed 1234 (data order), as in the run reported.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p output
docker run --rm --gpus all --ipc=host --network=host --ulimit memlock=-1 --shm-size=64g \
  -v "$PWD":/workspace -w /workspace "${NEXTRON_IMAGE:?NexTron image}" \
  nextron sft --config /workspace/config_1x8_h100.yaml
