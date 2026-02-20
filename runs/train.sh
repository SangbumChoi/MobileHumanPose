#!/bin/bash
# Train MobileHumanPose. Edit src/config.py for backbone, dataset.
# Single GPU:  python -m src.train
# Multi-GPU:   torchrun (see below)

cd "$(dirname "$0")/.."

# Default: 2 GPUs. Set NPROC=1 for single GPU, NPROC=8 for 8 GPUs.
NPROC=${NPROC:-2}

if [ "$NPROC" -eq 1 ]; then
  python -m src.train "$@"
else
  torchrun --standalone --nproc_per_node=$NPROC -m src.train "$@"
fi
