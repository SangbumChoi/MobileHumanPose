#!/bin/bash
# Train MobileHumanPose. Edit main/config.py for backbone, dataset.
# Single GPU:  python main/train.py
# Multi-GPU:   torchrun (see below)

cd "$(dirname "$0")/.."

# Default: 2 GPUs. Set NPROC=1 for single GPU, NPROC=8 for 8 GPUs.
NPROC=${NPROC:-2}

cd main
if [ "$NPROC" -eq 1 ]; then
  python train.py "$@"
else
  torchrun --standalone --nproc_per_node=$NPROC train.py "$@"
fi
