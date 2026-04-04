#!/usr/bin/env bash
# MPII Human Pose Dataset. Official: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/download
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p _download

MPII_IMAGES="https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz"
MPII_ANNOT="https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip"

echo "Downloading MPII images (~12.9 GB)..."
curl -L -o _download/mpii_human_pose_v1.tar.gz "$MPII_IMAGES" || { echo "Download failed"; exit 1; }
echo "Downloading MPII annotations..."
curl -L -o _download/mpii_human_pose_v1_u12_2.zip "$MPII_ANNOT" || { echo "Download failed"; exit 1; }

echo "Extracting images..."
tar -xzf _download/mpii_human_pose_v1.tar.gz -C .
echo "Extracting annotations..."
unzip -o _download/mpii_human_pose_v1_u12_2.zip -d annotations_mat
echo "Run: python download.py (or scripts/mpii_mat_to_coco.py) to generate annotations/train.json"
echo "Done. Remove _download/ to save space."
