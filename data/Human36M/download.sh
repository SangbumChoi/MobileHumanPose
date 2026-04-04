#!/usr/bin/env bash
# Human3.6M: official https://vision.imar.ro/human3.6m/ (login required).
# Optional: set H36M_GDRIVE_ID to a Drive folder ID for gdown.
set -e
cd "$(dirname "$0")"
python3 download.py
